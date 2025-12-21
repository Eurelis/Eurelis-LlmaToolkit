from typing import Any, Dict

from llama_index.core.schema import TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryResult,
    VectorStoreQueryMode,
)
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    metadata_dict_to_node,
)

from llama_index.vector_stores.mongodb.pipelines import (
    fulltext_search_stage,
    vector_search_stage,
    combine_pipelines,
    reciprocal_rank_stage,
    final_hybrid_stage,
)

from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
)
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch
from llama_index.vector_stores.mongodb.pipelines import map_lc_mql_filter_operators
import logging

logger = logging.getLogger(__name__)


class CustomMongoDBAtlasVectorSearch(MongoDBAtlasVectorSearch):
    @staticmethod
    def filters_to_mql(
        filters: MetadataFilters, metadata_key: str = "metadata"
    ) -> Dict[str, Any]:
        """This method is an alternative to the filters_to_mql method in the llama_index package with the support of recursive filters.

        Converts Llama-index's MetadataFilters into the MQL expected by $vectorSearch query.

        We are looking for something like

        "filter": {
                "$and": [
                    { "metadata.genres": { "$eq": "Comedy" } },
                    { "metadata.year": { "$gt": 2010 } }
                ]
        },

        See: See https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/#atlas-vector-search-pre-filter

        Args:
            filters: MetadataFilters object
            metadata_key: The key under which metadata is stored in the document

        Returns:
            MQL version of the filter.
        """
        if filters is None:
            return {}

        def prepare_key(key: str) -> str:
            # Prepares the key by prefixing it with the metadata key
            return (
                f"{metadata_key}.{key}"
                if not key.startswith(f"{metadata_key}.")
                else key
            )

        def process_filter(filter: MetadataFilter) -> Dict[str, Any]:
            # Processes a single filter
            return {
                prepare_key(filter.key): {
                    map_lc_mql_filter_operators(filter.operator): filter.value
                }
            }

        condition = f"${filters.condition.value}"
        filter_list = [
            (
                process_filter(mf)
                if isinstance(mf, MetadataFilter)
                else CustomMongoDBAtlasVectorSearch.filters_to_mql(mf, metadata_key)
            )
            for mf in filters.filters
        ]

        if len(filter_list) == 1:
            return filter_list[0]

        return {condition: filter_list}

    #  On reproduit la méthode _query de la classe MongoDBAtlasVectorSearch en utilisant la méthode filters_to_mql modifiée avec self.filters_to_mql
    def _query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        hybrid_top_k = query.hybrid_top_k or query.similarity_top_k
        sparse_top_k = query.sparse_top_k or query.similarity_top_k
        dense_top_k = query.similarity_top_k

        if query.mode == VectorStoreQueryMode.DEFAULT:
            if not query.query_embedding:
                raise ValueError("query_embedding in VectorStoreQueryMode.DEFAULT")
            # Atlas Vector Search, potentially with filter
            logger.debug(f"Running {query.mode} mode query pipeline")
            filter = self.filters_to_mql(query.filters, metadata_key=self._metadata_key)
            pipeline = [
                vector_search_stage(
                    query_vector=query.query_embedding,
                    search_field=self._embedding_key,
                    index_name=self._vector_index_name,
                    limit=dense_top_k,
                    filter=filter,
                    oversampling_factor=self._oversampling_factor,
                ),
                {"$set": {"score": {"$meta": "vectorSearchScore"}}},
            ]

        elif query.mode == VectorStoreQueryMode.TEXT_SEARCH:
            # Atlas Full-Text Search, potentially with filter
            if not query.query_str:
                raise ValueError("query_str in VectorStoreQueryMode.TEXT_SEARCH ")
            logger.debug(f"Running {query.mode} mode query pipeline")
            filter = self.filters_to_mql(query.filters, metadata_key=self._metadata_key)
            pipeline = fulltext_search_stage(
                query=query.query_str,
                search_field=self._text_key,
                index_name=self._fulltext_index_name,
                operator="text",
                filter=filter,
                limit=sparse_top_k,
            )
            pipeline.append({"$set": {"score": {"$meta": "searchScore"}}})

        elif query.mode == VectorStoreQueryMode.HYBRID:
            # Combines Vector and Full-Text searches with Reciprocal Rank Fusion weighting
            logger.debug(f"Running {query.mode} mode query pipeline")
            scores_fields = ["vector_score", "fulltext_score"]
            filter = self.filters_to_mql(query.filters, metadata_key=self._metadata_key)
            pipeline = []
            # Vector Search pipeline
            if query.query_embedding:
                vector_pipeline = [
                    vector_search_stage(
                        query_vector=query.query_embedding,
                        search_field=self._embedding_key,
                        index_name=self._vector_index_name,
                        limit=dense_top_k,
                        filter=filter,
                        oversampling_factor=self._oversampling_factor,
                    )
                ]
                vector_pipeline.extend(reciprocal_rank_stage("vector_score"))
                combine_pipelines(pipeline, vector_pipeline, self._collection.name)

            # Full-Text Search pipeline
            if query.query_str:
                text_pipeline = fulltext_search_stage(
                    query=query.query_str,
                    search_field=self._text_key,
                    index_name=self._fulltext_index_name,
                    operator="text",
                    filter=filter,
                    limit=sparse_top_k,
                )
                text_pipeline.extend(reciprocal_rank_stage("fulltext_score"))
                combine_pipelines(pipeline, text_pipeline, self._collection.name)

            # Compute weighted sum and sort pipeline
            alpha = (
                query.alpha or 0.5
            )  # If no alpha is given, equal weighting is applied
            pipeline += final_hybrid_stage(
                scores_fields=scores_fields, limit=hybrid_top_k, alpha=alpha
            )

            # Remove embeddings unless requested.
            if (
                query.output_fields is None
                or self._embedding_key not in query.output_fields
            ):
                pipeline.append({"$project": {self._embedding_key: 0}})

        else:
            raise NotImplementedError(
                f"{VectorStoreQueryMode.DEFAULT} (vector), "
                f"{VectorStoreQueryMode.HYBRID} and {VectorStoreQueryMode.TEXT_SEARCH} "
                f"are available. {query.mode} is not."
            )

        # Execution
        logger.debug("Running query pipeline: %s", pipeline)
        cursor = self._collection.aggregate(pipeline)  # type: ignore

        # Post-processing
        top_k_nodes = []
        top_k_ids = []
        top_k_scores = []
        for res in cursor:
            text = res.pop(self._text_key)
            score = res.pop("score")
            id = res.pop(self._id_key)
            metadata_dict = res.pop(self._metadata_key)

            try:
                node = metadata_dict_to_node(metadata_dict)
                node.set_content(text)
            except Exception:
                # NOTE: deprecated legacy logic for backward compatibility
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    metadata_dict
                )

                node = TextNode(
                    text=text,
                    id_=id,
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )

            top_k_ids.append(id)
            top_k_nodes.append(node)
            top_k_scores.append(score)
        result = VectorStoreQueryResult(
            nodes=top_k_nodes, similarities=top_k_scores, ids=top_k_ids
        )
        logger.debug("Result of query: %s", result)
        return result
