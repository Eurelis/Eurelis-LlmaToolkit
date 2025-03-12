from typing import Any, Dict
from llama_index.core.vector_stores.types import MetadataFilters, FilterCondition
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
            return (
                f"{metadata_key}.{key}"
                if not key.startswith(f"{metadata_key}.")
                else key
            )

        def process_filters(filters: MetadataFilters) -> Dict[str, Any]:
            if len(filters.filters) == 1:
                mf = filters.filters[0]
                return {
                    prepare_key(mf.key): {
                        map_lc_mql_filter_operators(mf.operator): mf.value
                    }
                }
            elif filters.condition == FilterCondition.AND:
                return {
                    "$and": [
                        (
                            process_filters(mf)
                            if isinstance(mf, MetadataFilters)
                            else {
                                prepare_key(mf.key): {
                                    map_lc_mql_filter_operators(mf.operator): mf.value
                                }
                            }
                        )
                        for mf in filters.filters
                    ]
                }
            elif filters.condition == FilterCondition.OR:
                return {
                    "$or": [
                        (
                            process_filters(mf)
                            if isinstance(mf, MetadataFilters)
                            else {
                                prepare_key(mf.key): {
                                    map_lc_mql_filter_operators(mf.operator): mf.value
                                }
                            }
                        )
                        for mf in filters.filters
                    ]
                }
            else:
                logger.debug("filters.condition not recognized. Returning empty dict")
                return {}

        return process_filters(filters)
