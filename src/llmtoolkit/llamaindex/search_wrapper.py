import logging
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper

logger = logging.getLogger(__name__)


class SearchWrapper(AbstractWrapper):
    """Search wrapper class to handle search operations."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._retriever = None
        logger.info("SearchWrapper initialized.")

    def get_filters_formatted(self) -> list:
        """Return the filters formatted."""
        filters = []

        if self._retriever is not None and self._retriever._filters is not None:
            for meta_filter in self._retriever._filters.filters:
                meta_filter: MetadataFilter

                filters.append(
                    {
                        "metadata": meta_filter.key,
                        "value": meta_filter.value,
                        "operator": meta_filter.operator,
                    }
                )

        return filters

    def search_documents(self, query: str, extract_filters: bool = False) -> list[dict]:
        """
        Search the index for the given query and return the results.

        Args:
            query: The search query

        Returns:
            List of documents
        """
        logger.info("Searching documents with query: %s", query)
        # recherche dans l'index
        results: list[NodeWithScore] = self.search_nodes(
            query, extract_filters=extract_filters
        )

        # tri des noeuds par document
        documents = {}

        for node in results:
            doc_id = node.node.metadata["source"]
            if doc_id not in documents:
                documents[doc_id] = {
                    "score": node.score,
                    "node": node,
                }
            else:
                if node.score > documents[doc_id]["score"]:
                    documents[doc_id] = {
                        "score": node.score,
                        "node": node,
                    }

        sorted_documents = sorted(
            [doc["node"] for doc in documents.values()],
            key=lambda x: x.score,
            reverse=True,
        )
        logger.debug("Documents sorted by score.")
        return sorted_documents

    def search_nodes(
        self, query: str, extract_filters: bool = False
    ) -> list[NodeWithScore]:
        """
        Search the index for the given query and return the results.

        Args:
            query: The search query

        Returns:
            List of NodeWithScore
        """
        logger.info("Searching nodes.")
        retriever = self._get_retriever(self._config["search_engine"])
        # recherche dans l'index
        if getattr(retriever, "supports_extract_filters", False):
            results = retriever.retrieve(query, extract_filters=extract_filters)
        else:
            results = retriever.retrieve(query)
        logger.debug("Nodes retrieved.")
        return results
