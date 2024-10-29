from llama_index.core.schema import NodeWithScore

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper


class SearchWrapper(AbstractWrapper):
    """Search wrapper class to handle search operations."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._retriever = None

    def search_documents(self, query: str, extract_filters: bool = False) -> list[dict]:
        """
        Search the index for the given query and return the results.

        Args:
            query: The search query

        Returns:
            List of documents
        """
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

        return sorted(
            [doc["node"] for doc in documents.values()],
            key=lambda x: x.score,
            reverse=True,
        )

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
        retriever = self._get_retriever(self._config["search_engine"])
        # recherche dans l'index
        if getattr(retriever, "supports_extract_filters", False):
            results = retriever.retrieve(query, extract_filters=extract_filters)
        else:
            results = retriever.retrieve(query)
        return results
