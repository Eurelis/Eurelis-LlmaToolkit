from llama_index.core.schema import NodeWithScore
from llama_index.core.storage import StorageContext

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper
from llama_index.core.vector_stores.types import MetadataFilters


class SearchWrapper(AbstractWrapper):
    """Search wrapper class to handle search operations."""

    def __init__(self, config: dict):
        super().__init__(config)
        self._retriever = None

    def search_documents(self, query: str) -> list[dict]:
        """
        Search the index for the given query and return the results.

        Args:
            query: The search query

        Returns:
            List of documents
        """
        # recherche dans l'index
        results: list[NodeWithScore] = self.search_nodes(query)

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

    def search_nodes(self, query: str) -> list[NodeWithScore]:
        """
        Search the index for the given query and return the results.

        Args:
            query: The search query

        Returns:
            List of NodeWithScore
        """
        self._initialize_retriever()
        # recherche dans l'index
        results = self._retriever.retrieve(query)
        return results

    def _initialize_retriever(self, filters: MetadataFilters = None):
        # Récupération du vector store
        vector_store = self._get_vector_store()

        # Récupération du document store
        document_store = self._get_document_store()

        # Création du contexte de stockage
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store, docstore=document_store
        )

        # Création de l'index
        vector_store_index = self._get_vector_store_index(storage_context)

        # Récupération du modèle d'embedding
        embedding_model = self._get_embeddings()

        # Création du retriever
        self._retriever = self._get_retriever(
            self._config["search_engine"],
            index=vector_store_index,
            filters=filters,
            embedding_model=embedding_model,
        )
