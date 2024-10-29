from abc import ABC
from typing import TYPE_CHECKING, Iterable, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores.types import MetadataFilters

from eurelis_llmatoolkit.llamaindex.factories.documentstore_factory import (
    DocumentStoreFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.embedding_factory import EmbeddingFactory
from eurelis_llmatoolkit.llamaindex.factories.retriever_factory import RetrieverFactory
from eurelis_llmatoolkit.llamaindex.factories.vectorstore_factory import (
    VectorStoreFactory,
)

if TYPE_CHECKING:
    from llama_index.core.vector_stores.types import BasePydanticVectorStore
    from llama_index.core.retrievers import BaseRetriever


class AbstractWrapper(ABC):
    def __init__(self, config: dict):
        self._config: dict = config
        self._vector_store: "BasePydanticVectorStore" = None
        self._document_store: Optional["BasePydanticVectorStore"] = None
        self._storage_context: Optional[StorageContext] = None
        self._vector_store_index: Optional[VectorStoreIndex] = None
        self._retriever: "BaseRetriever" = None
        self._embedding_model: "BaseEmbedding" = None

    def _get_vector_store(self):
        if self._vector_store is not None:
            return self._vector_store

        vectorstore_config = self._config["vectorstore"]
        self._vector_store = VectorStoreFactory.create_vector_store(vectorstore_config)
        return self._vector_store

    def _get_document_store(self):
        if self._document_store is not None:
            return self._document_store

        documentstore_config = self._config.get("documentstore")
        if documentstore_config:
            self._document_store = DocumentStoreFactory.create_document_store(
                documentstore_config
            )

        return self._document_store

    def _get_storage_context(self):
        """
        Creates a StorageContext using the vector store and document store.

        Returns:
            StorageContext: The configured storage context.
        """
        if self._storage_context is not None:
            return self._storage_context

        # Get the vector store and document store
        vector_store = self._get_vector_store()
        document_store = self._get_document_store()

        # Create the StorageContext
        self._storage_context = StorageContext.from_defaults(
            vector_store=vector_store, docstore=document_store
        )

        return self._storage_context

    def _get_embeddings(self):
        if self._embedding_model is not None:
            return self._embedding_model

        embedding_config = self._config["embeddings"]
        self._embedding_model = EmbeddingFactory.create_embedding(embedding_config)

        return self._embedding_model

    def _get_retriever(
        self,
        config: dict,
        filters: Optional[MetadataFilters] = None,
    ):
        if self._retriever is not None:
            return self._retriever

        retriever_config = config.get("retriever")

        if retriever_config:
            index = self._get_vector_store_index()

            embedding_model = self._get_embeddings()

            self._retriever = RetrieverFactory.create_retriever(
                retriever_config,
                index=index,
                filters=filters,
                embedding_model=embedding_model,
            )

        return self._retriever

    def _get_vector_store_index(self):
        """Create your index"""
        if self._vector_store_index is not None:
            return self._vector_store_index

        vector_store = self._get_vector_store()

        storage_context = self._get_storage_context()

        self._vector_store_index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

        return self._vector_store_index

    def _filter_datasets(self, dataset_id: Optional[str] = None) -> Iterable[dict]:
        """
        Retrieve all datasets or filter by dataset ID if provided.

        Args:
            dataset_id: Optional, if provided, only returns datasets that match the given ID.

        Returns:
            A list of all datasets if no dataset_id is provided, otherwise a list of datasets
            filtered by the dataset_id.
        """
        datasets = self._config.get("dataset", [])

        if not dataset_id:
            return datasets

        filtered_datasets = [
            dataset for dataset in datasets if dataset.get("id") == dataset_id
        ]

        return filtered_datasets
