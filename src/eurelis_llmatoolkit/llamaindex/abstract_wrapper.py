from abc import ABC
from typing import TYPE_CHECKING, Iterable, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.storage import StorageContext

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


class AbstractWrapper(ABC):
    def __init__(self, config: dict):
        self._config: dict = config
        self._vector_store: "BasePydanticVectorStore" = None
        self._document_store = None

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

    def _get_embeddings(self):
        embedding_config = self._config["embeddings"]
        embedding_model = EmbeddingFactory.create_embedding(embedding_config)
        return embedding_model

    def _get_retriever(
        self,
        config: dict,
        index: Optional[VectorStoreIndex] = None,
        embedding_model: Optional[BaseEmbedding] = None,
    ):
        retriever_config = config["retriever"]
        retriever = RetrieverFactory.create_retriever(
            retriever_config, index=index, embedding_model=embedding_model
        )
        return retriever

    def _get_vector_store_index(self, storage_context: Optional[StorageContext] = None):
        """Create your index"""
        return VectorStoreIndex.from_vector_store(
            self._vector_store, storage_context=storage_context
        )

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
