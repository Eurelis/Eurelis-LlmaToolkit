from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, List, Optional

from llama_index.core import Document, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline

from eurelis_llmatoolkit.llamaindex.factories.cache_factory import CacheFactory
from eurelis_llmatoolkit.llamaindex.factories.documentstore_factory import (
    DocumentStoreFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.embedding_factory import EmbeddingFactory
from eurelis_llmatoolkit.llamaindex.factories.reader_factory import ReaderFactory
from eurelis_llmatoolkit.llamaindex.factories.transformation_factory import (
    TransformationFactory,
)
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

    @abstractmethod
    def run(self, dataset_id: Optional[str] = None, use_cache: bool = False):
        """Run the wrapper process for the given dataset_id."""

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

    def get_vector_store_index(self):
        """Create your index"""
        return VectorStoreIndex.from_vector_store(self._get_vector_store())

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