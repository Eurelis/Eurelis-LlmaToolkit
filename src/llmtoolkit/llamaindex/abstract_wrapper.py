import logging
from abc import ABC
from typing import TYPE_CHECKING, Iterable, Optional

from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext

from eurelis_llmatoolkit.llamaindex.factories.documentstore_factory import (
    DocumentStoreFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.embedding_factory import EmbeddingFactory
from eurelis_llmatoolkit.llamaindex.factories.node_postprocessor_factory import (
    NodePostProcessorFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.retriever_factory import RetrieverFactory
from eurelis_llmatoolkit.llamaindex.factories.vectorstore_factory import (
    VectorStoreFactory,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llama_index.core.callbacks import CallbackManager
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        MetadataFilters,
    )


class AbstractWrapper(ABC):
    def __init__(
        self, config: dict, callback_manager: Optional["CallbackManager"] = None
    ):
        self._config: dict = config
        self._callback_manager = callback_manager
        self._vector_store: "BasePydanticVectorStore" = None
        self._document_store: Optional["BasePydanticVectorStore"] = None
        self._storage_context: Optional[StorageContext] = None
        self._vector_store_index: Optional[VectorStoreIndex] = None
        self._retriever: "BaseRetriever" = None
        self._node_postprocessors: Optional[list["BaseNodePostprocessor"]] = None
        self._embedding_model: "BaseEmbedding" = None
        logger.debug("AbstractWrapper initialized.")

    def _get_vector_store(self):
        if self._vector_store is not None:
            return self._vector_store

        vectorstore_config = self._config["vectorstore"]
        self._vector_store = VectorStoreFactory.create_vector_store(vectorstore_config)
        logger.info("Vector store created.")
        return self._vector_store

    def _get_document_store(self):
        if self._document_store is not None:
            return self._document_store

        documentstore_config = self._config.get("documentstore")
        if documentstore_config:
            self._document_store = DocumentStoreFactory.create_document_store(
                documentstore_config
            )
            logger.info("Document store created.")

        return self._document_store

    def _get_storage_context(self):
        """
        Creates a StorageContext using the vector store and document store.

        Returns:
            StorageContext: The configured storage context.
        """
        logger.debug("Creating storage context.")
        if self._storage_context is not None:
            return self._storage_context

        # Get the vector store and document store
        vector_store = self._get_vector_store()
        document_store = self._get_document_store()

        # Create the StorageContext
        self._storage_context = StorageContext.from_defaults(
            vector_store=vector_store, docstore=document_store
        )

        logger.debug("Storage context created.")
        return self._storage_context

    def _get_embedding_model(self):
        if self._embedding_model is not None:
            return self._embedding_model

        embedding_config = self._config["embedding_model"]
        self._embedding_model = EmbeddingFactory.create_embedding(
            embedding_config, callback_manager=self._callback_manager
        )

        logger.info("Embedding model created.")
        return self._embedding_model

    def _get_retriever(
        self,
        config: dict,
        filters: Optional["MetadataFilters"] = None,
    ) -> "BaseRetriever":
        retriever_config = config.get("retriever")

        if retriever_config:
            index = self._get_vector_store_index()

            embed_model = self._get_embedding_model()

            self._retriever = RetrieverFactory.create_retriever(
                {
                    "index": index,
                    "embed_model": embed_model,
                    "filters": filters,
                    **retriever_config,
                }
            )
            logger.debug("Retriever created.")

        return self._retriever

    def _get_node_postprocessors(self):
        if self._node_postprocessors is not None:
            return self._node_postprocessors

        post_processor_config = self._config.get("chat_engine", {}).get(
            "postprocessors", []
        )
        if not post_processor_config:
            return None

        self._node_postprocessors = NodePostProcessorFactory.create_node_postprocessors(
            configs=post_processor_config
        )
        logger.debug("Node postprocessors created.")
        return self._node_postprocessors

    def _get_vector_store_index(self):
        """Create your index"""
        logger.debug("Creating vector store index.")
        if self._vector_store_index is not None:
            return self._vector_store_index

        vector_store = self._get_vector_store()

        storage_context = self._get_storage_context()

        self._vector_store_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self._get_embedding_model(),
            storage_context=storage_context,
            callback_manager=self._callback_manager,
        )

        logger.debug("Vector store index created.")
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

        if not filtered_datasets:
            logger.warning(f"No dataset found with ID: {dataset_id}")

        return filtered_datasets
