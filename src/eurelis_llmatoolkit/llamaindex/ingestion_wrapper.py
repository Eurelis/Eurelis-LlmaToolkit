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


class IngestionWrapper:
    def __init__(self, config: dict):
        self._config: dict = config
        self._vector_store: "BasePydanticVectorStore" = None
        self._document_store = None

    def run(self, dataset_id: Optional[str] = None):
        indexes = self._process_datasets(dataset_id)
        return indexes

    def _load_documents_from_reader(self, dataset_config: dict) -> List[Document]:
        """Load documents using the appropriate reader based on the dataset configuration."""
        reader_adapter = ReaderFactory.create_reader(
            dataset_config["reader"],
            f"{self._config['project']}/{dataset_config['id']}",
        )
        return reader_adapter.load_data()

    def _load_documents_from_cache(self, dataset_config: dict) -> List[Document]:
        """Load documents from cache if the cache is available."""
        cache_config = self._config.get("scraping_cache", [])
        cache = CacheFactory.create_cache(cache_config)
        return cache.load_data(dataset_config["id"])

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

    def _process_datasets(self, dataset_id: Optional[str] = None):
        """Process all datasets or a specific dataset based on the dataset ID."""
        for dataset_config in self._filter_datasets(dataset_id):
            self._ingest_dataset(dataset_config)

    def _generate_cache(self, dataset_name: str, documents: list):
        cache_config = self._config.get("scraping_cache", [])
        cache = CacheFactory.create_cache(cache_config)
        cache.to_cache(dataset_name, documents)

    def _get_documents(self, dataset_config: dict, use_cache: bool) -> List[Document]:
        """
        Get documents from either cache or reader based on the configuration.

        Args:
            dataset_config (dict): The configuration for the dataset.
            use_cache (bool): Whether to use cached data or read from source.

        Returns:
            List[Document]: List of retrieved documents.
        """
        return (
            self._load_documents_from_cache(dataset_config)
            if use_cache
            else self._load_documents_from_reader(dataset_config)
        )

    def _get_transformations(self, dataset_config: dict) -> List:
        """
        Retrieve transformations and embeddings for the ingestion pipeline.

        Args:
            dataset_config (dict): The dataset configuration that specifies the transformations.
        """
        # Transformations
        transformations = [
            TransformationFactory.create_transformation(t_config)
            for t_config in dataset_config["transformations"]
        ]

        # Acronym (first transformation)
        transformations.insert(
            0, TransformationFactory.create_transformation(dataset_config["acronyms"])
        )

        # Embedding (last transformation)
        embedding_config = self._config["embeddings"]
        embedding_model = EmbeddingFactory.create_embedding(embedding_config)
        transformations.append(embedding_model)

        return transformations

    def _ingest_dataset(self, dataset_config: dict, use_cache: bool = False):
        """
        Ingest the dataset using the provided configuration.

        Args:
            dataset_config (dict): Configuration for the dataset.
        """
        #
        # READER / CACHE
        #
        # Récupérer les documents à partir du cache ou via le reader
        documents = self._get_documents(dataset_config, use_cache)

        #
        # ACRONYMS & NODE PARSER & EMBEDDINGS
        #
        # Transformations
        transformations = self._get_transformations(dataset_config)

        #
        # VECTOR STORE
        #
        # Récupérer le vector store
        vector_store = self._get_vector_store()

        #
        # DOCUMENT STORE
        #
        # Optionnellement, définir un document store pour gérer les documents
        document_store = self._get_document_store()

        #
        # INGESTION PIPELINE
        #
        # Créer le pipeline d'ingestion
        pipeline = IngestionPipeline(
            transformations=transformations,
            vector_store=vector_store,
            docstore=document_store,
        )
        # TODO: définir le show_progress via une variable d'environnement
        pipeline.run(documents=documents, show_progress=True)
