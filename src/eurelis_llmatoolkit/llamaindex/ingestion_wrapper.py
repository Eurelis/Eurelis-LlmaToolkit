from typing import List, Optional

from llama_index.core import Document
from llama_index.core.ingestion import IngestionPipeline

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper
from eurelis_llmatoolkit.llamaindex.factories.cache_factory import CacheFactory
from eurelis_llmatoolkit.llamaindex.factories.reader_factory import ReaderFactory
from eurelis_llmatoolkit.llamaindex.factories.transformation_factory import (
    TransformationFactory,
)

from eurelis_llmatoolkit.llamaindex.logger import Logger


class IngestionWrapper(AbstractWrapper):
    def __init__(self, config: dict):
        self._config = config
        self.logger = Logger().get_logger()
        self.logger.debug("IngestionWrapper initialized")

    def run(
        self,
        dataset_id: Optional[str] = None,
        use_cache: bool = False,
        delete: bool = False,
    ):
        self.logger.info(
            "Running ingestion with filtering dataset_id: %s, use_cache: %s",
            dataset_id,
            use_cache,
        )
        self._process_datasets(dataset_id, use_cache, delete)
        print("Ingestion completed!")

    def generate_cache(self, dataset_id: Optional[str] = None):
        self.logger.info("Generating cache for dataset_id: %s", dataset_id)
        # Récupérer la configuration des datasets
        datasets = list(
            self._filter_datasets(dataset_id)
        )  # Convertir l'iterable en liste

        if not datasets:
            self.logger.warning(f"No dataset found with ID: {dataset_id}")
            return

        # Parcourir tous les datasets et générer le cache
        for dataset_config in datasets:
            # Vérifier que l'ID du dataset n'est pas None
            dataset_id = dataset_config.get("id")
            if dataset_id is None:
                self.logger.warning(
                    f"Dataset configuration missing 'id': {dataset_config}"
                )
                continue

            documents = self._get_documents(dataset_config, use_cache=False)

            # Générer le cache
            self._generate_cache(dataset_id, documents)
            self.logger.info(f"Cache generated for dataset ID: {dataset_id}!")

    def _load_documents_from_reader(self, dataset_config: dict) -> List[Document]:
        """Load documents using the appropriate reader based on the dataset configuration."""
        self.logger.debug(
            "Loading documents from reader for dataset_config: %s", dataset_config
        )
        reader_adapter = ReaderFactory.create_reader(
            f"{self._config['project']}/{dataset_config['id']}",
            dataset_config["reader"],
        )
        documents = reader_adapter.load_data()

        # Add project metadata
        return self._add_project_metadata(documents, self._config["project"])

    def _load_documents_from_cache(self, dataset_config: dict) -> List[Document]:
        """Load documents from cache if the cache is available."""
        self.logger.debug(
            "Loading documents from cache for dataset_config: %s", dataset_config
        )
        cache_config = self._config.get("scraping_cache", [])
        cache = CacheFactory.create_cache(cache_config)
        documents = cache.load_data(dataset_config["id"])

        # Add project metadata
        return self._add_project_metadata(documents, self._config["project"])

    def _add_project_metadata(
        self, documents: List[Document], project: str
    ) -> List[Document]:
        """Add project metadata to each document in the list.

        Args:
            documents (List[Document]): List of documents to be processed.
            project (str): Project name to add as metadata.

        Returns:
            List[Document]: List of documents with added project metadata.
        """
        self.logger.debug(
            "Adding project metadata to documents for project: %s", project
        )
        for doc in documents:
            if not hasattr(doc, "metadata"):
                doc.metadata = {}
            doc.metadata["project"] = project
        return documents

    def _process_datasets(
        self,
        dataset_id: Optional[str] = None,
        use_cache: bool = False,
        delete: bool = False,
    ):
        """Process all datasets or a specific dataset based on the dataset ID."""
        self.logger.info(
            "Processing datasets with filtering dataset_id: %s, use_cache: %s",
            dataset_id,
            use_cache,
        )
        for dataset_config in self._filter_datasets(dataset_id):
            self._ingest_dataset(dataset_config, use_cache, delete)

    def _generate_cache(self, dataset_name: str, documents: list):
        self.logger.debug("Generating cache for dataset_name: %s", dataset_name)
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
        self.logger.debug(
            "Getting documents for dataset_config: %s, use_cache: %s",
            dataset_config,
            use_cache,
        )
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
        self.logger.debug(
            "Getting transformations for dataset_config: %s", dataset_config
        )
        # Transformations
        transformations = [
            TransformationFactory.create_transformation(t_config)
            for t_config in dataset_config["transformations"]
        ]

        # First transformations
        # Acronym
        acronyms = dataset_config.get("acronyms", None)
        if acronyms:
            transformations.insert(
                0,
                TransformationFactory.create_transformation(dataset_config["acronyms"]),
            )
        # Metadata
        metadata = dataset_config.get("metadata", None)
        if metadata:
            transformations.insert(
                0,
                TransformationFactory.create_transformation(dataset_config["metadata"]),
            )

        # Embedding (last transformation)
        transformations.append(self._get_embedding_model())

        return transformations

    def _get_doc_ids_from_document_store(self, id_dataset=None) -> List[str]:
        """Retrieve all doc_ids from the database."""
        document_store = self._get_document_store()
        documents: dict = document_store.docs

        if not documents:
            print("No documents found in the database.")
            return []

        # Filtrage basé sur id_dataset
        if id_dataset is not None:
            namespace_filter = f"{self._config['project']}/{id_dataset}"
            doc_ids = {
                doc.id_
                for doc in documents.values()
                if doc.metadata.get("namespace") == namespace_filter
            }
        else:
            # Si id_dataset est None, on ne filtre pas par namespace
            doc_ids = {doc.id_ for doc in documents.values()}

        return list(doc_ids)

    def _get_doc_ids_from_documents(self, documents: List[Document]) -> List[str]:
        """Create a list of doc_ids from the documents."""
        doc_ids = [str(doc.id_) for doc in documents]
        return doc_ids

    def _remove_unmatched_documents(
        self, doc_ids_doc_store: List[str], doc_ids_scraping: List[str]
    ):
        """Remove documents from the database that do not match any of the provided doc_ids."""
        doc_ids_to_delete = set(doc_ids_doc_store) - set(doc_ids_scraping)
        document_store = self._get_document_store()
        vector_store = self._get_vector_store()

        if doc_ids_to_delete:
            for url in doc_ids_to_delete:
                document_store.delete_document(doc_id=url)
                vector_store.delete(url)
                print(f"Deleting document with URL: {url}")
        else:
            print("Aucune URL à supprimer.")

    def _ingest_dataset(
        self, dataset_config: dict, use_cache: bool = False, delete: bool = False
    ):
        """
        Ingest the dataset using the provided configuration.

        Args:
            dataset_config (dict): Configuration for the dataset.
        """
        self.logger.info("Ingesting dataset with use_cache: %s", use_cache)
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

        # Récupérer les doc_ids en base => doc_ids_doc_store
        doc_ids_doc_store = self._get_doc_ids_from_document_store(
            id_dataset=dataset_config["id"]
        )

        # Faire une liste des doc_ids des documents => doc_ids_scraping
        doc_ids_scraping = self._get_doc_ids_from_documents(documents)

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

        # Supprimer les documents du document_store qui ne sont pas dans doc_ids_scraping
        if delete:
            print("Deleting old documents...")
            self._remove_unmatched_documents(doc_ids_doc_store, doc_ids_scraping)
        else:
            print("The delete option is disabled: old documents will not be deleted.")
        self.logger.info(f"Dataset {dataset_config['id']} ingested successfully.")
