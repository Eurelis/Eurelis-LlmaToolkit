from typing import TYPE_CHECKING, Iterable, Optional

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline

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
        # Create your index
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
        # On boucle sur chaque dataset dans la configuration

        for dataset_config in self._filter_datasets(dataset_id):
            self._ingest_dataset(dataset_config)

    def _ingest_dataset(self, dataset_config: dict):
        #
        # READER
        #
        # Charger le reader pour extraire les données
        reader_adapter = ReaderFactory.create_reader(dataset_config["reader"])

        # Obtenir les paramètres de chargement depuis le reader
        load_data_params = reader_adapter.get_load_data_params(dataset_config)
        documents = reader_adapter.load_data(**load_data_params)

        #
        # NODE PARSER
        #
        # Charger les transformations
        transformations = [
            TransformationFactory.create_transformation(t_config)
            for t_config in dataset_config["transformations"]
        ]

        #
        # VECTOR STORE
        #
        # Récupérer le vector store
        vector_store = self._get_vector_store()

        #
        # EMBEDDINGS
        #
        # Charger le modèle d'embedding
        embedding_config = self._config["embeddings"]
        embedding_model = EmbeddingFactory.create_embedding(embedding_config)

        # Définir le modèle d'embedding dans les transformations
        transformations.append(embedding_model)

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
