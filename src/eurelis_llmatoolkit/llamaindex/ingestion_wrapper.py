from typing import TYPE_CHECKING

from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline

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

    def run(self):
        indexes = self._process_datasets()
        return indexes

    def _get_vector_store(self):
        if self._vector_store is not None:
            return self._vector_store

        vectorstore_config = self._config["vectorstore"]
        self._vector_store = VectorStoreFactory.create_vector_store(vectorstore_config)
        return self._vector_store

    def get_vector_store_index(self):
        # Create your index
        return VectorStoreIndex.from_vector_store(self._get_vector_store())

    def _process_datasets(self):
        # On boucle sur chaque dataset dans la configuration
        for dataset_config in self._config["dataset"]:
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
        # INGESTION PIPELINE
        #
        # Créer le pipeline d'ingestion
        pipeline = IngestionPipeline(
            transformations=transformations, vector_store=vector_store
        )
        # TODO: définir le show_progress via une variable d'environnement
        pipeline.run(documents=documents, show_progress=True)
