from factories.reader_factory import ReaderFactory
from factories.transformation_factory import TransformationFactory
from factories.embedding_factory import EmbeddingFactory
from factories.vectorstore_factory import VectorStoreFactory

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.ingestion import IngestionPipeline


class IndexationWrapper:
    def __init__(self, config):
        self.config = config

    def run_test(self):
        print(self.config)

    def run_indexation(self):
        indexes = self.process_datasets()
        return indexes

    def process_datasets(self):
        indexes = []  # Pour stocker tous les index créés pour chaque dataset

        # On boucle sur chaque dataset dans la configuration
        for dataset_config in self.config["dataset"]:
            index = self.index_dataset(dataset_config)
            indexes.append(index)

        return indexes

    def index_dataset(self, dataset_config):
        # Charger le reader pour extraire les données
        reader = ReaderFactory.create_reader(dataset_config["reader"])

        # Obtenir les paramètres de chargement depuis le reader
        load_data_params = reader.get_load_data_params(dataset_config)
        documents = reader.load_data(**load_data_params)

        # Charger les transformations
        transformations = [
            TransformationFactory.create_transformation(t_config)
            for t_config in dataset_config["transformations"]
        ]

        # Créer le pipeline d'ingestion
        pipeline = IngestionPipeline(transformations=transformations)
        nodes = pipeline.run(documents=documents)

        # Embedding
        embed_config = self.config["embeddings"]
        embed_model = EmbeddingFactory.create_embedding(embed_config)

        # TODO : Vérifier si bonne facon de faire
        for node in nodes:
            embedding = embed_model.get_text_embedding(node.get_text())
            node.embedding = embedding

        # Vector Store
        vectorstore_config = self.config["vectorstore"]
        vector_store = VectorStoreFactory.create_vector_store(vectorstore_config)

        # # Stockage des embeddings et indexation
        # storage_context = StorageContext.from_defaults(vector_store=vector_store)
        # index = VectorStoreIndex.from_documents(
        #     nodes, storage_context=storage_context, show_progress=True
        # )

        # return index
