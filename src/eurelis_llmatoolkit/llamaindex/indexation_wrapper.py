from factories.reader_factory import ReaderFactory


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

        print("### Docs : ")
        print(documents[0])
