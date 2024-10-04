from abc import ABC, abstractmethod


class ReaderAdapter(ABC):
    def __init__(self, config):
        self.config = config
        self.reader = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Méthode abstraite que chaque sous-classe doit implémenter pour appeler la méthode de chargement des données du reader"""

    @staticmethod
    @abstractmethod
    def get_load_data_params(dataset_config):
        """Méthode abstraite que chaque sous-classe doit implémenter pour extraire les urls à charger"""
