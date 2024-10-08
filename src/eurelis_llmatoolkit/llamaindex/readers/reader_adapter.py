from abc import ABC, abstractmethod


class ReaderAdapter(ABC):
    required_params = []

    def __init__(self, config):
        self.config = config
        self.reader = None

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Méthode abstraite que chaque sous-classe doit implémenter pour appeler la méthode de chargement des données du reader"""

    def _get_load_data_params(self):
        """Récupère les paramètres nécessaires pour charger les données à partir de la configuration."""
        return {param: self.config[param] for param in self.__class__.required_params}
