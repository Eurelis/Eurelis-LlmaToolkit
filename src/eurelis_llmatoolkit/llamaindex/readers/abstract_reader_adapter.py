from abc import ABC, abstractmethod


class AbstractReaderAdapter(ABC):
    required_params = []

    def __init__(self, config: dict):
        self.config = config
        self.reader = None
        self._unsuccessful_docs: list[str] = (
            []
        )  # Liste des docs non récupérés(ex:pages en timeout)

    @abstractmethod
    def load_data(self, *args, **kwargs):
        """Méthode abstraite que chaque sous-classe doit implémenter pour appeler la méthode de chargement des données du reader"""

    def _get_load_data_params(self):
        """Récupère les paramètres nécessaires pour charger les données à partir de la configuration."""
        return {param: self.config[param] for param in self.__class__.required_params}

    def get_unsuccessful_docs(self) -> list[str]:
        """Retourne une liste vide par défaut pour les URLs/Docs échouées."""
        return self._unsuccessful_docs
