import json
from typing import Optional

from eurelis_llmatoolkit.api.misc.singleton import Singleton


class AgentManager(metaclass=Singleton):
    """La classe AgentManager permet de gérer les informations relatives aux agents.


    Elle s'appuie sur un dictionnaire de clés API autorisées dont la structure est la suivante:
    {
        "uuid": {
            "id": "uuid",
            "is_active": true | false,
            "name": "name",
            "default_response": "default_response",
            "ui_params": {
                "locale": "en",
                "labels": {
                    "title": "title",
                    "description": "description",
                    "button": "button"
                }
            }
        }
    }
    """

    def __init__(self):
        super().__init__()
        self.agents = None
        with open("config/agents.json") as json_file:
            self.agents = json.load(json_file)

        if self.agents is None:
            raise RuntimeError

    def is_authorized(self, key):
        """Vérifie si la clé API est autorisée à accéder à la ressource.

        Args:
            key (Str): Clé API

        Returns:
            Str: "authorized" si la clé est autorisée, "unauthorized" si la clé n'est pas autorisée, "forbidden" si la clé est désactivée
        """
        if key is None:
            return "unauthorized"

        key = self.agents.get(key)
        if key is None:
            return "unauthorized"
        if key["is_active"]:
            return "authorized"
        return "forbidden"
    
    def is_search_active(self, key):
        """Vérifie si la clé API est autorisée à accéder à search.

        Args:
            key (Str): Clé API

        Returns:
            Bool: True si la clé est True, False si désactivée
        """
        if key is None:
            return False

        key = self.agents.get(key)
        if key is None:
            return False
        if key["is_search_active"]:
            return True
        return False

    def is_similarity_active(self, key):
        """Vérifie si la clé API est autorisée à accéder à similarity.

        Args:
            key (Str): Clé API

        Returns:
            Bool: True si la clé est True, False si désactivée
        """
        if key is None:
            return False

        key = self.agents.get(key)
        if key is None:
            return False
        if key["is_similarity_active"]:
            return True
        return False

    def get_ui_params(self, key):
        """Retourne les paramètres d'interface de l'agent associé à la clé API.

        Args:
            key (Str): Clé API

        Returns:
            Dict: Paramètres d'interface
        """
        if key is None:
            return None

        key = self.agents.get(key)
        if key is None or "ui_params" not in key:
            return None
        return key["ui_params"]

    def get_max_history(self, key: str) -> Optional[int]:
        """Retourne le nombre maximum de messages à renvoyer.

        Args:
            key (Str): Clé API

        Returns:
            Int: Nombre maximum de messages à conserver en mémoire
        """
        if key is None:
            return None

        agent = self.agents.get(key)
        if agent is None or "max_history" not in agent:
            return None
        return agent["max_history"]

    def get_max_results(self, key: str) -> Optional[int]:
        """Retourne le nombre maximum de résultats à renvoyer.

        Args:
            key (Str): Clé API

        Returns:
            Int: Nombre maximum de résultats à renvoyer
        """
        if key is None:
            return None

        agent = self.agents.get(key)
        if agent is None or "max_results" not in agent:
            return None
        return agent["max_results"]

    def get_prefixes_img(self, key: str) -> Optional[dict]:
        """Retourne les préfixes des urls des images.

        Args:
            key (Str): Clé API

        Returns:
            Dict: Préfixes des URLs des images
        """
        if key is None:
            return None

        agent = self.agents.get(key)
        if agent is None or "prefixes_url_image" not in agent:
            return None
        return agent["prefixes_url_image"]


    def get_allowed_origines(self) -> list:
        """Retourne les origines autorisées pour les agents actifs.

        Returns:
            List: Liste des origines autorisées
        """

        allowed_origines = []
        for key in self.agents:
            if self.agents[key]["is_active"]:
                allowed_origines.extend(self.agents[key]["origins"])

        # Supression des doublons
        allowed_origines = list(set(allowed_origines))

        return allowed_origines

    def get_default_response(self, key):
        """Retourne la réponse par défaut de l'agent associé à la clé API.

        Args:
            key (Str): Clé API

        Returns:
            Str: Réponse par défaut
        """
        if key is None:
            return None

        key = self.agents.get(key)
        if key is None or "default_response" not in key:
            return None
        else:
            return key["default_response"]

    def get_agent_mode(self, key) -> str:
        """Retourne le mode de l'agent.

        Returns:
            str: Mode de l'agent
        """
        if key is None:
            return "llm"

        key = self.agents.get(key)
        if key is None or "agent_mode" not in key:
            return "llm"
        else:
            return key["agent_mode"]

    def get_agent_name(self, key) -> str:
        """Retourne le nom de l'agent.

        Returns:
            str: Nom de l'agent
        """
        if key is None:
            return "llm"

        key = self.agents.get(key)
        if key is None or "name" not in key:
            return "llm"
        else:
            return key["name"]

    def get_agent_version(self, key) -> str:
        """Retourne le nom de l'agent.

        Returns:
            str: Nom de l'agent
        """
        if key is None:
            return ""

        key = self.agents.get(key)
        if key is None or "version" not in key:
            return ""
        else:
            return key["version"]

    def get_llmatoolkit_config(self, key: str|None) -> Optional[str]:
        """Retourne le fichier de configuration associé à la clé API.

        Args:
            key (Str): Clé API

        Returns:
            Str: Fichier de configuration
        """
        if key is None:
            return None

        agent = self.agents.get(key)

        if agent is None or "llmatoolkit_config" not in agent:
            return None
        return agent["llmatoolkit_config"]
