import logging
from eurelis_llmatoolkit.api.misc.base_config import config
from eurelis_llmatoolkit.api.misc.singleton import Singleton

class ConsoleManager(metaclass=Singleton):
    """La classe ConsoleManager permet de gérer les configurations de la console.

    Elle utilise le module logging pour gérer les niveaux de verbosité et de sortie.
    """

    def __init__(self):
        # Obtenir le logger configuré par uvicorn
        self.logger = logging.getLogger()

    def get_output(self):
        """Renvoie le logger configuré."""
        return self.logger


# Exemple d'utilisation
#     Configurer le logger pour avec le lancement d'uvicorn avec un fichier .ini
#     console_manager = ConsoleManager()
#     logger = console_manager.get_output()

#     logger.debug("Ceci est un message de débogage")
#     logger.info("Ceci est un message d'information")
#     logger.warning("Ceci est un message d'avertissement")
#     logger.error("Ceci est un message d'erreur")
#     logger.critical("Ceci est un message critique")

# DEBUG(10)>INFO(20)>WARNING(30)>ERROR(40)>CRITICAL(50)