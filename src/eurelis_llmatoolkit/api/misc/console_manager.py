from eurelis_llmatoolkit.api.misc.base_config import config
from eurelis_llmatoolkit.api.misc.singleton import Singleton
from eurelis_llmatoolkit.utils.output import Verbosity, OutputFactory

class ConsoleManager(metaclass=Singleton):
    """La classe ConsoleManager permet de gérer les configurations de la console.

    Elle s'appuie sur OutputFactory pour gérer les niveaux de verbosité et de sortie.
    """

    def __init__(self):
        self.output_factory = OutputFactory()
        # Configurer le niveau de verbosité par défaut
        console_verbosity = config.get("CONSOLE_VERBOSITY", Verbosity.CONSOLE_DEBUG)
        self.output_factory.set_verbose(console_verbosity)
        self.output = self.output_factory.build(None)

    @staticmethod
    def get_instance():
        """Renvoie l'instance unique de ConsoleManager."""
        return ConsoleManager()

    def get_output(self):
        """Renvoie l'objet output configuré."""
        return self.output