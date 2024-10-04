from .config_loader import ConfigLoader
from .ingestion_wrapper import IngestionWrapper

# Pour les tests
if __name__ == "__main__":

    config = ConfigLoader.load_config("./eurelis_llmatoolkit/llamaindex/config.json")
    ingestion_wrapper = IngestionWrapper(config)
    ingestion_wrapper.run()
