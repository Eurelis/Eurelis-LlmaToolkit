from eurelis_llmatoolkit.llamaindex.config_loader import ConfigLoader
from eurelis_llmatoolkit.llamaindex.ingestion_wrapper import IngestionWrapper


# Pour les tests
if __name__ == "__main__":

    config = ConfigLoader.load_config("./eurelis_llmatoolkit/llamaindex/config.json")
    ingestion_wrapper = IngestionWrapper(config)
    ingestion_wrapper.run()
