from eurelis_llmatoolkit.llamaindex.ingestion_wrapper import IngestionWrapper
from eurelis_llmatoolkit.llamaindex.config_loader import ConfigLoader


def test_simple_init():
    indexation_wrapper = IngestionWrapper(None)
    assert indexation_wrapper is not None


def test_init_with_basic_config():
    config = ConfigLoader.load_config("../etc/config_samples/basic_config.json")

    indexation_wrapper = IngestionWrapper(config)
    assert indexation_wrapper is not None


def test_run_simple_ingestion():
    config = ConfigLoader.load_config("../etc/config_samples/basic_config.json")

    indexation_wrapper = IngestionWrapper(config)
    indexation_wrapper.run()
