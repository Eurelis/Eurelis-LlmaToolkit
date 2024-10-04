from eurelis_llmatoolkit.llamaindex.ingestion_wrapper import IngestionWrapper


def test_simple_init():
    indexation_wrapper = IngestionWrapper(None)
    assert indexation_wrapper is not None
