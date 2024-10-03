from eurelis_llmatoolkit.llamaindex.indexation_wrapper import IndexationWrapper


def test_simple_init():
    indexation_wrapper = IndexationWrapper(None)
    assert indexation_wrapper is not None
