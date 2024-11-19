from eurelis_llmatoolkit.llamaindex.factories.retriever_factory import RetrieverFactory


def test_load_vector_index_retriever(embedding, index):
    """
    Test to ensure the factory correctly instantiates the built-in VectorIndexRetriever
    """

    config = {
        "provider": "VectorIndexRetriever",
        "similarity_top_k": 10,
        "index": index,
        "embed_model": embedding,
        "filters": None,
    }
    retriever = RetrieverFactory.create_retriever(config)
    assert retriever is not None, "Retriever should not be None"
    assert (
        retriever.__class__.__name__ == "VectorIndexRetriever"
    ), "Unexpected retriever class name"


def test_load_custom_retriever(embedding, index):
    """
    Test to verify that the factory can load a custom retriever class using a path
    """

    config = {
        "provider": "llama_index.core.retrievers.VectorIndexRetriever",
        "similarity_top_k": 10,
        "index": index,
        "embed_model": embedding,
        "filters": None,
    }
    retriever = RetrieverFactory.create_retriever(config)
    assert retriever is not None, "Retriever should not be None"
    assert (
        retriever.__class__.__name__ == "VectorIndexRetriever"
    ), "Unexpected retriever class name"


def test_invalid_provider_short_name():
    """
    Test to ensure the factory raises a ValueError when the provider name is not recognized
    """
    config = {"provider": "NonExistentRetriever"}
    try:
        RetrieverFactory.create_retriever(config)
        assert False, "Expected ValueError for invalid provider"
    except ValueError as e:
        assert (
            "Provider attribute must reference a standard Retriever short name or a fully qualified class path"
            in str(e)
        ), "Unexpected error message"


def test_invalid_provider_path():
    """
    Test to verify the factory raises an ImportError for an invalid provider path
    """
    config = {"provider": "invalid.path.to.Retriever"}
    try:
        RetrieverFactory.create_retriever(config)
        assert False, "Expected ImportError for invalid provider path"
    except ImportError as e:
        assert "Failed to import retriever class" in str(e), "Unexpected error message"


def test_missing_provider():
    """
    Test to confirm that the factory raises a ValueError when the provider field is missing
    """
    config = {}
    try:
        RetrieverFactory.create_retriever(config)
        assert False, "Expected ValueError for missing provider"
    except ValueError as e:
        assert "The 'provider' field is required in the configuration." in str(
            e
        ), "Unexpected error message"
