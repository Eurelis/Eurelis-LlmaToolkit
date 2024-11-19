import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext

from eurelis_llmatoolkit.llamaindex.factories.embedding_factory import EmbeddingFactory
from eurelis_llmatoolkit.llamaindex.factories.vectorstore_factory import (
    VectorStoreFactory,
)


@pytest.fixture
def embedding():
    pass
    return EmbeddingFactory.create_embedding(
        {
            "provider": "OpenAI",
            "model": "text-embedding-3-small",
            "openai_api_key": "FAKE_KEY",
        }
    )


@pytest.fixture
def vector_store():
    pass
    return VectorStoreFactory.create_vector_store(
        {"provider": "Chroma", "collection_name": "llamaindex", "mode": "ephemeral"}
    )


@pytest.fixture
def storage_context(vector_store):
    pass
    return StorageContext.from_defaults(vector_store=vector_store)


@pytest.fixture
def index(vector_store, embedding, storage_context):
    pass
    return VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embedding, storage_context=storage_context
    )
