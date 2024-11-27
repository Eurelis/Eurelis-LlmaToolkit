import os
import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)

from eurelis_llmatoolkit.llamaindex.chatbot_wrapper import ChatbotWrapper
from eurelis_llmatoolkit.llamaindex.factories.embedding_factory import EmbeddingFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_factory import MemoryFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_persistence_factory import (
    MemoryPersistenceFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.vectorstore_factory import (
    VectorStoreFactory,
)

from dotenv import load_dotenv

load_dotenv()


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


@pytest.fixture
def memory_config():
    return {"provider": "ChatMemoryBuffer", "token_limit": 1500}


@pytest.fixture
def memory_persistence_config():
    return {
        "provider": "JSONPersistenceHandler",
        "persist_conversation_path": "../etc/no-commit/conversation_history_for_unit_tests.json",
    }


@pytest.fixture
def chatbot_config(memory_config, memory_persistence_config):
    return {
        "chat_engine": {
            "provider": "ContextChatEngine",
            "retriever": {"provider": "VectorIndexRetriever", "similarity_top_k": 10},
            "memory": memory_config,
            "memory_persistence": memory_persistence_config,
            "system_prompt": [
                "You are a chatbot, able to have normal interactions, as well as talk"
            ],
        },
        "embedding_model": {
            "provider": "OpenAI",
            "model": "text-embedding-3-small",
            "openai_api_key": os.getenv("OPENAI_API_KEY}"),
        },
        "vectorstore": {
            "provider": "MongoDB",
            "url": os.getenv("MONGO_CONNECTION_STRING"),
            "db_name": os.getenv("MONGO_DB_NAME"),
            "collection_name": os.getenv("MONGO_COLLECTION_NAME"),
            "vector_index_name": "vector_index",
        },
    }


@pytest.fixture
def chatbot(chatbot_config):
    return ChatbotWrapper(chatbot_config, "test_conversation")


@pytest.fixture
def memory(memory_config):
    return MemoryFactory.create_memory(memory_config, "test_conversation")


@pytest.fixture
def memory_persistence(memory_persistence_config, memory):
    return MemoryPersistenceFactory.create_memory_persistence(
        memory_persistence_config, memory
    )


@pytest.fixture
def filters():
    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="c_product",
                operator=FilterOperator.EQ,
                value="product-flexacryl-instant-waterproof-compound",
            )
        ]
    )
