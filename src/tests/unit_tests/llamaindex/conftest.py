import os
import pytest
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
    FilterCondition,
)

from eurelis_llmatoolkit.llamaindex.chatbot_wrapper import ChatbotWrapper
from eurelis_llmatoolkit.llamaindex.config_loader import ConfigLoader
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
        memory_persistence_config, memory, "test_conversation"
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


@pytest.fixture
def chatbot_config_project1():
    config_dict = ConfigLoader.load_config(
        "../etc/config_tests/tests/unit_tests/llamaindex/test_eurelis1.json"
    )
    return config_dict


@pytest.fixture
def chatbot_config_project2():
    config_dict = ConfigLoader.load_config(
        "../etc/config_tests/tests/unit_tests/llamaindex/test_eurelis2.json"
    )
    return config_dict


@pytest.fixture
def common_filter():
    return MetadataFilters(
        filters=[MetadataFilter(key="common_field", value="common_value")],
        condition=FilterCondition.AND,
    )


@pytest.fixture
def unique_filter_project1():
    return MetadataFilters(
        filters=[
            MetadataFilter(key="unique_field_project1", value="eurelis_unique_value1")
        ],
        condition=FilterCondition.AND,
    )


@pytest.fixture
def unique_filter_project2():
    return MetadataFilters(
        filters=[
            MetadataFilter(key="unique_field_project2", value="eurelis_unique_value2")
        ],
        condition=FilterCondition.AND,
    )


@pytest.fixture
def chatbot_project1(chatbot_config_project1):
    return ChatbotWrapper(chatbot_config_project1, "test_conversation_project1")


@pytest.fixture
def chatbot_project2(chatbot_config_project2):
    return ChatbotWrapper(chatbot_config_project2, "test_conversation_project2")


@pytest.fixture
def chatbot_project1_with_common_filter(chatbot_config_project1, common_filter):
    return ChatbotWrapper(
        chatbot_config_project1,
        "test_conversation_project1",
        permanent_filters=common_filter,
    )


@pytest.fixture
def chatbot_project2_with_common_filter(chatbot_config_project2, common_filter):
    return ChatbotWrapper(
        chatbot_config_project2,
        "test_conversation_project2",
        permanent_filters=common_filter,
    )


@pytest.fixture
def combined_permanent_filter():
    return MetadataFilters(
        filters=[
            MetadataFilter(key="common_field", value="common_value"),
            MetadataFilter(key="unique_field_project1", value="eurelis_unique_value1"),
        ],
        condition=FilterCondition.AND,
    )


@pytest.fixture
def chatbot_with_combined_permanent_filter(
    chatbot_config_project1, combined_permanent_filter
):
    return ChatbotWrapper(
        chatbot_config_project1,
        "test_conversation_project1",
        permanent_filters=combined_permanent_filter,
    )
