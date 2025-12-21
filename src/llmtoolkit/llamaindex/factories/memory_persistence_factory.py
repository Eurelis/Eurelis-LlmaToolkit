import importlib
from typing import Optional
from llama_index.core.memory.types import BaseChatStoreMemory

from eurelis_llmatoolkit.llamaindex.chat_memory_persistence.json_persistence_handler import (
    JSONPersistenceHandler,
)


class MemoryPersistenceFactory:
    @staticmethod
    def create_memory_persistence(
        memory_persistence_config: dict,
        memory: BaseChatStoreMemory,
        conversation_id: Optional[str] = None,
        _persistance_data: Optional[dict] = None,
    ):
        provider = memory_persistence_config["provider"]

        if provider == "JSONPersistenceHandler":
            return JSONPersistenceHandler(
                memory_persistence_config, memory, conversation_id
            )

        #
        # If the provider is a custom reader
        #
        if provider.count(".") == 0:
            raise ValueError(
                "Provider attribute must reference a standard MemoryPersistence short name or a fully qualified class path"
            )

        module_name, class_name = provider.rsplit(".", 1)

        module = importlib.import_module(module_name)

        reader_class = getattr(module, class_name)

        return reader_class(
            memory_persistence_config, memory, conversation_id, _persistance_data
        )
