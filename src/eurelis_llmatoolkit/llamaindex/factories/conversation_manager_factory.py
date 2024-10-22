from llama_index.core.memory.types import BaseChatStoreMemory

from eurelis_llmatoolkit.llamaindex.chat_manager.json_conversation_manager import (
    JSONConversationManager,
)


class ConversationManagerFactory:
    @staticmethod
    def create_conversation_manager(
        conversation_manager_config: dict, memory: BaseChatStoreMemory
    ):
        provider = conversation_manager_config["provider"]

        if provider == "JSONConversationManager":
            return JSONConversationManager(conversation_manager_config, memory)

        raise ValueError(f"ConversationManager provider {provider} is not supported.")
