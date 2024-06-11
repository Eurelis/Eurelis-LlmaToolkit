
from eurelis_llmatoolkit.utils.base_factory import ProviderFactory
from langchain_core.memory import BaseMemory


class GenericMemoryFactory(ProviderFactory[BaseMemory]):
    """
    Generic chains factory, provide with a conversational with memory question oriented chain
    """

    ALLOWED_PROVIDERS = {
        "conversation-buffer": "eurelis_llmatoolkit.langchain.memory.conversation_buffer_memory.ConversationBufferMemoryFactory",
    }

    def __init__(self):
        super().__init__()
        self.params["provider"] = "conversation-buffer"
