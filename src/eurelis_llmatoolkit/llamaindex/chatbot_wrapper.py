from typing import TYPE_CHECKING, Optional
from llama_index.core.query_engine import RetrieverQueryEngine

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper
from eurelis_llmatoolkit.llamaindex.factories.chat_engine_factory import (
    ChatEngineFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.memory_persistence_factory import (
    MemoryPersistenceFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.memory_factory import MemoryFactory
from eurelis_llmatoolkit.llamaindex.factories.llm_factory import (
    LLMFactory,
)

if TYPE_CHECKING:
    from llama_index.core.base.llms.base import BaseLLM
    from llama_index.core.memory import BaseMemory


class ChatbotWrapper(AbstractWrapper):
    def __init__(self, config: dict):
        super().__init__(config)
        self._llm: "BaseLLM" = None
        self._query_engine: Optional[RetrieverQueryEngine] = None
        self._memory: "BaseMemory" = None
        self._chat_engine = None
        self._memory_persistence = None

    def run(
        self,
        conversation_id: str,
        message: str,
    ):
        """
        Runs the chatbot by initializing the vector store, storage context,
        index, retriever, query engine, and memory.

        Args:
            conversation_id: str, ID of the current conversation.
        """
        # Création d'un chat_engine pour avoir une conversation avec id
        chat_engine = self._get_chat_engine(chat_store_key=conversation_id)

        # Pour sauver et restaurer la conversation
        memory = self._get_memory()
        memory_persistence = self._get_memory_persistence(memory)
        memory_persistence.load_history()

        response = chat_engine.chat(message)

        # Sauvegarder les conversations
        memory_persistence.save_history()

        # Si EmptyResponse est retourné, vérifier la création de vector_index dans la BDD vectorielle
        return response

    def _get_memory(self, chat_store_key: str | None = None):
        """
        Creates a BaseMemory.

        Returns:
            BaseMemory: The configured memory.
        """
        if self._memory is not None:
            return self._memory

        memory_config = self._config["chat_engine"].get("memory")
        if memory_config and chat_store_key is not None:
            self._memory = MemoryFactory.create_memory(memory_config, chat_store_key)
        return self._memory

    def _get_llm(self):
        """
        Retrieves the language model (LLM) from the configuration.

        Returns:
            BaseLLM: The configured language model.
        """
        if self._llm is not None:
            return self._llm

        llm_config = self._config.get("llm")
        if llm_config:
            self._llm = LLMFactory.create_llm(llm_config)

        return self._llm

    def _get_chat_engine(self, chat_store_key: str):
        """
        Creates a chat engine using the index, chat mode, memory, and system prompt.

        Returns:
            ChatEngine: The configured chat engine.
        """
        if self._chat_engine is not None:
            return self._chat_engine

        llm = self._get_llm()
        index = self._get_vector_store_index()
        memory = self._get_memory(chat_store_key)

        # Create the chat engine with the specified configuration
        chat_engine_config = self._config["chat_engine"]
        system_prompt_list = chat_engine_config.get("system_prompt")

        if isinstance(system_prompt_list, list):
            system_prompt = "\n".join(system_prompt_list)
        elif isinstance(system_prompt_list, str):
            system_prompt = system_prompt_list
        else:
            raise ValueError(
                "The 'system_prompt' should be either a list of strings or a single string."
            )

        # Par défaut, utilise le LLM d'OPENAI si non précisé
        retriever = self._get_retriever(config=chat_engine_config)

        chat_engine = ChatEngineFactory.create_chat_engine(chat_engine_config)
        self._chat_engine = chat_engine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=memory,
            system_prompt=system_prompt,
        )

        return self._chat_engine

    def _get_memory_persistence(self, memory=None):
        """
        Get or create a memory persistence.

        If a memory persistence already exists, return it. If not, and if
        a configuration is available, create one using the provided memory.

        Args:
            memory (Optional[BaseChatStoreMemory]): Memory instance for initializing
            the memory persistence, if needed.

        Returns:
            MemoryPersistence: The memory persistence instance.
        """
        if self._memory_persistence is not None:
            return self._memory_persistence

        memory_persistence_config = self._config["chat_engine"].get(
            "memory_persistence"
        )
        if memory_persistence_config and memory is not None:
            self._memory_persistence = (
                MemoryPersistenceFactory.create_memory_persistence(
                    memory_persistence_config, memory
                )
            )

        return self._memory_persistence
