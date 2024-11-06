from typing import TYPE_CHECKING, Optional
from llama_index.core.query_engine import RetrieverQueryEngine

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper
from eurelis_llmatoolkit.llamaindex.factories.conversation_manager_factory import (
    ConversationManagerFactory,
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
        self._conversation_manager = None

    def run(
        self,
        conversation_id: str,
        message: str,
        dataset_id: Optional[str] = None,
        use_cache: bool = False,
    ):
        """
        Runs the chatbot by initializing the vector store, storage context,
        index, retriever, query engine, and memory.

        Args:
            conversation_id: str, ID of the current conversation.
            dataset_id: Optional, ID of the dataset to process.
            use_cache: Optional, flag to indicate if caching should be used.
        """
        # Création d'un chat_engine pour avoir une conversation avec id
        chat_engine = self._get_chat_engine(chat_store_key=conversation_id)

        # Pour sauver et restaurer la conversation
        memory = self._get_memory()
        conversation_manager = self._get_conversation_manager(memory)
        conversation_manager.load_history()

        response = chat_engine.chat(message)

        # Sauvegarder les conversations
        conversation_manager.save_history()

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

        memory_config = self._config.get("memory")
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

    def _get_query_engine(self, retriever, llm):
        """
        Creates a query engine using the retriever and language model.

        Args:
            retriever: The retriever to be used for querying.
            llm: The language model to be used for querying.

        Returns:
            RetrieverQueryEngine: The configured query engine.
        """
        if self._query_engine is not None:
            return self._query_engine

        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever, llm=llm
        )

        return self._query_engine

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
        chat_mode = self._config.get("chat_mode")
        system_prompt_list = self._config.get("system_prompt")

        if isinstance(system_prompt_list, list):
            system_prompt = "\n".join(system_prompt_list)
        elif isinstance(system_prompt_list, str):
            system_prompt = system_prompt_list
        else:
            raise ValueError(
                "The 'system_prompt' should be either a list of strings or a single string."
            )

        # Par défaut, utilise le LLM d'OPENAI si non précisé
        self._chat_engine = index.as_chat_engine(
            chat_mode=chat_mode,
            llm=llm,
            memory=memory,
            system_prompt=system_prompt,
        )

        return self._chat_engine

    def _get_conversation_manager(self, memory=None):
        """
        Get or create a conversation manager.

        If a conversation manager already exists, return it. If not, and if
        a configuration is available, create one using the provided memory.

        Args:
            memory (Optional[BaseChatStoreMemory]): Memory instance for initializing
            the conversation manager, if needed.

        Returns:
            ConversationManager: The conversation manager instance.
        """
        if self._conversation_manager is not None:
            return self._conversation_manager

        conversation_manager_config = self._config.get("conversation_manager")
        if conversation_manager_config and memory is not None:
            self._conversation_manager = (
                ConversationManagerFactory.create_conversation_manager(
                    conversation_manager_config, memory
                )
            )

        return self._conversation_manager
