from typing import TYPE_CHECKING, Optional

from llama_index.core.query_engine import RetrieverQueryEngine

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper
from eurelis_llmatoolkit.llamaindex.factories.chat_engine_factory import (
    ChatEngineFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.llm_factory import LLMFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_factory import MemoryFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_persistence_factory import (
    MemoryPersistenceFactory,
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
        filters=None,
        custom_system_prompt=None,
    ):
        """
        Runs the chatbot by initializing the vector store, storage context,
        index, retriever, query engine, and memory.

        Args:
            conversation_id: str, ID of the current conversation.
        """
        # Création d'un chat_engine
        chat_engine = self._get_chat_engine(
            chat_store_key=conversation_id,
            filters=filters,
            custom_system_prompt=custom_system_prompt,
        )
        response = chat_engine.chat(message)

        # Sauvegarder l'historique des conversations mises à jour en utilisant la mémoire du chat_engine
        self._save_memory(chat_engine._memory)

        return response

    def _initialize_memory(self, chat_store_key: str) -> "BaseMemory":
        """
        Initializes a memory instance and loads its conversation history.

        Args:
            chat_store_key (str): The unique key associated with the conversation.

        Returns:
            BaseMemory: A memory instance, with loaded conversation history.
        """
        memory_config = self._config["chat_engine"].get("memory")
        if not memory_config:
            raise ValueError("Memory configuration is missing in chat_engine settings.")

        # Création d'une instance de mémoire vide
        memory = MemoryFactory.create_memory(memory_config, chat_store_key)

        # Chargement des conversations dans la mémoire
        memory_persistence = self._get_memory_persistence(memory)
        memory_persistence.load_history()

        return memory_persistence._memory

    def _get_memory_persistence(self, memory=None):
        """
        Retrieve or create a memory persistence instance.

            - If a memory persistence already exists and no new memory is provided, return it.
            - If a memory persistence exists and a memory is provided, update the persistence with the new memory.
            - If memory persistence does not exist and a memory is provided, create a new memory persistence.
            - If memory persistence does not exist and no memory is provided, raises an error.

            Args:
                memory (Optional[BaseChatStoreMemory]): Memory instance to initialize or update the persistence if needed.

            Returns:
                MemoryPersistence: The memory persistence instance.

        """
        # Si la persistance de la mémoire existe déjà et qu'aucune nouvelle mémoire n'est fournie
        if self._memory_persistence is not None:
            # Si une mémoire est fournie, on met à jour la persistance avec cette nouvelle mémoire
            if memory is not None:
                self._memory_persistence._memory = memory
            return self._memory_persistence

        # Si la persistance de mémoire n'existe pas, et qu'une mémoire est fournie
        if memory is None:
            raise ValueError(
                "Memory is required when creating a new memory persistence."
            )

        # Créer une nouvelle persistance de mémoire à partir de la configuration
        memory_persistence_config = self._config["chat_engine"].get(
            "memory_persistence"
        )
        if not memory_persistence_config:
            raise ValueError("Memory persistence configuration is missing.")

        self._memory_persistence = MemoryPersistenceFactory.create_memory_persistence(
            memory_persistence_config, memory
        )

        return self._memory_persistence

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

    def _get_prompt(self, chat_engine_config, custom_prompt=None):
        """
        Generates the system prompt for the chat engine.

        Args:
            chat_engine_config (dict): The chat engine configuration containing system prompts.
            custom_prompt (str, optional): A custom prompt to use. If not provided, the prompt will be generated from the configuration.

        Returns:
            str: The generated system prompt.
        """
        if custom_prompt is not None:
            return custom_prompt

        system_prompt_list = chat_engine_config.get("system_prompt")

        if isinstance(system_prompt_list, list):
            return "\n".join(system_prompt_list)
        elif isinstance(system_prompt_list, str):
            return system_prompt_list
        elif system_prompt_list is None:
            return None
        else:
            raise ValueError(
                "The 'system_prompt' should be either a list of strings or a single string."
            )

    def _get_chat_engine(
        self, chat_store_key: str, filters=None, custom_system_prompt=None
    ):
        """
        Creates and configures a chat engine.

        Args:
            chat_store_key (str): Key to access the memory chat store.
            filters (optional): Filters to apply for data retrieval (MetadataFilters).
            custom_system_prompt (str, optional): A custom prompt to use instead of the one configured.

        Returns:
            ChatEngine: The configured chat engine.
        """
        llm = self._get_llm()

        # Initialisation de la mémoire avec chargement de l'historique
        self._memory = self._initialize_memory(chat_store_key)

        chat_engine_config = self._config["chat_engine"]
        system_prompt = self._get_prompt(chat_engine_config, custom_system_prompt)
        retriever = self._get_retriever(config=chat_engine_config, filters=filters)

        chat_engine = ChatEngineFactory.create_chat_engine(chat_engine_config)
        self._chat_engine = chat_engine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=self._memory,
            system_prompt=system_prompt,
        )

        return self._chat_engine

    def _save_memory(self, memory):
        """
        Save the current state of the memory's conversation history.

        Args:
            memory (BaseMemory): Memory instance to save.
        """
        if not memory:
            raise ValueError("Cannot save history: memory is not provided.")

        memory_persistence = self._get_memory_persistence(memory)
        memory_persistence.save_history()
