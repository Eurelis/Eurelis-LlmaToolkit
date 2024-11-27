from typing import TYPE_CHECKING, Optional

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper
from eurelis_llmatoolkit.llamaindex.factories.chat_engine_factory import (
    ChatEngineFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.llm_factory import LLMFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_factory import MemoryFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_persistence_factory import (
    MemoryPersistenceFactory,
)
from llama_index.core.vector_stores import (
    MetadataFilters,
    FilterCondition,
)

if TYPE_CHECKING:
    from llama_index.core.base.llms.base import BaseLLM
    from llama_index.core.memory import BaseMemory


class ChatbotWrapper(AbstractWrapper):
    def __init__(
        self,
        config: dict,
        conversation_id: str,
        permanent_filters: Optional["MetadataFilters"] = None,
    ):
        super().__init__(config)
        self._llm: "BaseLLM" = None
        self._memory: "BaseMemory" = None
        self._memory_persistence = None
        self._permanent_filters: Optional["MetadataFilters"] = permanent_filters

        # Création d'un chat_engine
        self._chat_engine = self._create_chat_engine(chat_store_key=conversation_id)

    def run(
        self,
        message: str,
        filters: Optional["MetadataFilters"] = None,
        filter_condition: Optional["FilterCondition"] = None,
        custom_system_prompt=None,
    ):
        """
        Runs the chatbot by initializing the vector store, storage context,
        index, retriever, query engine, and memory.

        Args:
            conversation_id: str, ID of the current conversation.
        """
        # Récupération du chat_engine instancié
        chat_engine = self._get_chat_engine(
            filters=filters,
            filter_condition=filter_condition,
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
        if memory_persistence is not None:
            memory_persistence.load_history()

        return memory_persistence._memory if memory_persistence else memory

    def _get_memory_persistence(self, memory=None):
        """
        Retrieve or create a memory persistence instance.

            - If a memory persistence already exists and no new memory is provided, return it.
            - If a memory persistence exists and a memory is provided, update the persistence with the new memory.
            - If memory persistence does not exist and a memory is provided, create a new memory persistence.
            - If memory persistence does not exist and no memory is provided, raises an error.
            - If the configuration for memory persistence is missing, set `_memory_persistence` to None.

            Args:
                memory (Optional[BaseChatStoreMemory]): Memory instance to initialize or update the persistence if needed.

            Returns:
                Optional[MemoryPersistence]: The memory persistence instance, or None if not configured.
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
            self._memory_persistence = None
            return None

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

    def _create_chat_engine(
        self,
        chat_store_key: str,
        custom_system_prompt=None,
    ):
        """
        Create and configure a chat engine with memory, retriever, and LLM.

        Args:
            chat_store_key (str): Unique key to identify the chat history store.
            custom_system_prompt (str, optional): Custom prompt to override the default system prompt.

        Returns:
            ChatEngine: The fully configured chat engine instance.
        """

        chat_engine_config = self._config["chat_engine"]
        system_prompt = self._get_prompt(chat_engine_config, custom_system_prompt)

        # Initialisation de la mémoire avec chargement de l'historique
        self._memory = self._initialize_memory(chat_store_key)

        llm = self._get_llm()
        retriever = self._get_retriever(config=chat_engine_config)

        chat_engine = ChatEngineFactory.create_chat_engine(chat_engine_config)
        self._chat_engine = chat_engine.from_defaults(
            retriever=retriever,
            llm=llm,
            memory=self._memory,
            system_prompt=system_prompt,
        )

        return self._chat_engine

    def _get_chat_engine(
        self,
        filters: Optional[MetadataFilters] = None,
        filter_condition: Optional[FilterCondition] = None,
        custom_system_prompt=None,
    ):
        """
        Retrieve the configured chat engine, optionally applying metadata filters.

        Args:
            filters (MetadataFilters, optional): Filters to combine with `_permanent_filters` for data retrieval.
            filter_condition (FilterCondition, optional): Logical condition (AND, OR) to combine filters.
            custom_system_prompt (str, optional): Custom prompt to override the default system prompt.

        Raises:
            ValueError: If `_chat_engine` is not initialized.
            AttributeError: If the retriever does not support filters.

        Returns:
            ChatEngine: The chat engine with applied filters (if supported).
        """

        chat_engine_config = self._config["chat_engine"]
        system_prompt = self._get_prompt(chat_engine_config, custom_system_prompt)

        if self._chat_engine is None:
            raise ValueError(
                "The '_chat_engine' must be initialized using the '_create_chat_engine' method."
            )

        if hasattr(self._chat_engine._retriever, "_filters"):
            # Combinaison des filtres
            combined_filters = None

            if self._permanent_filters or filters:
                # Fusion des filtres permanents et des filtres personnalisés
                permanent_filters = (
                    self._permanent_filters.filters if self._permanent_filters else []
                )
                custom_filters = filters.filters if filters else []

                combined_filters = MetadataFilters(
                    filters=permanent_filters + custom_filters,
                    condition=filter_condition,
                )

            # Appliquer les filtres combinés si existants
            self._chat_engine._retriever._filters = combined_filters
        else:
            raise AttributeError(
                "The '_filters' attribute is not available for this retriever."
            )

            # TODO : System prompt
            # self._chat_engine._prefix_messages = system_prompt

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
        if memory_persistence is not None:
            memory_persistence.save_history()
