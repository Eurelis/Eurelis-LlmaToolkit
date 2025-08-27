import logging
from typing import TYPE_CHECKING, List, Optional, Union, Callable

from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import BaseTool

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper
from eurelis_llmatoolkit.llamaindex.factories.llm_factory import LLMFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_factory import MemoryFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_persistence_factory import (
    MemoryPersistenceFactory,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from llama_index.core.base.llms.base import BaseLLM
    from llama_index.core.memory import BaseMemory


class ReActWrapper(AbstractWrapper):
    def __init__(
        self,
        config: dict,
        conversation_id: str,
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        persistence_data: Optional[dict] = None,
        custom_system_prompt=None,
    ):
        super().__init__(config)
        self._llm: Optional["BaseLLM"] = None
        self._memory: Optional["BaseMemory"] = None
        self._memory_persistence = None
        self._persistence_data = persistence_data
        self._conversation_id = conversation_id
        self._custom_system_prompt = custom_system_prompt

        logger.info(
            "ChatbotWrapper initialized with conversation_id: %s", conversation_id
        )

        # Création d'un chat_engine
        self._react_agent = self._create_react_agent(
            chat_store_key=conversation_id,
            tools=tools,
            custom_system_prompt=custom_system_prompt,
        )

    async def run(self, message: str):
        """
        Runs the chatbot by initializing the vector store, storage context,
        index, retriever, query engine, and memory.

        Args:
            conversation_id: str, ID of the current conversation.
        """
        logger.info("Running chatbot with message: %s", message)

        # Récupération du react_agent instancié
        react_agent = self._get_react_agent()
        response = await react_agent.run(message, memory=self._memory)
        logger.debug("Chatbot response: %s", response)

        # Sauvegarder l'historique des conversations mises à jour en utilisant la mémoire du chat_engine
        self._save_memory(self._memory)

        return response

    def _initialize_memory(self, chat_store_key: str) -> "BaseMemory":
        """
        Initializes a memory instance and loads its conversation history.

        Args:
            chat_store_key (str): The unique key associated with the conversation.

        Returns:
            BaseMemory: A memory instance, with loaded conversation history.
        """
        logger.info("Initializing memory with chat_store_key: %s", chat_store_key)
        memory_config = self._config["react_agent"].get("memory")
        if not memory_config:
            logger.error("Memory configuration is missing in react_agent settings.")
            raise ValueError("Memory configuration is missing in react_agent settings.")

        # Création d'une instance de mémoire vide
        memory = MemoryFactory.create_memory(memory_config, chat_store_key)

        # Chargement des conversations dans la mémoire
        memory_persistence = self._get_memory_persistence(memory, chat_store_key)
        if memory_persistence is not None:
            memory_persistence.load_history()
            logger.debug("Memory history loaded for chat_store_key: %s", chat_store_key)

        return memory_persistence._memory if memory_persistence else memory

    def _get_memory_persistence(self, memory=None, chat_store_key: str = None):
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
        logger.debug(
            "Getting memory persistence for chat_store_key: %s", chat_store_key
        )
        # Si la persistance de la mémoire existe déjà et qu'aucune nouvelle mémoire n'est fournie
        if self._memory_persistence is not None:
            # Si une mémoire est fournie, on met à jour la persistance avec cette nouvelle mémoire
            if memory is not None:
                self._memory_persistence.set_memory(memory)
                logger.debug("Memory persistence updated with new memory.")
            return self._memory_persistence

        # Si la persistance de mémoire n'existe pas, et qu'une mémoire est fournie
        if memory is None:
            logger.error("Memory is required when creating a new memory persistence.")
            raise ValueError(
                "Memory is required when creating a new memory persistence."
            )

        # Créer une nouvelle persistance de mémoire à partir de la configuration
        memory_persistence_config = self._config["react_agent"].get(
            "memory_persistence"
        )
        if not memory_persistence_config:
            self._memory_persistence = None
            logger.warning("Memory persistence configuration is missing.")
            return None

        self._memory_persistence = MemoryPersistenceFactory.create_memory_persistence(
            memory_persistence_config, memory, chat_store_key, self._persistence_data
        )

        logger.debug(
            "Memory persistence created for chat_store_key: %s", chat_store_key
        )
        return self._memory_persistence

    def _get_llm(self):
        """
        Retrieves the language model (LLM) from the configuration.

        Returns:
            BaseLLM: The configured language model.
        """
        logger.debug("Retrieving language model (LLM) from configuration.")
        if self._llm is not None:
            return self._llm

        llm_config = self._config.get("llm")
        if llm_config:
            self._llm = LLMFactory.create_llm(llm_config)

        return self._llm

    def _get_prompt(self, react_agent_config, custom_prompt=None):
        """
        Generates the system prompt for the ReAct agent.

        Args:
            react_agent_config (dict): The ReAct agent configuration containing system prompts.
            custom_prompt (str, optional): A custom prompt to use. If not provided, the prompt will be generated from the configuration.

        Returns:
            str: The generated system prompt.
        """
        logger.debug("Get system prompt for ReAct agent.")
        if custom_prompt is not None:
            return custom_prompt

        system_prompt_list = react_agent_config.get("system_prompt")

        if isinstance(system_prompt_list, list):
            return "\n".join(system_prompt_list)
        elif isinstance(system_prompt_list, str):
            return system_prompt_list
        elif system_prompt_list is None:
            return None
        else:
            logger.error("Invalid 'system_prompt' format in configuration.")
            raise ValueError(
                "The 'system_prompt' should be either a list of strings or a single string."
            )

    def _create_react_agent(
        self,
        chat_store_key: str,
        tools: Optional[List[Union[BaseTool, Callable]]],
        custom_system_prompt=None,
    ):
        """
        Create and configure a ReAct agent with memory and LLM.

        Args:
            chat_store_key (str): Unique key to identify the chat history store.
            custom_system_prompt (str, optional): Custom prompt to override the default system prompt.

        Returns:
            ReActAgent: The fully configured ReActAgent instance.
        """
        logger.debug("Creating ReActAgent with chat_store_key: %s", chat_store_key)

        react_agent_config: dict = self._config["react_agent"]
        system_prompt = self._get_prompt(react_agent_config, custom_system_prompt)

        # Initialisation de la mémoire avec chargement de l'historique
        self._memory = self._initialize_memory(chat_store_key)

        llm = self._get_llm()

        react_agent = ReActAgent(
            name=react_agent_config.get("name", "ReAct Agent"),
            description=react_agent_config.get(
                "description", "An agent that uses ReAct workflow."
            ),
            system_prompt=system_prompt,
            tools=tools,
            llm=llm,
        )

        # Remarque : Dans ReActAgent, l'argument system_prompt n'est pas directement utilisé pour la logique principale du prompt.
        # Le prompt 'react_header' contient les instructions principales du workflow ReAct.
        # Le prompt 'react_header' semble prévaloir sur 'system_prompt' dans la réponse.
        # Pour personnaliser l'agent, on préfixe notre system_prompt au template existant de react_header,
        # afin de conserver la logique ReAct originale tout en injectant nos instructions personnalisées.

        # Récupérer le react_header actuel
        react_header = react_agent.get_prompts()["react_header"]
        # Ajouter le system_prompt avant le template existant
        react_header.template = system_prompt + react_header.template
        # Mettre à jour l’agent
        react_agent.update_prompts({"react_header": react_header})

        logger.info("ReAct Agent created successfully.")
        return react_agent

    def _get_react_agent(self):
        """
        Retrieve the configured ReAct agent.

        Args:

        Raises:
            ValueError: If `_react_agent` is not initialized.

        Returns:
            ReActAgent: The ReAct agent.
        """
        logger.debug("Retrieving ReAct agent with filters and custom system prompt.")

        if self._react_agent is None:
            logger.error(
                "The '_react_agent' must be initialized using the '_create_react_agent' method."
            )
            raise ValueError(
                "The '_react_agent' must be initialized using the '_create_react_agent' method."
            )

        return self._react_agent

    def _save_memory(self, memory):
        """
        Save the current state of the memory's conversation history.

        Args:
            memory (BaseMemory): Memory instance to save.
        """
        logger.debug("Saving memory state.")
        if not memory:
            logger.error("Cannot save history: memory is not provided.")
            raise ValueError("Cannot save history: memory is not provided.")

        memory_persistence = self._get_memory_persistence(memory)
        if memory_persistence is not None:
            memory_persistence.save_history()
            logger.info("Memory history saved successfully.")

    def update_tools(self, tools: Optional[List[Union["BaseTool", Callable]]]):
        """
        Re-create the react agent with the new tools.
        """
        self._react_agent = self._create_react_agent(
            chat_store_key=self._conversation_id,
            tools=tools,
            custom_system_prompt=self._custom_system_prompt,
        )
