from abc import ABC, abstractmethod
from typing import Optional

from llama_index.core.memory.types import BaseChatStoreMemory


class AbstractMemoryPersistenceHandler(ABC):
    def __init__(
        self,
        config: dict,
        memory: BaseChatStoreMemory,
        conversation_id: Optional[str] = None,
        persistence_config: Optional[dict] = None,
    ) -> None:
        self._config = config
        self._memory = memory
        self._conversation_id = conversation_id
        self._persistence_config = persistence_config

    @abstractmethod
    def load_history(self) -> None:
        pass

    @abstractmethod
    def save_history(self) -> None:
        pass

    def set_memory(self, memory: BaseChatStoreMemory) -> None:
        self._memory = memory
