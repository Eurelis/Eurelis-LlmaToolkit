from abc import ABC, abstractmethod

from llama_index.core.memory.types import BaseChatStoreMemory


class AbstractMemoryPersistenceHandler(ABC):
    def __init__(
        self,
        config: dict,
        memory: BaseChatStoreMemory,
    ) -> None:
        self._config = config
        self._memory = memory

    @abstractmethod
    def load_history(self) -> None:
        pass

    @abstractmethod
    def save_history(self) -> None:
        pass
