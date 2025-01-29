import logging
import json
from typing import Any, Dict

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseChatStoreMemory

from eurelis_llmatoolkit.llamaindex.chat_memory_persistence.abstract_memory_persistence_handler import (
    AbstractMemoryPersistenceHandler,
)

logger = logging.getLogger(__name__)


class JSONPersistenceHandler(AbstractMemoryPersistenceHandler):
    def __init__(
        self,
        config: dict,
        memory: BaseChatStoreMemory,
        conversation_id: str = None,
    ) -> None:
        super().__init__(config, memory)
        self._filename = config["persist_conversation_path"]
        self._conversation_id = conversation_id

    def load_history(self) -> None:
        """Loads conversation history from a JSON file, filtered by self._conversation_id if set."""
        if self._conversation_id is None:
            raise ValueError("Conversation ID is required.")

        try:
            with open(self._filename, "r") as f:
                data = json.load(f)
                conversations = data.get("conversations", {})

                if self._conversation_id:
                    # If self._conversation_id is defined, load only this conversation
                    messages = conversations.get(self._conversation_id, [])
                    if messages:
                        chat_messages = [
                            ChatMessage(
                                role=MessageRole(msg["role"]), content=msg["content"]
                            )
                            for msg in messages
                        ]
                        # Store converted messages in memory
                        self._memory.chat_store.set_messages(
                            self._conversation_id, chat_messages
                        )
                    else:
                        self._memory.chat_store.set_messages(self._conversation_id, [])

        except FileNotFoundError:
            logger.info("No history found. Create a new history.")

    def save_history(self) -> None:
        """Saves the history of all conversations in a JSON file."""
        history_dict: Dict[str, Any] = {"conversations": {}}

        # Load existing conversations from JSON file
        try:
            with open(self._filename, "r") as f:
                history_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # If the file doesn't exist or any other error occurs, we continue with a new history

        # Retrieve all conversation IDs
        conversation_ids = self._memory.chat_store.get_keys()

        # Save messages from each conversation
        for key in conversation_ids:
            messages = self._memory.chat_store.get_messages(key)
            history_dict["conversations"][str(key)] = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

        # Write all history to JSON file
        with open(self._filename, "w") as f:
            json.dump(history_dict, f, indent=4)
