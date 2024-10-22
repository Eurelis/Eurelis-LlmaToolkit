import json
from typing import Any, Dict

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.memory.types import BaseChatStoreMemory


class JSONConversationManager:
    def __init__(
        self,
        config: dict,
        memory: BaseChatStoreMemory,
    ) -> None:
        self.memory = memory
        self.filename = config["persist_conversation_path"]

    def load_history(self) -> None:
        """Loads conversation history from a JSON file."""
        try:
            with open(self.filename, "r") as f:
                data = json.load(f)
                for conversation_id, messages in data.get("conversations", {}).items():
                    # Convert each JSON message into a ChatMessage object
                    chat_messages = [
                        ChatMessage(
                            role=MessageRole(msg["role"]), content=msg["content"]
                        )
                        for msg in messages
                    ]

                    # Store converted messages in memory
                    self.memory.chat_store.set_messages(conversation_id, chat_messages)
        except FileNotFoundError:
            print("No history found. Create a new history.")

    def save_history(self) -> None:
        """Saves the history of all conversations in a JSON file."""
        history_dict: Dict[str, Any] = {"conversations": {}}

        # Load existing conversations from JSON file
        try:
            with open(self.filename, "r") as f:
                history_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # If the file doesn't exist or any other error occurs, we continue with a new history

        # Retrieve all conversation IDs (you might need a method for this)
        conversation_ids = self.memory.chat_store.get_keys()

        # Save messages from each conversation
        for key in conversation_ids:
            messages = self.memory.chat_store.get_messages(key)
            history_dict["conversations"][str(key)] = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]

        # Write all history to JSON file
        with open(self.filename, "w") as f:
            json.dump(history_dict, f, indent=4)
