class ChatEngineFactory:
    @staticmethod
    def create_chat_engine(chat_engine_config: dict):
        provider = chat_engine_config.get("provider")

        if provider == "ContextChatEngine":
            from llama_index.core.chat_engine import ContextChatEngine

            return ContextChatEngine

        raise ValueError(f"ChatEngine provider {provider} is not supported.")
