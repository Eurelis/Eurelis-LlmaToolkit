class MemoryFactory:
    @staticmethod
    def create_memory(config: dict, chat_store_key: str):

        provider = config["provider"]

        #
        # Check for built-in memory
        #
        if provider == "ChatMemoryBuffer":
            from llama_index.core.memory import ChatMemoryBuffer

            return ChatMemoryBuffer.from_defaults(
                token_limit=config["token_limit"], chat_store_key=chat_store_key
            )
        raise ValueError(f"Memory provider {provider} is not supported.")
