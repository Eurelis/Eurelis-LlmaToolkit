from llama_index.core.memory import ChatMemoryBuffer


class MemoryFactory:
    @staticmethod
    def create_memory(config: dict):

        provider = config["provider"]

        #
        # Check for built-in memory
        #
        if provider == "ChatMemoryBuffer":
            return ChatMemoryBuffer.from_defaults(config["token_limit"])
        raise ValueError(f"Memory provider {provider} non supporté.")
