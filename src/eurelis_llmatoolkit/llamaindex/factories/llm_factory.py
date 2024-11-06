class LLMFactory:
    @staticmethod
    def create_llm(config: dict):
        """
        Crée une instance d'un LLM à partir de la configuration.

        Args:
            config (dict): Configuration pour le LLM.

        Returns:
            LLM: Instance du LLM.
        """

        provider = config["provider"]
        if provider == "OpenAI":
            from llama_index.llms.openai import OpenAI

            return OpenAI(model=config.get("model", "gpt-4o"))

        if provider == "Anthropic":
            from llama_index.llms.anthropic import Anthropic

            return Anthropic(model=config.get("model", "claude-3-5-sonnet-20241022"))

        raise ValueError(f"LLM provider {provider} is not supported.")
