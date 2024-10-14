from llama_index.llms.openai import (
    OpenAI,
)


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
            return OpenAI(model=config.get("model", "gpt-4o"))

        raise ValueError(f"LLM provider {provider} non supporté.")
