class LLMFactory:
    @staticmethod
    def create_llm(config: dict):
        """
        Create an instance of an LLM from a given configuration.

        Args:
            config (dict): Configuration for the LLM.
                Must include at least:
                  - provider (str): Name of the provider ("OpenAI", "Anthropic", ...).

                All other keys are forwarded directly to the constructor
                of the selected LLM. Supported parameters depend on the provider, e.g.:
                  - OpenAI    : model, temperature, max_tokens, reasoning_effort, ...
                  - Anthropic : model, temperature, max_tokens, ...

        Returns:
            LLM: Instance of the selected LLM.
        """
        provider = config.get("provider")
        if not provider:
            raise ValueError("The 'provider' field is required in the config.")

        # Exclude `provider` from constructor arguments
        kwargs = {k: v for k, v in config.items() if k != "provider"}

        if provider == "OpenAI":
            from llama_index.llms.openai import OpenAI

            kwargs.setdefault("model", "gpt-4o")  # default value
            return OpenAI(**kwargs)

        if provider == "Anthropic":
            from llama_index.llms.anthropic import Anthropic

            kwargs.setdefault("model", "claude-3-5-sonnet-20241022")  # default value
            return Anthropic(**kwargs)

        raise ValueError(f"LLM provider '{provider}' is not supported.")
