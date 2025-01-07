import importlib


class RetrieverFactory:
    _built_in = {
        "VectorIndexRetriever": "llama_index.core.retrievers.VectorIndexRetriever",
    }

    @staticmethod
    def create_retriever(config: dict):
        """
        Creates a retriever based on the provided configuration.

        Args:
            config (dict): Configuration for the retriever, including parameters
            like 'provider' and any additional instantiation arguments.

        Returns:
            Instance of the retriever class.
        """
        provider = config.get("provider")
        if not provider:
            raise ValueError("The 'provider' field is required in the configuration.")

        # Check for built-in retriever or custom class
        real_provider = RetrieverFactory._built_in.get(provider, provider)

        #
        # If the provider is a custom retriever
        #
        if real_provider.count(".") == 0:
            raise ValueError(
                "Provider attribute must reference a standard Retriever short name or a fully qualified class path"
            )

        # Extract module and class from custom provider path
        try:
            module_name, class_name = real_provider.rsplit(".", 1)
            module = importlib.import_module(module_name)
            retriever_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import retriever class '{provider}': {e}")

        # Instantiate the retriever with the parameters passed in `config`
        return retriever_class(**config)
