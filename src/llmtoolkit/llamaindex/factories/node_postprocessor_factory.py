import importlib

from llama_index.core.postprocessor.types import BaseNodePostprocessor


class NodePostProcessorFactory:
    _built_in = {
        "MetadataReplacementPostProcessor": "llama_index.core.postprocessor.MetadataReplacementPostProcessor",
    }

    @staticmethod
    def create_node_postprocessor(config: dict):

        provider = config["provider"]

        # Filter the provider key in the config to initialize the postprocessor
        postprocessor_params = {
            key: value for key, value in config.items() if key != "provider"
        }

        # Check for built-in retriever or custom class
        real_provider = NodePostProcessorFactory._built_in.get(provider, provider)

        #
        # If the provider is a custom Postprocessor
        #
        if real_provider.count(".") == 0:
            raise ValueError(
                "Provider attribute must reference a standard Postprocessor short name or a fully qualified class path"
            )

        # Extract module and class from custom Postprocessor path
        try:
            module_name, class_name = real_provider.rsplit(".", 1)
            module = importlib.import_module(module_name)
            postprocesseur_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import retriever class '{provider}': {e}")

        return postprocesseur_class(**postprocessor_params)

    @staticmethod
    def create_node_postprocessors(
        configs: list[dict],
    ) -> list[BaseNodePostprocessor] | None:
        return [
            NodePostProcessorFactory.create_node_postprocessor(config)
            for config in configs
        ]
