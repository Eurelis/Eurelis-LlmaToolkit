import importlib
import inspect
from llama_index.core.postprocessor.types import BaseNodePostprocessor


class NodePostProcessorFactory:

    @staticmethod
    def create_node_postprocessor(config: dict):

        provider = config["provider"]

        # Filter the provider key in the config to initialize the postprocessor
        postprocessor_params = {
            key: value for key, value in config.items() if key != "provider"
        }

        #
        # Check for built-in Postprocesseur
        #
        if provider == "MetadataReplacementPostProcessor":
            from llama_index.core.postprocessor import MetadataReplacementPostProcessor

            return MetadataReplacementPostProcessor(**postprocessor_params)

        #
        # If the provider is a custom Postprocesseur
        #
        if provider.count(".") == 0:
            raise ValueError(
                "Provider attribute must reference a standard Postprocesseur short name or a fully qualified class path"
            )

        module_name, class_name = provider.rsplit(".", 1)

        module = importlib.import_module(module_name)

        postprocesseur_class = getattr(module, class_name)

        init_params = inspect.signature(postprocesseur_class.__init__)

        if "namespace" in init_params.parameters:
            return postprocesseur_class(**postprocessor_params)
        return postprocesseur_class(**postprocessor_params)

    @staticmethod
    def create_node_postprocessors(
        configs: list[dict],
    ) -> list[BaseNodePostprocessor] | None:
        return [
            NodePostProcessorFactory.create_node_postprocessor(config)
            for config in configs
        ]
