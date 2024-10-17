import importlib
import inspect


class ReaderFactory:
    @staticmethod
    def create_reader(config: dict, namespace: str):
        provider_path = config["provider"]

        if provider_path.count(".") == 0:
            raise ValueError("Reader provider path must contain one dot")

        module_name, class_name = provider_path.rsplit(".", 1)

        module = importlib.import_module(module_name)

        reader_class = getattr(module, class_name)

        init_params = inspect.signature(reader_class.__init__)

        if "namespace" in init_params.parameters:
            return reader_class(config, namespace)
        return reader_class(config)
