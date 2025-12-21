import importlib
from typing import List, TYPE_CHECKING
from llama_index.core.callbacks import CallbackManager

if TYPE_CHECKING:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler


class CallbackFactory:
    _built_in = {
        "ErrorLogging": "eurelis_llmatoolkit.llamaindex.callbacks.error_logging.VerboseErrorLoggingHandler",
        # Ajouter d'autres callbacks built-in ici
    }

    @staticmethod
    def create_callback_manager(configs: List[dict]) -> CallbackManager:
        """
        Creates a CallbackManager with multiple callbacks based on the provided configurations.

        Args:
            configs (List[dict]): List of callback configurations, each containing 'provider'
                                and any additional instantiation arguments.

        Returns:
            CallbackManager: Initialized with all specified callbacks.
        """
        callbacks = [CallbackFactory.create_callback(config) for config in configs]
        return CallbackManager(callbacks)

    @staticmethod
    def create_callback(config: dict) -> "BaseCallbackHandler":
        """
        Creates a single callback based on the provided configuration.

        Args:
            config (dict): Configuration for the callback, including 'provider'
                         and any additional instantiation arguments.

        Returns:
            BaseCallbackHandler: Instance of the callback class.
        """
        provider = config.get("provider")
        if not provider:
            raise ValueError("The 'provider' field is required in the configuration.")

        # Check for built-in callback or custom class
        real_provider = CallbackFactory._built_in.get(provider, provider)

        if real_provider.count(".") == 0:
            raise ValueError(
                "Provider attribute must reference a standard Callback short name "
                "or a fully qualified class path"
            )

        try:
            module_name, class_name = real_provider.rsplit(".", 1)
            module = importlib.import_module(module_name)
            callback_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import callback class '{provider}': {e}")

        # Remove provider from config before instantiation
        callback_config = config.copy()
        callback_config.pop("provider", None)

        return callback_class(**callback_config)
