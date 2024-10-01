from abc import ABC
from collections import OrderedDict
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from eurelis_llmatoolkit.llamaindex.dataset import DatasetFactory

from eurelis_llmatoolkit.llamaindex.types import FACTORY
from eurelis_llmatoolkit.llamaindex.utils.class_loader import ClassLoader

from llama_index.core.base.llms.base import BaseLLM

if TYPE_CHECKING:
    from llama_index.core.vector_stores.types import BasePydanticVectorStore
    from llama_index.core.base.embeddings.base import BaseEmbedding


class BaseContext(ABC):
    """
    Base context class
    """

    def __init__(self, class_loader: ClassLoader, console=None):
        self.loader = class_loader
        self.console = console
        self.opt_embeddings: Optional["BaseEmbedding"] = None
        self.opt_vector_store: Optional["BasePydanticVectorStore"] = None
        self.is_verbose = False

    def copy_context(self) -> "BaseContext":
        new_context = BaseContext(self.loader)
        new_context.console = self.console
        new_context.opt_embeddings = self.opt_embeddings
        new_context.opt_vector_store = self.opt_vector_store
        new_context.is_verbose = self.is_verbose

        return new_context

    @property
    def embeddings(self) -> "BaseEmbedding":
        if not self.opt_embeddings:
            raise ValueError("No embeddings provided")

        return self.opt_embeddings

    @property
    def vector_store(self) -> "BasePydanticVectorStore":
        if not self.opt_vector_store:
            raise ValueError("No vectorstore provided")

        return self.opt_vector_store


class IndexationWrapper(BaseContext):
    """
    Indexation wrapper, main class of the project
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        super().__init__(ClassLoader())
        self._datasets: Optional[OrderedDict] = None
        self._datasets_data: Optional[Union[List[dict], dict]] = None
        self.index_fn = None
        self.opt_project: Optional[str] = None
        # self.opt_record_manager_db_url: Optional[str] = None  # TODO trouver equivalent
        self.llm: Optional[BaseLLM] = None
        self.llm_factory: Optional[FACTORY] = None
        self.chain_factory: Optional[FACTORY] = None
        self.vector_store_data: Optional[FACTORY] = None
        self.is_initialized = False
        self._acronyms_data = None
        self._acronyms = None

    @property
    def project(self) -> str:
        if not self.opt_project:
            raise RuntimeError("project was not set")
        return self.opt_project

    def ensure_initialized(self):
        """
        Method to ensure the wrapper is initialized

        Raise:
            ValueError if wrapper is not initialized
        """
        if not self.is_initialized:
            raise ValueError("Indexation wrapper is not initialized")

    def set_output(self, console):
        """
        Setter for console
        Args:
            console: output object to print on the console

        Returns:

        """
        self.console = console

    def set_verbose(self, verbose):
        """
        Setter for verbose parameter
        Args:
            verbose: default to False

        Returns:

        """
        self.is_verbose = verbose

    def load_config(self, path):
        """
        Load the configuration from a json file
        Args:
            path: path of the json configuration file

        Returns:

        """

        with open(path) as config_file:
            try:
                config = json.load(config_file)
            except json.decoder.JSONDecodeError as e:
                self.console.critical_print(f"Error parsing config file: {e}")
                return

            # TODO
            self.opt_project = parse_param_value(
                config.get("project", "knowledge_base")
            )

            # TODO
            self._parse_embeddings(config.get("embeddings"))
            self._parse_vector_store(config.get("vectorstore"))

            self._datasets_data = config.get("dataset", [])
            self._acronyms_data = config.get("acronyms", None)

            self.llm_factory = config.get("llm")
            self.chain_factory = config.get("chain", {})

            # TODO : Trouver équivalent
            # self.opt_record_manager_db_url = parse_param_value(
            #     config.get("record_manager", "sqlite:///record_manager_cache.sql")
            # )

            # sqlite_prefix = "sqlite:///"

            # if self.record_manager_db_url.startswith(sqlite_prefix):
            #     sqlite_length = len(sqlite_prefix)
            #     path = self.record_manager_db_url[sqlite_length:]
            #     path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
            #     file_folder = Path(os.path.dirname(path))
            #     os.makedirs(file_folder, exist_ok=True)

            self.is_initialized = True

    @property
    def datasets(self):
        if not self._datasets:
            self._datasets = self.console.status(
                f"Parsing datasets",
                lambda: DatasetFactory.build_instances(
                    self, self._datasets_data
                ),  # TODO
            )

        return self._datasets
