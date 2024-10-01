import os
import logging
from enum import Enum
from typing import TYPE_CHECKING, Optional, Union

from eurelis_llmatoolkit.llamaindex.utils.base_factory import BaseFactory
from eurelis_llmatoolkit.llamaindex.utils.output.base_console_output import (
    BaseConsoleOutput,
)
from eurelis_llmatoolkit.llamaindex.utils.output.output import Output

if TYPE_CHECKING:
    from eurelis_llmatoolkit.llamaindex.indexation_wrapper import BaseContext


class Verbosity(Enum):
    CONSOLE_INFO = "console-info"
    CONSOLE_DEBUG = "console-debug"
    LOG_INFO = "log-info"
    LOG_DEBUG = "log-debug"


VERBOSE_VALUE = Union[Optional[bool], Verbosity]


class OutputFactory(BaseFactory[Output]):
    """
    Console output factory
    """

    def __init__(self) -> None:
        self.verbosity_level: Verbosity = Verbosity.LOG_INFO
        self.logger_config_path: str | None = None
        self.logger_name: str | None = None

    def set_verbose(self, verbose: VERBOSE_VALUE):
        """
        setter for the verbose property
        Args:
            verbose (None, boolean, Verbosity): if None or True will use log_info, if False will use log_debug

        Returns:

        """
        if verbose is None:
            self.verbosity_level = Verbosity.LOG_INFO
        elif isinstance(verbose, bool):
            self.verbosity_level = (
                Verbosity.LOG_DEBUG if verbose else Verbosity.LOG_INFO
            )
        elif not isinstance(verbose, Verbosity):
            raise ValueError(
                f"Invalid verbose parameter type, expecting None, True, False or Verbosity enum value, got {type(verbose)}"
            )
        else:
            self.verbosity_level = verbose

    def set_logger_config(
        self, logger_config_path: str | None, logger_name: str | None = None
    ):
        """
        setter for the logger_config_path and logger_name properties
        Args:
            logger_config_path (str, None)

        Returns:

        """
        self.logger_name = logger_name

        if logger_config_path is None:
            self.logger_config_path = None
        elif os.path.exists(logger_config_path):
            self.logger_config_path = logger_config_path
        else:
            raise ValueError(
                f"Invalid logger_config_path parameter, expecting a valid path."
            )

    def build(self, context: "BaseContext") -> Output:
        """
        Method to construct a BaseConsoleOutput
        Args:
            context: context object, usually the current instance of langchain_wrapper

        Returns:
            instance of BaseConsoleOutput or of a class inheriting it
        """
        if self.verbosity_level == Verbosity.LOG_INFO:
            from eurelis_llmatoolkit.llamaindex.utils.output.logging_console_output import (
                LoggingConsoleOutput,
            )

            return LoggingConsoleOutput(
                logging.INFO, self.logger_config_path, self.logger_name
            )
        elif self.verbosity_level == Verbosity.LOG_DEBUG:
            from eurelis_llmatoolkit.utils.output.logging_console_output import (
                LoggingConsoleOutput,
            )

            return LoggingConsoleOutput(
                logging.DEBUG, self.logger_config_path, self.logger_name
            )
        elif self.verbosity_level == Verbosity.CONSOLE_DEBUG:
            from eurelis_llmatoolkit.utils.output.verbose_console_output import (
                VerboseConsoleOutput,
            )

            return VerboseConsoleOutput()

        return BaseConsoleOutput()
