from pydantic import BaseModel
from typing import TYPE_CHECKING, cast, List, Optional, Callable

from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.langchain import LangchainWrapperFactory
from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext
from eurelis_llmatoolkit.utils.output import Verbosity


if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import LangchainWrapper

logger = ConsoleManager().get_output()

def get_wrapper(verbose: bool, config: str) -> "LangchainWrapper":
    base_config = BaseConfig()
    factory = LangchainWrapperFactory()

    logger_log = os.getenv('LANGCHAIN_LOG', 'False').lower() == 'true'

    if verbose and logger_log:
        verbosity_level = Verbosity.LOG_DEBUG
    elif verbose:
        verbosity_level = Verbosity.CONSOLE_DEBUG
    elif logger_log:
        verbosity_level = Verbosity.LOG_INFO
    else:
        verbosity_level = Verbosity.CONSOLE_INFO

    factory.set_verbose(verbosity_level)
    
    if config:
        factory.set_config_path(config)

    factory.set_logger_config(base_config.get("LANGCHAIN_LOGGER_CONFIG", None), base_config.get("LANGCHAIN_LOGGER_NAME", None))

    instance = factory.build(cast(BaseContext, None))
    return instance

# Model to validate input for dataset commands
class DatasetCommandModel(BaseModel):
    dataset_id: str = None
    content_path: str = None

def dataset_index(
    command: DatasetCommandModel | None,
    agent_id: str,
    add_background_tasks: Callable,
    verbose: bool = False
):
    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
        logger.error(f"Error in dataset_index: No configuration file found for this agent")
        raise RuntimeError("No configuration file found for this agent", 500)
    wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

    dataset_id = command.dataset_id if command else None 
    content_path = command.content_path if command else None

    add_background_tasks(wrapper.index_documents, dataset_id, content_path)


def dataset_cache(
    command: DatasetCommandModel | None,
    agent_id: str,
    add_background_tasks: Callable,
    verbose: bool = False
):
    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
        logger.error(f"Error in dataset_index: No configuration file found for this agent")
        raise RuntimeError("No configuration file found for this agent", 500)

    wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

    dataset_id = command.dataset_id if command else None 

    add_background_tasks(wrapper.write_files, dataset_id)


def dataset_list(
    agent_id: str,
    verbose: bool = False
):
    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
        raise RuntimeError("Error in list_datasets: No configuration file found for this agent", 500)
    wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

    return {"datasets": wrapper.list_datasets()}


def dataset_clear(
    command: DatasetCommandModel | None,
    agent_id: str,
    add_background_tasks: Callable,
    verbose: bool = False
):
    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
      raise RuntimeError("Error in clear_datasets: No configuration file found for this agent", 500)

    wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

    dataset_id = command.dataset_id if command else None

    add_background_tasks(wrapper.clear_datasets, dataset_id)


# Model to validate input for delete and search commands
class FilterCommandModel(BaseModel):
    dataset_id: Optional[str] = None
    filters: Optional[List[str]] = None

def delete_content(
    command: FilterCommandModel | None,
    agent_id: str,
    add_background_tasks: Callable,
    verbose: bool = False
):
    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
        raise RuntimeError("Error in delete_content: No configuration file found for this agent", 500)

    wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

    dataset_id = command.dataset_id if command else None 
    filters = command.filters if command else None

    add_background_tasks(wrapper.clear_datasets, dataset_id)
    filter_args = parse_filters(filters)
    add_background_tasks(wrapper.delete, filter_args, dataset_id)


def unitary_delete(
    command: FilterCommandModel,
    agent_id: str,
    add_background_tasks: Callable,
    verbose: bool = False
):
    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
        raise RuntimeError("Error in unitary_delete: No configuration file found for this agent", 500)
    
    wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

    filters = command.filters if command else None

    filter_args = parse_filters(filters)
    add_background_tasks(wrapper.unitaryDelete, filter_args)


def parse_filters(filters: Optional[List[str]]) -> Optional[dict]:
    filter_args = {}
    if filters:
        for filter_arg in filters:
            if filter_arg:
                filter_split = filter_arg.split(":")
                filter_key = filter_split[0]
                filter_value = ":".join(filter_split[1:])
                filter_args[filter_key] = filter_value

    return filter_args if filter_args else None
