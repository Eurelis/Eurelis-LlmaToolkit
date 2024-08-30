from fastapi import HTTPException
from pydantic import BaseModel
from typing import TYPE_CHECKING, cast, List, Optional, Callable

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.langchain import LangchainWrapperFactory
from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext
from eurelis_llmatoolkit.utils.output import Verbosity


if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import LangchainWrapper

logger = ConsoleManager().get_output()

def get_wrapper(verbose: bool, config: str) -> "LangchainWrapper":
    factory = LangchainWrapperFactory()
    if verbose:
        factory.set_verbose(Verbosity.CONSOLE_DEBUG)
    else:
        factory.set_verbose(Verbosity.CONSOLE_INFO)

    if config:
        factory.set_config_path(config)

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
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        dataset_id = command.dataset_id if command else None 
        content_path = command.content_path if command else None

        add_background_tasks(wrapper.index_documents, dataset_id, content_path)
    except Exception as e:
        logger.error(f"Error in dataset_index: {e}")
        raise HTTPException(status_code=500, detail=str("Error in dataset_index"))


def dataset_cache(
    command: DatasetCommandModel | None,
    agent_id: str,
    add_background_tasks: Callable,
    verbose: bool = False
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        dataset_id = command.dataset_id if command else None 

        add_background_tasks(wrapper.write_files, dataset_id)

    except Exception as e:
        logger.error(f"Error in write_files: {e}")
        raise HTTPException(status_code=500, detail=str("Error in write_files"))


def dataset_list(
    agent_id: str,
    verbose: bool = False
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        return {"datasets": wrapper.list_datasets()}
    except Exception as e:
        logger.error(f"Error in list_datasets: {e}")
        raise HTTPException(status_code=500, detail=str("Error in list_datasets"))


def dataset_clear(
    command: DatasetCommandModel | None,
    agent_id: str,
    add_background_tasks: Callable,
    verbose: bool = False
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        dataset_id = command.dataset_id if command else None

        add_background_tasks(wrapper.clear_datasets, dataset_id)
    except Exception as e:
        logger.error(f"Error in clear_datasets: {e}")
        raise HTTPException(status_code=500, detail=str("Error in clear_datasets"))


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
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        dataset_id = command.dataset_id if command else None 
        filters = command.filters if command else None

        add_background_tasks(wrapper.clear_datasets, dataset_id)
        filter_args = parse_filters(filters)
        add_background_tasks(wrapper.delete, filter_args, dataset_id)
    except Exception as e:
        logger.error(f"Error in delete_content: {e}")
        raise HTTPException(status_code=500, detail=str("Error in delete_content"))


def unitary_delete(
    command: FilterCommandModel,
    agent_id: str,
    add_background_tasks: Callable,
    verbose: bool = False
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        filters = command.filters if command else None

        filter_args = parse_filters(filters)
        add_background_tasks(wrapper.unitaryDelete, filter_args)
    except Exception as e:
        logger.error(f"Error in unitary_delete: {e}")
        raise HTTPException(status_code=500, detail=str("Error in unitary_delete"))


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
