import json

from fastapi import APIRouter, HTTPException, BackgroundTasks, Security
from pydantic import BaseModel
from typing import TYPE_CHECKING, cast, List, Optional

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.routers.security.security import token_required
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.langchain import LangchainWrapperFactory
from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext
from eurelis_llmatoolkit.utils.output import Verbosity


if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import LangchainWrapper

logger = ConsoleManager.get_instance().get_output()

router = APIRouter()

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

# API route to index documents
@router.post("/dataset/index")
async def dataset_index(
    command: DatasetCommandModel,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        background_tasks.add_task(wrapper.index_documents, command.dataset_id, command.content_path)
        print_datasets_data(wrapper._datasets_data)
        return {"status": "Indexing started"}, 202
    except Exception as e:
        logger.print(f"Error in dataset_index: {e}")
        raise HTTPException(status_code=500, detail=str("Error in dataset_index"))


def print_datasets_data(data):
    print("wrapper")
    if data is None:
        print("None")
    elif isinstance(data, list):
        for item in data:
            print(item)
    elif isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {value}")
    else:
        print("Unknown data type")

    # print(json.dump(wrapper._datasets_data))
    out_file = open("final_myfile.json", "w")   
    json.dump(data, out_file, indent = 6)
    out_file.close()


# API route to Print first doc metadata
@router.post("/dataset/cache")
async def dataset_cache(
    command: DatasetCommandModel,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        background_tasks.add_task(wrapper.write_files, command.dataset_id)
        return {"status": "Writing cache files for metadata"}, 202
    except Exception as e:
        logger.print(f"Error in write_files: {e}")
        raise HTTPException(status_code=500, detail=str("Error in write_files"))


# API route to List dataset
@router.post("/dataset/list")
async def dataset_list(
    verbose: bool = False,
    agent_id: str = Security(token_required)
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        return {"datasets": wrapper.list_datasets()}, 202
    except Exception as e:
        logger.print(f"Error in list_datasets: {e}")
        raise HTTPException(status_code=500, detail=str("Error in list_datasets"))


# API route to Clear dataset
@router.post("/dataset/clear")
async def dataset_clear(
    command: DatasetCommandModel,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        background_tasks.add_task(wrapper.clear_datasets, command.dataset_id)
        
        return {"datasets": "Cleaning dataset started"}, 202
    except Exception as e:
        logger.print(f"Error in clear_datasets: {e}")
        raise HTTPException(status_code=500, detail=str("Error in clear_datasets"))


# Model to validate input for delete and search commands
class FilterCommandModel(BaseModel):
    dataset_id: Optional[str] = None
    filters: Optional[List[str]] = None

# API route for Delete content from database using a query
@router.post("/delete")
async def delete_content(
    command: FilterCommandModel,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        background_tasks.add_task(wrapper.clear_datasets, command.dataset_id)
        filter_args = parse_filters(command.filters)
        background_tasks.add_task(wrapper.delete, filter_args, command.dataset_id)
        return {"status": "Delete operation started"}, 202
    except Exception as e:
        logger.print(f"Error in delete_content: {e}")
        raise HTTPException(status_code=500, detail=str("Error in delete_content"))

# API route for unitarydelete command
@router.post("/unitarydelete")
async def unitary_delete(
    command: FilterCommandModel,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
        if llmatk_config is None:
            raise HTTPException(
                status_code=500, detail="No configuration file found for this agent"
            )
        wrapper = get_wrapper(verbose, f"config/{llmatk_config}")

        filter_args = parse_filters(command.filters)
        background_tasks.add_task(wrapper.unitaryDelete, filter_args)
        return {"status": "Unitary delete operation started"}, 202
    except Exception as e:
        logger.print(f"Error in unitary_delete: {e}")
        raise HTTPException(status_code=500, detail=str("Error in unitary_delete"))


class SearchCommandModel(BaseModel):
    query: str
    filters: Optional[List[str]] = None

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
