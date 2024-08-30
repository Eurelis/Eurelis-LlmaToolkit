from fastapi import APIRouter, HTTPException, BackgroundTasks, Security
from typing import TYPE_CHECKING, cast

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.fastapi.security.security import token_required
from eurelis_llmatoolkit.langchain import LangchainWrapperFactory
from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext
from eurelis_llmatoolkit.utils.output import Verbosity

from eurelis_llmatoolkit.api.service.command_service import (
    dataset_index,
    dataset_cache,
    dataset_list,
    dataset_clear,
    delete_content,
    unitary_delete,
    DatasetCommandModel,
    FilterCommandModel
)

if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import LangchainWrapper

logger = ConsoleManager().get_output()

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


# API route to index documents
@router.post("/dataset/index")
async def dataset_index_endpoint(
    command: DatasetCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        dataset_index(command, agent_id, background_tasks.add_task, verbose)
        return {"status": "Indexing started"}, 202
    except Exception as e:
        logger.error(f"Error in dataset_index: {e}")
        raise HTTPException(status_code=500, detail=str("Error in dataset_index"))


# API route to Print first doc metadata
@router.post("/dataset/cache")
async def dataset_cache_endpoint(
    command: DatasetCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    dataset_cache(command, agent_id, background_tasks.add_task, verbose)
    return {"status": "Writing cache files for metadata"}, 202


# API route to List dataset
@router.post("/dataset/list")
async def dataset_list_endpoint(
    verbose: bool = False,
    agent_id: str = Security(token_required)
):
    datasets = dataset_list(agent_id, verbose)
    return datasets, 202


# API route to Clear dataset
@router.post("/dataset/clear")
async def dataset_clear_endpoint(
    command: DatasetCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    dataset_clear(command, agent_id, background_tasks.add_task, verbose)
    return {"datasets": "Cleaning dataset started"}, 202


# API route for Delete content from database using a query
@router.post("/delete")
async def delete_content_endpoint(
    command: FilterCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    delete_content(command, agent_id, background_tasks.add_task, verbose)
    return {"status": "Delete operation started"}, 202


# API route for unitarydelete command
@router.post("/unitarydelete")
async def unitary_delete_endpoint(
    command: FilterCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    unitary_delete(command, agent_id, background_tasks.add_task, verbose)
    return {"status": "Unitary delete operation started"}, 202
