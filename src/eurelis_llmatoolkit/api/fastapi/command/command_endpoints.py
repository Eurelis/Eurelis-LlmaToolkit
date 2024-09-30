import os
from fastapi import APIRouter, HTTPException, BackgroundTasks, Security
from typing import TYPE_CHECKING, cast

from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
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
url_prefix = "/api/command"

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


# API route to index documents
@router.post("/dataset/index")
async def dataset_index_endpoint(
    command: DatasetCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    logger.info(f"dataset_index route called with url_prefix: {url_prefix}")
    try:
        dataset_index(command, agent_id, background_tasks.add_task, verbose)
        return {"status": "Indexing started"}, 202
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")


# API route to Print first doc metadata
@router.post("/dataset/cache")
async def dataset_cache_endpoint(
    command: DatasetCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        logger.info(f"dataset_cache route called with url_prefix: {url_prefix}")
        dataset_cache(command, agent_id, background_tasks.add_task, verbose)
        return {"status": "Writing cache files for metadata"}, 202
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

# API route to List dataset
@router.post("/dataset/list")
async def dataset_list_endpoint(
    verbose: bool = False,
    agent_id: str = Security(token_required)
):
    try:
        logger.info(f"dataset_list route called with url_prefix: {url_prefix}")
        datasets = dataset_list(agent_id, verbose)
        return datasets, 202
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")



# API route to Clear dataset
@router.post("/dataset/clear")
async def dataset_clear_endpoint(
    command: DatasetCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        logger.info(f"dataset_clear route called with url_prefix: {url_prefix}")
        dataset_clear(command, agent_id, background_tasks.add_task, verbose)
        return {"datasets": "Cleaning dataset started"}, 202
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")


# API route for Delete content from database using a query
@router.post("/delete")
async def delete_content_endpoint(
    command: FilterCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        logger.info(f"delete_content route called with url_prefix: {url_prefix}")
        delete_content(command, agent_id, background_tasks.add_task, verbose)
        return {"status": "Delete operation started"}, 202
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")


# API route for unitarydelete command
@router.post("/unitarydelete")
async def unitary_delete_endpoint(
    command: FilterCommandModel = None,
    verbose: bool = False,
    agent_id: str = Security(token_required),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    try:
        logger.info(f"unitary_delete route called with url_prefix: {url_prefix}")
        unitary_delete(command, agent_id, background_tasks.add_task, verbose)
        return {"status": "Unitary delete operation started"}, 202
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")
