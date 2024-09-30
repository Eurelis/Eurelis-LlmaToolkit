from fastapi import APIRouter, Security, Depends, HTTPException

from eurelis_llmatoolkit.api.fastapi.security.security import token_required
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.search_service import hello, search

logger = ConsoleManager().get_output()

router = APIRouter()
url_prefix = "/api/search"

@router.get("/hello")
async def hello_endpoint(agent_id: str = Depends(token_required)):
    try:
        logger.info(f"Hello route called with url_prefix: {url_prefix}")
        return hello(agent_id)
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")

@router.get("/search")
def search_endpoint(q: str, agent_id: str = Security(token_required)):
    try:
        logger.info(f"Search route called with url_prefix: {url_prefix}")
        return search(q, agent_id)
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")