from fastapi import APIRouter, Security, Depends

from eurelis_llmatoolkit.api.fastapi.security.security import token_required
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.search_service import hello, search

logger = ConsoleManager().get_output()

router = APIRouter()
url_prefix = "/api/search"

@router.get("/hello")
async def hello_endpoint(agent_id: str = Depends(token_required)):
    logger.info(f"Hello route called with url_prefix: {url_prefix}")
    return hello(agent_id)

@router.get("/search")
def search_endpoint(q: str, agent_id: str = Security(token_required)):
    return search(q, agent_id)
