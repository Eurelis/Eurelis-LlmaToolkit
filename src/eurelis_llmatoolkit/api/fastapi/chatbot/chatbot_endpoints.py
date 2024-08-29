from fastapi import APIRouter

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
logger = ConsoleManager().get_output()

router = APIRouter()
url_prefix = "/api/chat"


@router.get("/endpoint")
async def hello():
    logger.info(f"Endpoint route called with url_prefix: {url_prefix}")
    return {"Hello"}, 200