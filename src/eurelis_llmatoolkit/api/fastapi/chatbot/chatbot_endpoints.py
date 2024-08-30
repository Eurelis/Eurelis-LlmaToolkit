from fastapi import APIRouter

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.chatbot_service import hello

logger = ConsoleManager().get_output()

router = APIRouter()
url_prefix = "/api/chat"

# TODO Implémentation chatbot

@router.get("/hello-endpoint")
async def hello_endpoint():
    logger.info(f"Endpoint route called with url_prefix: {url_prefix}")
    result = hello()
    return result, 200