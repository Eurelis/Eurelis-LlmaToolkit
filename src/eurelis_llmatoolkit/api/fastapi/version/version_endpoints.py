from fastapi import APIRouter, Request

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.version_service import get_system_info

logger = ConsoleManager().get_output()

router = APIRouter()

url_prefix = "/api/version"

@router.get("/info")
async def info_endpoint(request: Request):
    """Retourne les informations système et de l'application au format JSON."""
    logger.info(f"Info route called with url_prefix: {url_prefix}")
    return get_system_info(request)
