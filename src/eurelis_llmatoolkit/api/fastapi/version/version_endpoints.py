from fastapi import APIRouter, Request

from eurelis_llmatoolkit.api.service.version_service import get_system_info

router = APIRouter()

@router.get("/info")
async def info_endpoint(request: Request):
    """Retourne les informations système et de l'application au format JSON."""
    return get_system_info(request)
