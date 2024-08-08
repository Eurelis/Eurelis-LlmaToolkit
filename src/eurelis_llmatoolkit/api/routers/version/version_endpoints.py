from fastapi import APIRouter, Request
from datetime import datetime
import os
import platform

from eurelis_llmatoolkit.api.misc.version_info import VersionInfo 

router = APIRouter()

@router.get("/info")
async def info(request: Request):
    # Retourne la version de l'application au format JSON
    return {
        "application": {
            "version_info": VersionInfo.get_version_info_dict(),
            "base_directory": os.getcwd(),
        },
        "system": {
            "system_time": datetime.now().astimezone().isoformat(),
            "system_name": platform.system(),
            "python_version": platform.python_version(),
        },
        "context": {
            "request_url_root": request.base_url,
            "request_host": request.client.host if request.client else None,
        },
    }
