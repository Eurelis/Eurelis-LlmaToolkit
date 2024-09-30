from datetime import datetime
import os
import platform

from eurelis_llmatoolkit.api.misc.version_info import VersionInfo

def get_system_info(request):
    """Retourne un dictionnaire avec les informations système et de l'application."""
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
