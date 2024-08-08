from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-Agent-ID")

def token_required(api_key_header: str = Security(api_key_header)):
    if api_key_header:
        access = AgentManager().is_authorized(api_key_header)
        if access == "unauthorized":
            raise HTTPException(status_code=401, detail="Unauthorized")
        if access == "forbidden":
            raise HTTPException(status_code=403, detail="Forbidden")
        else:
            return api_key_header
    else:
        raise HTTPException(status_code=403, detail="Forbidden")
