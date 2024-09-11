from fastapi import APIRouter, Depends, Body, HTTPException
from pydantic import BaseModel, Field
from typing import List

from eurelis_llmatoolkit.api.fastapi.security.security import token_required
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.similarity_service import similarity

logger = ConsoleManager().get_output()

router = APIRouter()
url_prefix = "/api/similarity"

class SimilarityQuery(BaseModel):
    urls: List[str] = Field(..., description="List of URLs")


@router.post("/similarity")
async def similarity_endpoint(agent_id: str = Depends(token_required), query: SimilarityQuery = Body(...),):
    try:
        logger.info(f"Endpoint route called with url_prefix: {url_prefix}")
        
        data = query.model_dump()
        
        urls = data.get("urls", [])
        if not urls:
            raise HTTPException(status_code=400, detail="Request body must contain a 'urls' field")

        result = similarity(urls, agent_id)

        return result
    except Exception as e:
        if len(e.args) == 2:
            message, code = e.args
            raise HTTPException(status_code=code, detail=message)
        else:
            raise HTTPException(status_code=500, detail="An unexpected error occurred")