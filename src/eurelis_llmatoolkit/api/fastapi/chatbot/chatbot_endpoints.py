from typing import Any, Callable, Literal, Optional
from bson import ObjectId
from fastapi import APIRouter, Security, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from eurelis_llmatoolkit.api.fastapi.security.security import token_required
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.chatbot_service import (
    check_answer,
    query,
    rate,
    session,
    solved,
)
from eurelis_llmatoolkit.api.service.worker_manager import AbstractWorker


def create_chat_router(worker_factory: Callable[[], AbstractWorker] | None):

    logger = ConsoleManager().get_output()

    router = APIRouter()
    url_prefix = "/api/chat"

    class QueryRequest(BaseModel):
        query: str = Field(..., description="Query", min_length=1)
        src_page: Optional[str] = Field(None, description="Source page")

    @router.post("/query/")
    @router.post("/query/{session_id}")
    async def query_endpoint(
        request: QueryRequest,
        session_id: Optional[str] = None,
        agent_id: str = Security(token_required),
    ):
        try:
            logger.info(f"Query route called with url_prefix: {url_prefix}")
            logger.debug(
                f"query_endpoint with session_id: {session_id},  agent_id: {agent_id}, request: {request}"
            )

            response_body, status_code = query(
                worker_factory, session_id, agent_id, request.query, request.src_page
            )
            return JSONResponse(content=response_body, status_code=status_code)
        except Exception as e:
            if len(e.args) == 2:
                message, code = e.args
                raise HTTPException(status_code=code, detail=message)
            else:
                raise HTTPException(
                    status_code=500, detail="An unexpected error occurred"
                )

    @router.get("/check_answer/{session_id}/{process_id}")
    def check_answer_endpoint(
        session_id,
        process_id,
        agent_id: str = Security(token_required),
    ):
        logger.debug(f"Check process {process_id} for session {session_id}")

        try:
            logger.info(f"Check process route called with url_prefix: {url_prefix}")

            response_body, status_code = check_answer(session_id, process_id)
            return JSONResponse(content=response_body, status_code=status_code)
        except Exception as e:
            if len(e.args) == 2:
                message, code = e.args
                raise HTTPException(status_code=code, detail=message)
            else:
                raise HTTPException(
                    status_code=500, detail="An unexpected error occurred"
                )

    def serialize_object(obj: Any) -> Any:
        """Convertit les objets non sérialisables en JSON en types sérialisables."""
        if isinstance(obj, ObjectId):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: serialize_object(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [serialize_object(i) for i in obj]
        return obj

    @router.get("/session/{session_id}")
    def session_endpoint(
        session_id,
        agent_id: str = Security(token_required),
    ):
        logger.debug(f"Check session {session_id}")

        try:
            logger.info(f"Session route called with url_prefix: {url_prefix}")

            response_body, status_code = session(session_id)
            return JSONResponse(
                content=serialize_object(response_body), status_code=status_code
            )
        except Exception as e:
            if len(e.args) == 2:
                message, code = e.args
                raise HTTPException(status_code=code, detail=message)
            else:
                raise HTTPException(
                    status_code=500, detail="An unexpected error occurred"
                )

    class RateRequest(BaseModel):
        rating: int = Field(..., description="Rating", ge=1, le=5)

    @router.post("/rate/{session_id}")
    async def rate_endpoint(
        request: RateRequest,
        session_id: str,
        agent_id: str = Security(token_required),
    ):
        try:
            logger.info(f"Rate route called with url_prefix: {url_prefix}")

            response_body, status_code = rate(session_id, request.rating)
            return JSONResponse(content=response_body, status_code=status_code)
        except Exception as e:
            if len(e.args) == 2:
                message, code = e.args
                raise HTTPException(status_code=code, detail=message)
            else:
                raise HTTPException(
                    status_code=500, detail="An unexpected error occurred"
                )

    class SolvedRequest(BaseModel):
        solved: Literal["yes", "no", "partially"] = Field(
            ..., description="Solved status"
        )

    @router.post("/solved/{session_id}")
    async def solved_endpoint(
        request: SolvedRequest,
        session_id: str,
        agent_id: str = Security(token_required),
    ):
        try:
            logger.info(f"Solved route called with url_prefix: {url_prefix}")

            response_body, status_code = solved(session_id, request.solved)
            return JSONResponse(content=response_body, status_code=status_code)
        except Exception as e:
            if len(e.args) == 2:
                message, code = e.args
                raise HTTPException(status_code=code, detail=message)
            else:
                raise HTTPException(
                    status_code=500, detail="An unexpected error occurred"
                )

    return router
