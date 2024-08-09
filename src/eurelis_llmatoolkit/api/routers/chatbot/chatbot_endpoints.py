import json
import time
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from fastapi import APIRouter, HTTPException, Security

# from chatbot.worker.graph_worker import GraphWorker

from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
from eurelis_llmatoolkit.api.model.api_model import AgentSearchListResponse
from eurelis_llmatoolkit.api.routers.security.security import token_required
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.api.service.rag import format_documents, get_wrapper, rerank

# from eurelis_llmatoolkit.utils.output import 

# from eurelis_llmatoolkit.api.service.session_manager import SessionManager
# from eurelis_llmatoolkit.api.service.worker_manager import WorkerManager
# from eurelis_llmatoolkit.api.worker.default_worker import DefaultWorker

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
logger = ConsoleManager.get_instance().get_output()

router = APIRouter()
url_prefix = "/api/chat"


@router.get("/hello")
async def hello(agent_id: str = Depends(token_required)):
    print(agent_id)
    logger.print(f"Hello route called with url_prefix: {url_prefix}")
    ui_params = AgentManager().get_ui_params(agent_id)
    return {"message": "Hello", "ui_params": ui_params}, 200


@router.get("/search")
def search(q: str, agent_id: str = Security(token_required)):
    if q == "" or q is None:
        raise HTTPException(
            status_code=400, detail="Query parameter 'q' cannot be empty"
        )
    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
        raise HTTPException(
            status_code=500, detail="No configuration file found for this agent"
        )
    wrapper = get_wrapper(llmatk_config)
    
    max_results = AgentManager().get_max_results(agent_id)

    results = wrapper.search_documents(
        q.lower(), k=max_results, include_relevance=True
    )

    results = rerank(q, results, score_threshold=0.0)

    prefix_url_img = AgentManager().get_prefixes_img(agent_id)

    formated = format_documents(results, prefix_url_img, max_results)

    return AgentSearchListResponse(content=formated[:max_results])


# TODO a terminer
# @router.post("/query/{session_id}")
# @router.post("/query")
# async def query(
#     request: Request,
#     session_id: str | None = None,
#     agent_id: str = Depends(token_required),
# ):
#     logger.print(f"Query route called with url_prefix: {url_prefix}")

#     #
#     # Request management
#     request_body = await request.json()

#     if not request_body:
#         return JSONResponse(
#             {"status": "error", "message": "No json provided"}, status_code=400
#         )

#     if "query" not in request_body:
#         return JSONResponse(
#             {"status": "error", "message": "No query provided"}, status_code=400
#         )

#     if not request_body["query"]:
#         return JSONResponse(
#             {"status": "error", "message": "Empty query provided"}, status_code=400
#         )

#     src_page = request_body.get("src_page", None)

#     #
#     # Session management
#     if not session_id:
#         logger.print(f"No session id provided, creating new")
#         session_object = SessionManager.init_session(agent_id, src_page)
#     else:
#         logger.print(f"Session id provided: {session_id}")
#         session_object = SessionManager.load_session(session_id)

#     if not session_object:
#         return JSONResponse(
#             {"status": "error", "message": "Session not found"}, status_code=404
#         )

#     status = SessionManager.compute_status(session_object["id"])

#     if status in ("processing", "aborted", "terminated"):
#         return JSONResponse({"status": "error", "message": "Conflict"}, status_code=409)

#     current_timestamp = round(time.time() * 1000)

#     if SessionManager.has_expired(session_object):
#         logger.print(f"Session {session_id} has expired, creating new")
#         session_object = SessionManager.init_session(agent_id, src_page)

#     logger.print(
#         f"Session running for: {(current_timestamp - session_object['timestamp']) / 1000}"
#     )

#     #
#     # Worker
#     process_id = str(current_timestamp)
#     process_object = {
#         "query": request_body["query"],
#         "status": "processing",
#         "id": process_id,
#         "started": SessionManager.compute_timestamp(),
#         "session_id": session_object["id"],
#         "responses": [],
#     }
#     SessionManager.save_process(process_object)
#     agent_mode = AgentManager().get_agent_mode(agent_id)

#     # if agent_mode in {"graph", "graph-subquery"}:
#     #     is_subquery = agent_mode == "graph-subquery"
#     #     worker = GraphWorker(
#     #         logger,
#     #         agent_id,
#     #         session_object["id"],
#     #         process_object,
#     #         is_subquery=is_subquery,
#     #     )
#     # else:
#     worker = DefaultWorker(
#         logger,
#         agent_id,
#         session_object["id"],
#         process_object,
#         agent_mode,
#     )

#     WorkerManager().execute(worker)

#     #
#     # Response management
#     response = {}
#     response["session_id"] = session_object["id"]
#     response["status"] = SessionManager.compute_status(session_object["id"])
#     response["retry_after"] = (
#         int(BaseConfig().API_RETRY_AFTER)
#         if response["status"] == "processing"
#         else None
#     )

#     response["process_id"] = process_id
#     response_code = 202 if response["status"] == "processing" else 200

#     return response, response_code


# TODO a terminer
# @router.get("/check_answer/{session_id}/{process_id}")
# async def check_answer(
#     session_id: str,
#     process_id: str,
#     request: Request,
#     agent_id: str = Depends(token_required),
# ):
#     logger.print(f"Check process {process_id} for session {session_id}")

#     #
#     # Session management
#     process_object = SessionManager.load_process(session_id, process_id)

#     if not process_object:
#         logger.print(f"Process {process_id} not found for session {session_id}")

#         return JSONResponse(
#             {"status": "error", "message": "Not Found"}, status_code=404
#         )

#     response = {}
#     response["status"] = SessionManager.compute_status(session_id)
#     response["session_id"] = session_id
#     response["process_id"] = process_id

#     #
#     # Build message history
#     response["message_history"] = SessionManager.compute_message_history(session_id)

#     #
#     # Response management
#     if process_object["status"] == "processing":
#         response["retry_after"] = BaseConfig().API_RETRY_AFTER
#         response["continue_requested"] = False

#         return response, 202
#     else:
#         response["continue_requested"] = process_object.get("continue_requested", False)

#     return response, 200


# @router.get("/session/{session_id}")
# async def session(
#     session_id: str,
#     agent_id: str = Depends(token_required),
# ):
#     logger.print(f"Session route called with url_prefix: {url_prefix}")
#     session_object = SessionManager.load_session(session_id)

#     if not session_object:
#         return JSONResponse({"message": "Session not found"}, status_code=404)

#     full_session_object = SessionManager.compute_full_session_object(session_id)

#     return full_session_object, 200


# @router.post("/rate/{session_id}")
# async def rate(
#     session_id: str,
#     request: Request,
#     agent_id: str = Depends(token_required),
# ):
#     logger.print(f"Rate route called with url_prefix: {url_prefix}")
#     request_body = await request.json()

#     if not request_body or "rating" not in request_body:
#         return JSONResponse({"message": "Bad Request"}, status_code=400)

#     rating = request_body["rating"]

#     # Vérifier si le rating est un entier entre 1 et 5
#     if not isinstance(rating, int) or rating not in range(1, 6):
#         return JSONResponse({"message": "Bad Request"}, status_code=400)

#     # Charger l'objet de session
#     session_object = SessionManager.load_session(session_id)

#     if not session_object:
#         return JSONResponse({"message": "Session not found"}, status_code=404)

#     # Vérifier l'état de la session et s'il y a déjà un rating
#     session_status = SessionManager.compute_status(session_id)

#     if session_status == "processing" or "rating" in session_object:
#         return JSONResponse({"message": "Conflict"}, status_code=409)

#     # Enregistrer le rating et mettre à jour l'état de la session
#     session_object["rating"] = rating
#     session_object["status"] = "terminated"
#     SessionManager.save_session(session_object)

#     return JSONResponse({"status": session_object["status"]}, status_code=200)
