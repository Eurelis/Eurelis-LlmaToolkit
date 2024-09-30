import time

from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.api.service.session_manager import SessionManager
from eurelis_llmatoolkit.api.service.worker_manager import WorkerManager
from eurelis_llmatoolkit.api.worker.default_worker import DefaultWorker

logger = ConsoleManager().get_output()
base_config = BaseConfig()


def query(worker_factory, session_id, agent_id, query, src_page):

    # Session management
    if session_id is None:
        logger.debug(f"No session id provided, creating new")
        session_object = SessionManager.init_session(agent_id, src_page)
    else:
        logger.debug(f"Session id provided: {session_id}")
        session_object = SessionManager.load_session(session_id)

    if session_object is None:
        raise RuntimeError({"status": "error", "message": "Session not found"}, 404)

    status = SessionManager.compute_status(session_object["id"])

    if status in ("processing", "aborted", "terminated"):
        raise RuntimeError({"status": "error", "message": "Conflict"}, 409)

    current_timestamp = round(time.time() * 1000)

    if SessionManager.has_expired(session_object):
        logger.debug(f"Session {session_id} has expired, creating new")
        session_object = SessionManager.init_session(agent_id, src_page)

    logger.debug(
        f"Session running for: {(current_timestamp - int(session_object['timestamp'])) / 1000}"
    )

    #
    # Worker
    # session_object["status"] = "processing"
    process_id = str(current_timestamp)
    process_object = {
        "query": query,
        "status": "processing",
        "id": process_id,
        "started": SessionManager.compute_timestamp(),
        "session_id": session_object["id"],
        "responses": [],
    }
    # session_object["compute_history"][process_id] = chat_object
    SessionManager.save_process(process_object)

    agent_mode = AgentManager().get_agent_mode(agent_id)
    if worker_factory is None:
        worker = DefaultWorker(
            logger,
            agent_id,
            session_object["id"],
            process_object,
            agent_mode,
        )
    else:
        worker = worker_factory(
            logger,
            agent_id,
            session_object["id"],
            process_object,
            agent_mode,
        )
    WorkerManager().execute(worker)

    #
    # Response management
    response = {}
    response["session_id"] = session_object["id"]
    response["status"] = SessionManager.compute_status(session_object["id"])
    response["retry_after"] = (
        int(base_config.API_RETRY_AFTER) if response["status"] == "processing" else None
    )
    response["process_id"] = process_id
    response_code = 202 if response["status"] == "processing" else 200
    return response, response_code


def check_answer(session_id, process_id):
    #
    # Session management
    process_object = SessionManager.load_process(session_id, process_id)

    if process_object is None:
        logger.debug(f"Process {process_id} not found for session {session_id}")
        raise RuntimeError({"status": "error", "message": "Not Found"}, 404)

    response = {}
    response["status"] = SessionManager.compute_status(session_id)
    response["session_id"] = session_id
    response["process_id"] = process_id

    #
    # Build message history
    response["message_history"] = SessionManager.compute_message_history(session_id)

    #
    # Response management
    if process_object["status"] == "processing":
        response["retry_after"] = base_config.API_RETRY_AFTER
        response["solved_requested"] = False
        return response, 202
    else:
        response["solved_requested"] = process_object.get("solved_requested", False)

    return response, 200


def session(session_id):
    session_object = SessionManager.load_session(session_id)
    if session_object is None:
        return {"message": "Session not found"}, 404

    full_session_object = SessionManager.compute_full_session_object(session_id)
    return full_session_object, 200


def rate(session_id, rating):
    session_object = SessionManager.load_session(session_id)

    if session_object is None:
        return {"message": "Session not found"}, 404

    session_status = SessionManager.compute_status(session_id)

    if session_status == "processing" or "rating" in session_object:
        return {"message": "Conflict"}, 409

    session_object["rating"] = rating
    session_object["status"] = "terminated"
    SessionManager.save_session(session_object)

    return {"status": session_object["status"]}, 200


def solved(session_id, solved):
    session_object = SessionManager.load_session(session_id)

    if session_object is None:
        return {"message": "Session not found"}, 404

    session_status = SessionManager.compute_status(session_id)

    if session_status == "processing":
        return {"message": "Conflict"}, 409

    session_object["solved"] = solved
    SessionManager.save_session(session_object)

    return {"status": session_object["status"]}, 200
