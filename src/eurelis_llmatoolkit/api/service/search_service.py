from eurelis_llmatoolkit.api.model.api_model import AgentSearchListResponse
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.api.service.rag import format_documents, get_wrapper, rerank

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
logger = ConsoleManager().get_output()

def hello(agent_id: str):
    ui_params = AgentManager().get_ui_params(agent_id)
    max_history = AgentManager().get_max_history(agent_id)
    is_active = AgentManager().is_authorized(agent_id)
    is_search_active = AgentManager().is_search_active(agent_id)

    return {
        "message": "Hello",
        "is_active": is_active == "authorized",
        "is_search_active": is_search_active,
        "max_history": max_history,
        "ui_params": ui_params
    }

def search(q: str, agent_id: str):
    if q == "" or q is None:
        raise ValueError("Query parameter 'q' cannot be empty", 400)

    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
        raise RuntimeError("No configuration file found for this agent", 500)

    wrapper = get_wrapper(llmatk_config)
    
    max_results = AgentManager().get_max_results(agent_id)

    results = wrapper.search_documents(
        q.lower(), k=max_results, include_relevance=True
    )

    results = rerank(q, results, score_threshold=0.0)

    prefix_url_img = AgentManager().get_prefixes_img(agent_id)

    formated = format_documents(results, prefix_url_img, max_results)

    return AgentSearchListResponse(content=formated[:max_results])