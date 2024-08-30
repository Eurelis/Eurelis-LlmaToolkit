from fastapi import HTTPException

from eurelis_llmatoolkit.api.model.api_model import AgentSearchListResponse
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.api.service.rag import format_documents, get_wrapper, rerank

from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
logger = ConsoleManager().get_output()

def hello(agent_id: str):
    ui_params = AgentManager().get_ui_params(agent_id)
    return {"message": "Hello", "ui_params": ui_params}, 200

def search(q: str, agent_id: str):
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