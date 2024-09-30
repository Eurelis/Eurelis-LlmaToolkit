from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.api.service.rag import format_documents, get_wrapper

from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from typing import List

logger = ConsoleManager().get_output()
MAX_RESULTS = 3

def similarity(urls: List[str], agent_id: str):
    # Clear duplicates
    urls = list(set(urls))

    base_config = BaseConfig()

    if len(urls) < 3:
        return []

    llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
    if llmatk_config is None:
        raise RuntimeError("No configuration file found for this agent", 500)

    wrapper = get_wrapper(llmatk_config)


    # Cleaning for debug (FIXME: To delete)
    cleaned_urls = []
    for url in urls:
        new_url = url.replace(
            "https://local.eurelis.com:8080", "https://www.int.eurelis.info"
        )
        cleaned_urls.append(new_url)
    urls = cleaned_urls
    # End cleaning for debug

    urls = urls[:3]  # Limit to 3 URLs

    # TODO: Add coefs to the request to get the associated documents
    results = wrapper.get_embedding_associated_documents(
        *urls,
        k=MAX_RESULTS,
        coefs=None,
    )
    prefix_url_img = AgentManager().get_prefixes_img(agent_id)
    formated = format_documents(results, prefix_url_img, MAX_RESULTS)
    return formated
