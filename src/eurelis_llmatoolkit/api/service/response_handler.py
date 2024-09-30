from typing import Tuple
from eurelis_llmatoolkit.api.wrapper.llmatk_wrapper import LlmaTkWrapper
from eurelis_llmatoolkit.api.misc.singleton import Singleton


class ResponseHandler(metaclass=Singleton):

    def generate_llm_response(
        self, agent_id: str, session_id: str, query: str
    ) -> Tuple:
        """Generation de la réponse LLM à partir de la requête

        Args:
            query (str): requête client

        Returns:
            str: réponse LLM
        """

        llmatk_wrapper = LlmaTkWrapper()
        llmatk_memory = llmatk_wrapper.generate_response(agent_id, session_id, query)
        if (
            "source_documents" in llmatk_memory
            and len(llmatk_memory["source_documents"]) > 0
        ):
            source_page = LlmaTkWrapper.get_recommended_page(
                llmatk_memory["source_documents"]
            )
        else:
            source_page = None

        return (
            llmatk_memory["answer"]["answer"],
            llmatk_memory["chain_memory"],
            source_page,
        )
