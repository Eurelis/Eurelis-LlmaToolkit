# import threading
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.api.service.response_handler import ResponseHandler
from eurelis_llmatoolkit.api.service.richcontent_manager import RichContentManager
from eurelis_llmatoolkit.api.service.session_manager import SessionManager
from eurelis_llmatoolkit.api.service.worker_manager import AbstractWorker


class DefaultWorker(AbstractWorker):
    def __init__(
        self,
        logger,
        agent_id: str,
        session_id: str,
        process_object: dict,
        agent_mode: str,
    ):
        super().__init__(logger)
        self._session_id = session_id
        self._agent_id = agent_id
        self._process_object = process_object
        self._agent_mode = agent_mode

    def _llm_treatment(self, session_object):
        """Query the LlmaTk and generate a response

        Args:
            session_object (dict): session object
        """
        self._logger.debug(f"[{self._process_object['id']}] llm treatment launched")

        try:
            session_object["status"] = "completed"
            self._process_object["agent"] = "llm"
            (
                response,
                llmatk_answer,
                source_page,
            ) = ResponseHandler().generate_llm_response(
                self._agent_id, self._session_id, self._process_object["query"]
            )
            self._process_object["response_id"] = (
                SessionManager.compute_timestamp_as_str()
            )
            self._process_object["created"] = SessionManager.compute_timestamp()

            self._process_object["response"] = response
            self._process_object["llm"] = llmatk_answer
            if source_page is not None:
                self._process_object["rich_content"] = (
                    RichContentManager().get_page_metadata(source_page)
                )
                self._process_object["rich_content"]["type"] = "url"
            self._process_object["status"] = "done"

            self._process_object["responses"] = [
                {
                    "response_id": self._process_object["response_id"],
                    "created": self._process_object["created"],
                    "response": self._process_object["response"],
                    "rich_content": self._process_object["rich_content"],
                }
            ]

        except Exception as e:
            self._logger.error(
                f"Error in worker for session {self._session_id}, process {self._process_object['id']} : {repr(e)}"
            )
            self._process_object["status"] = "error"

    def run(self):
        self._logger.debug(f"[{self._process_object['id']}] Default Worker launched")

        session_object = SessionManager.load_session(self._session_id)

        # Traitement Génératif
        self._llm_treatment(session_object)

        if "response" not in self._process_object:
            self._process_object["response"] = AgentManager().get_default_response(
                self._agent_id
            )

        SessionManager.save_session(session_object)
        SessionManager.save_process(self._process_object)

        self._logger.debug(f"[{self._process_object['id']}] Default Worker finished")
