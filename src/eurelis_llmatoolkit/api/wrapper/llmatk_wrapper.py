import json
import threading
import numpy as np
from bson import ObjectId

from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
from eurelis_llmatoolkit.api.misc.singleton import Singleton
from eurelis_llmatoolkit.api.model.dao import DAOFactory
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.langchain import LangchainWrapperFactory
from eurelis_llmatoolkit.langchain.addons.output.markdown_html_callback import (
    MarkdownHtmlCallback,
)
from eurelis_llmatoolkit.utils.output import Verbosity

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.schema.messages import BaseMessage


class DocumentEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseMessage):
            return obj.dict()
        if isinstance(obj, Document):
            return obj.dict()
        return json.JSONEncoder.default(self, obj)


class LlmaTkMemoryEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, BaseMessage):
            return obj.dict()
        if isinstance(obj, Document):
            return obj.dict()
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


class LlmaTkWrapper(metaclass=Singleton):
    # 30 minutes
    SESSION_TTL = 3600 * 30

    _markdown_html_callback = MarkdownHtmlCallback(
        input_field="answer", output_field="answer"
    )

    def __init__(self) -> None:
        self._wrapper_store_lock = threading.Lock()
        self._wrapper_store = {}

    def _get_wrapper(self, agent_id):
        base_config = BaseConfig()

        # Réponse rapide
        wrapper = self._wrapper_store.get(agent_id, None)
        if wrapper:
            return wrapper

        # Réponse lente (worst case)
        self._wrapper_store_lock.acquire()
        wrapper = self._wrapper_store.get(agent_id, None)
        if wrapper is None:
            factory = LangchainWrapperFactory()
            factory.set_verbose(Verbosity.CONSOLE_INFO)
            factory.set_logger_config(
                base_config.get("LANGCHAIN_LOGGER_CONFIG", None),
                base_config.get("LANGCHAIN_LOGGER_NAME", None),
            )

            llmatk_config = AgentManager().get_llmatoolkit_config(agent_id)
            if llmatk_config is None:
                raise RuntimeError("No configuration file found for this agent", 500)

            factory.set_base_dir("config/")
            factory.set_config_path(f"{llmatk_config}")

            # self._factory.set_base_dir("config/kb-index")
            # self._factory.set_config_path(".")

            # None fonctionne, il n'y a pas de vérification du typage en execution
            wrapper = factory.build(None)
            self._wrapper_store[agent_id] = wrapper

        self._wrapper_store_lock.release()
        return wrapper

    def _save_llmatk_memory(self, agent_id, session_id, llmatk_memory):
        DAOFactory().get_cache_dao().save(
            f"{agent_id}-{session_id}",
            json.loads(LlmaTkMemoryEncoder().encode(llmatk_memory)),
            self.SESSION_TTL,
        )

    def _load_llmatk_memory(self, agent_id, session_id):
        return DAOFactory().get_cache_dao().get(f"{agent_id}-{session_id}")

    def generate_response(
        self, agent_id, session_id: str, query: str, format: str = "html"
    ):
        wrapper = self._get_wrapper(agent_id)
        chain = None

        #
        # Load
        #
        llmatk_memory = None
        chain_memory = None

        if session_id:
            llmatk_memory = self._load_llmatk_memory(agent_id, session_id)
        if llmatk_memory:
            temp_chain_memory = llmatk_memory["chain_memory"]
            temp_chain_memory["chat_memory"] = ChatMessageHistory(
                **temp_chain_memory.get("chat_memory", None)
            )
            chain_memory = ConversationBufferMemory(**temp_chain_memory)
        chain = wrapper.get_chain(memory=chain_memory)

        #
        # Process
        #
        callbacks = []
        if format == "html":
            callbacks.append(self._markdown_html_callback)
        answer = chain(query, callbacks=callbacks)
        answer.pop("chat_history")

        #
        # Save
        #
        llmatk_memory = {
            "answer": answer,
            "chain_memory": chain.memory.dict(),
            "source_documents": answer["source_documents"],
        }
        if session_id:
            self._save_llmatk_memory(agent_id, session_id, llmatk_memory)

        #
        # Return
        #
        return llmatk_memory

    @staticmethod
    def get_recommended_page(source_documents: list) -> str:
        """Retourne la page source de la réponse LLM depuis la liste des sources des metadatas des documents source

        Args:
            source_documents (list): Liste des documents source

        Returns:
            dict: Page source
        """
        if len(source_documents) == 0:
            return None

        source_pages = [
            source_document.metadata["source"] for source_document in source_documents
        ]

        return max(source_pages, key=source_pages.count)
