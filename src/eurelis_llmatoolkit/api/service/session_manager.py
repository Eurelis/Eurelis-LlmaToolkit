import datetime
import json
import os
import pickle
import threading
import time
import uuid

from eurelis_llmatoolkit.api.model.dao import DAOFactory
from eurelis_llmatoolkit.api.service.agent_manager import AgentManager
from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager

logger = ConsoleManager().get_output()


class SessionManager:
    @staticmethod
    def init_session(agent_id, src_page) -> dict:
        current_time = time.time()
        session_object = {}
        session_object["id"] = uuid.uuid4().hex
        session_object["agent_id"] = agent_id
        session_object["status"] = "init"
        session_object["created"] = SessionManager.compute_timestamp(current_time)
        session_object["timestamp"] = str(round(current_time * 1000))
        session_object["src_page"] = src_page
        session_object["store"] = BaseConfig().MONGO_LLMATK_COLLECTION_NAME
        session_object["version"] = AgentManager().get_agent_version(agent_id)
        # session_object["compute_history"] = {}
        SessionManager.save_session(session_object)
        logger.debug(f"Session created with id: {session_object['id']}")
        return session_object

    @staticmethod
    def load_session(session_id: str) -> dict:
        return DAOFactory().get_session_dao().get(session_id)

    @staticmethod
    def save_session(session_object: dict):
        DAOFactory().get_session_dao().save(session_object)

    @staticmethod
    def load_process(session_id: str, process_id: str) -> dict:
        return DAOFactory().get_process_dao().get(process_id, session_id)

    @staticmethod
    def load_last_process(session_id: str) -> dict:
        return DAOFactory().get_process_dao().get_last(session_id)

    @staticmethod
    def list_processes(session_id: str) -> list:
        return DAOFactory().get_process_dao().list_all(session_id)

    @staticmethod
    def save_process(process_object: dict):
        DAOFactory().get_process_dao().update(process_object)

    @staticmethod
    def delete_process(process_object: dict):
        DAOFactory().get_process_dao().delete(process_object)

    @staticmethod
    def compute_timestamp(time=None) -> str:
        if time is not None:
            return datetime.datetime.fromtimestamp(time).isoformat()
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    @staticmethod
    def compute_timestamp_as_str() -> str:
        """Retourne le timestamp courant au format str

        Returns:
            str: Timestamp courant au format str
        """
        return str(round(time.time() * 1000))

    @staticmethod
    def compute_agent_responses(responses: list) -> list:
        """Transforme la liste des processus en liste de réponses pour l'affichage côté client

        Args:
            responses (list): Liste des processus

        Returns:
            list: Liste des réponses
        """
        agent_responses = []
        for response in responses:
            if response.get("response", None) is not None:
                agent_response = {}
                agent_response["response_id"] = response["response_id"]
                agent_response["created"] = response["created"]
                agent_response["response"] = response["response"]
                if response.get("rich_content", None) is not None:
                    agent_response["rich_content"] = response["rich_content"]
                agent_responses.append(agent_response)
        return agent_responses

    @staticmethod
    def compute_message_history(session_id: str) -> list:
        """Transforme la liste des processus en historique de messages pour l'affichage côté client

        Args:
            session_id (str): ID de la session

        Returns:
            list: Historique des messages
        """
        message_history = []
        process_list = SessionManager.list_processes(session_id)
        for process in process_list:
            message = {
                "process_id": process["id"],
                "user": process["query"],
                "agent": SessionManager.compute_agent_responses(process["responses"]),
            }
            message_history.append(message)
        return message_history

    @staticmethod
    def compute_compute_history(session_id: str) -> dict:
        compute_history = {}
        process_list = SessionManager.list_processes(session_id)
        for process in process_list:
            compute_history[process["id"]] = process
        return compute_history

    @staticmethod
    def compute_full_session_object(session_id: str) -> dict:
        session_object = SessionManager.load_session(session_id)

        session_object["status"] = SessionManager.compute_status(session_object["id"])
        session_object["message_history"] = SessionManager.compute_message_history(
            session_id
        )
        session_object["compute_history"] = SessionManager.compute_compute_history(
            session_id
        )
        return session_object

    @staticmethod
    def compute_status(session_id: str) -> str:
        """Retourne le statut de la session:
        - "init" : la sessions n'a pas de process enregistrés
        - "processing" : la session a au moins un process en cours
        - "active" : la session n'a plus de process en cours
        - "expired" : la session a expiré
        - "completed" : la problématique a été résolue
        - "aborted" : la session a été terminée par l'utilisateur
        - "terminated" : la session est fermée suite à une notation

        Args:
            session_id (str): ID de sessions

        Returns:
            str: Statut de la session : "init", "processing", "active", "expired", "completed", "aborted", "terminated"
        """
        session_object = SessionManager.load_session(session_id)
        if session_object["status"] in ("completed", "aborted", "terminated"):
            return session_object["status"]
        if SessionManager.has_expired(session_object):
            return "expired"
        process_list = SessionManager.list_processes(session_id)
        if len(process_list) == 0:
            return "init"
        else:
            for process in process_list:
                if process["status"] == "processing":
                    return "processing"
            return "active"

    @staticmethod
    def has_expired(session_object: dict) -> bool:
        """Retourne True si la session a expiré

        Args:
            session_id (str): ID de la session

        Returns:
            bool: True si la session a expiré
        """
        current_timestamp = round(time.time() * 1000)
        last_process = SessionManager.load_last_process(session_object["id"])

        timestamp_to_check = (
            int(last_process["id"])
            if last_process
            else int(session_object["timestamp"])
        )

        if (current_timestamp - timestamp_to_check) / 1000 > int(
            BaseConfig().API_SESSION_TIMEOUT
        ):
            return True
        return False

    @staticmethod
    def update_status(session_id: str, status: str):
        session_object = SessionManager.load_session(session_id)
        session_object["status"] = status
        SessionManager.save_session(session_object)
