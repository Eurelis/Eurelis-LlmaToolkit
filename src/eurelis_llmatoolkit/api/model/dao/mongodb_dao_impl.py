import random
from datetime import datetime, timedelta, timezone

from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
from eurelis_llmatoolkit.api.misc.singleton import Singleton
from pymongo import MongoClient


class MongoDAOFactoryImpl(metaclass=Singleton):
    def __init__(self):
        self._mongodb_client = MongoClient(BaseConfig().MONGO_CONNECTION_STRING)
        self._mongodb_db = self._mongodb_client[BaseConfig().MONGO_DB_NAME]

    def get_session_dao(self):
        return MongoDBSessionDAOImpl(self._mongodb_db)

    def get_process_dao(self):
        return MongoDBProcessDAOImpl(self._mongodb_db)

    def get_cache_dao(self):
        return MongoDBCacheDAOImpl(self._mongodb_db)


class MongoDBSessionDAOImpl:
    """MongoDB implementation to manage sessions collection"""

    def __init__(self, db):
        self.collection = db.sessions

    def get(self, session_id):
        return self.collection.find_one({"id": session_id})

    def save(self, session):
        query = {"id": session["id"]}
        update = {"$set": session}
        insert_result = self.collection.update_one(query, update, upsert=True)
        return insert_result.acknowledged

    def list(self, filter=None, limit=0):
        return list(self.collection.find(filter).limit(limit).sort("created", -1))


class MongoDBProcessDAOImpl:
    """MongoDB implementation to manage processes collection"""

    def __init__(self, db):
        self.collection = db.processes

    def get(self, process_id, session_id):
        return self.collection.find_one({"id": process_id, "session_id": session_id})

    def update(self, process):
        query = {"id": process["id"], "session_id": process["session_id"]}
        update = {"$set": process}
        insert_result = self.collection.update_one(query, update, upsert=True)
        return insert_result.acknowledged

    def delete(self, process):
        query = {"id": process["id"], "session_id": process["session_id"]}
        delete_result = self.collection.delete_one(query)
        return delete_result.acknowledged

    def list_all(self, session_id: str) -> list:
        """Retourne la liste des processus de la session classés par ordre croissant de timestamp

        Args:
            session_id (_type_): ID de la session

        Returns:
            _type_: Liste des processus de la session classés par ordre croissant de timestamp
        """
        return list(
            self.collection.find({"session_id": session_id}).sort("timestamp", -1)
        )

    def get_last(self, session_id: str) -> dict:
        """Retourne le dernier processus de la session

        Args:
            session_id (str): ID de la session

        Returns:
            dict: Dernier processus de la session
        """
        return self.collection.find_one({"session_id": session_id}, sort=[("id", -1)])


class MongoDBCacheDAOImpl:
    # Auto-cleaning probability
    CLEANING_PROBABILITY = 0.01

    def __init__(self, db):
        self.collection = db.cache

    def get(self, key):
        # Auto-cleaning
        self._random_auto_cleaning()

        result = self.collection.find_one(
            {"key": key, "expiration_time": {"$gt": datetime.utcnow().isoformat()}}
        )
        if result:
            return result["value"]
        return None

    def save(self, key, value, ttl_seconds=86400):
        # Auto-cleaning
        self._random_auto_cleaning()

        now = datetime.now(timezone.utc)
        updated_time = now.isoformat()
        expiration_time = (now + timedelta(seconds=ttl_seconds)).isoformat()
        query = {"key": key}
        update = {
            "$set": {
                "key": key,
                "updated_time": updated_time,
                "expiration_time": expiration_time,
                "value": value,
            }
        }
        insert_result = self.collection.update_one(query, update, upsert=True)
        return insert_result.acknowledged

    def _random_auto_cleaning(self):
        """Supprime les données expirées de la collection selon la probabilité définie dans CLEANING_PROBABILITY"""
        if random.random() < self.CLEANING_PROBABILITY:
            self.collection.delete_many(
                {"expiration_time": {"$lt": datetime.utcnow().isoformat()}}
            )
