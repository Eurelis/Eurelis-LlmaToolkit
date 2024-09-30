from eurelis_llmatoolkit.api.model.dao.mongodb_dao_impl import MongoDAOFactoryImpl
from eurelis_llmatoolkit.api.misc.singleton import Singleton


class DAOFactory(metaclass=Singleton):
    def __init__(self):
        self._dao_factory_impl = None
        self._dao_factory_impl = self._get_factory_impl()

    def _get_factory_impl(self):
        if self._dao_factory_impl is None:
            return MongoDAOFactoryImpl()
        else:
            return self._dao_factory_impl

    def get_session_dao(self):
        return self._dao_factory_impl.get_session_dao()

    def get_process_dao(self):
        return self._dao_factory_impl.get_process_dao()

    def get_cache_dao(self):
        return self._dao_factory_impl.get_cache_dao()
