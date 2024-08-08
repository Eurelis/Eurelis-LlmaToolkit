import os

import sentry_sdk
from dotenv import load_dotenv

from eurelis_llmatoolkit.api.misc.singleton import Singleton

class BaseConfig(metaclass=Singleton):
    DEV_ENV_NAME = "dev"
    INTEG_ENV_NAME = "int"
    PREPROD_ENV_NAME = "preprod"
    PROD_ENV_NAME = "prod"
    load_dotenv()

    def __getattr__(self, name):
        # print(f"__getattr__ {name}")
        return self._get_from_env(name)

    def get(self, property, default=None):
        return os.getenv(property, default=default)

    def is_set(self, property, default=False):
        return self.get(property, default=str(default)).lower() in ["true", "yes"]

    @staticmethod
    def _get_from_env(property):
        env_var = os.environ.get(property)
        if not env_var:
            raise RuntimeError(f"Undefined property {property}")
        return env_var

    @staticmethod
    def _get_app_base_path():
        env_var = os.environ.get("APP_BASE_PATH")
        if not env_var:
            return "."
        else:
            return env_var

    def _get_safe_value_from_key(self, key):
        try:
            # if self.get_current_environment() == BaseConfig.DEV_ENV_NAME:
            return os.environ[key]
            # else:
            # return BaseConfig._get_secret(key)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise RuntimeError(e)

    def get_current_environment(self):
        """Returns the current environment.

        Returns:
            {string} -- Current environement ID : dev | integ | preprod | prod
        """
        try:
            return os.environ["CURRENT_ENV"]
        except:
            return BaseConfig.DEV_ENV_NAME

    def is_dev_env(self) -> bool:
        return self.get_current_environment() == BaseConfig.DEV_ENV_NAME

    def is_prod_env(self) -> bool:
        return self.get_current_environment() == BaseConfig.PROD_ENV_NAME

config = BaseConfig()