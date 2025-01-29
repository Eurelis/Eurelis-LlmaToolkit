import logging
import logging.config
import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)


class Logger:
    _instance = None

    def __new__(cls, logging_config=None, enable_sentry=True):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(logging_config, enable_sentry)
        return cls._instance

    def _initialize(self, logging_config=None, enable_sentry=True):
        sentry_dsn = os.getenv("SENTRY_DSN")
        config_file = logging_config or os.getenv("LOGGING_CONFIG_FILE", "logging.ini")

        # Load logger configuration from logging.ini
        if os.path.exists(config_file):
            logging.config.fileConfig(config_file)
        self.logger = logging.getLogger("llmatoolkit")

        # Sentry integration
        if enable_sentry and sentry_dsn:
            self._setup_sentry(sentry_dsn)

    def _setup_sentry(self, sentry_dsn):
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_logging = LoggingIntegration(
            level=logging.INFO,  # Capture INFO and higher-level data as breadcrumbs
            event_level=logging.ERROR,  # Send errors as events
        )
        sentry_sdk.init(dsn=sentry_dsn, integrations=[sentry_logging])
        self.logger.info("Sentry initialized.")

    def get_logger(self, name):
        return logging.getLogger(name)
