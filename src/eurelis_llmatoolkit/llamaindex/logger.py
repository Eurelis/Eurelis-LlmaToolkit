import logging
import logging.config
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Logger:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)

            # Load sentry_dsn and config_file from environment variables
            sentry_dsn = os.getenv("SENTRY_DSN")
            config_file = os.getenv("LOGGING_CONFIG_FILE", "logging.ini")

            cls._instance._initialize(sentry_dsn, config_file)
        return cls._instance

    def _initialize(self, sentry_dsn=None, config_file="logging.ini"):
        if os.path.exists(config_file):
            # Load configuration from logging.ini
            logging.config.fileConfig(config_file)
            self.logger = logging.getLogger()

        # Sentry integration
        if sentry_dsn:
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

    def get_logger(self):
        return self.logger
