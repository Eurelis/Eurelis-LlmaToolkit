import sys
from io import StringIO
from sentry_sdk import init, capture_message

import os


class Sentry:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            sentry_dsn = os.getenv("SENTRY_DSN")
            cls._instance._initialize(dsn=sentry_dsn)
        return cls._instance

    def _initialize(self, dsn):
        if dsn:
            init(dsn=dsn)
            self.redirect_stdout_to_sentry()

    def redirect_stdout_to_sentry(self):
        sys.stdout = SentryCapture()


class SentryCapture(StringIO):
    def write(self, message):
        if message.strip():
            capture_message(message)
        super().write(message)
