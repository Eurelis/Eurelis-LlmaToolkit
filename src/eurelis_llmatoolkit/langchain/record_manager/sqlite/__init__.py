import os.path
from pathlib import Path

from langchain.indexes import SQLRecordManager, index

class SQLiteRecordManagerFactory:
    def __init__(self, db_url):
        self.db_url = db_url

    def build(self):
        if not self.db_url:
            raise ValueError("SQLite DB URL must be provided")
        
        sqlite_prefix = "sqlite:///"
        if self.db_url.startswith(sqlite_prefix):
            sqlite_length = len(sqlite_prefix)
            path = self.db_url[sqlite_length:]
            path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
            file_folder = Path(os.path.dirname(path))
            os.makedirs(file_folder, exist_ok=True)

        return self

    def create_record_manager(self, namespace):
        return SQLRecordManager(
            namespace=namespace,
            db_url=self.db_url
        )
