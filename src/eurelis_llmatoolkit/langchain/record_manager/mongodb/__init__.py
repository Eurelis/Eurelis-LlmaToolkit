from eurelis_llmatoolkit.langchain.indexes.mongo_record_manager import MongoRecordManager

class MongoDBRecordManagerFactory:
    def __init__(self, mongodb_url, db_name, collection_name):
        self.mongodb_url = mongodb_url
        self.db_name = db_name
        self.collection_name = collection_name

    def build(self):
        if not self.mongodb_url or not self.db_name:
            raise ValueError("MongoDB URL and DB name must be provided")
        return self

    def create_record_manager(self, namespace):
        return MongoRecordManager(
            namespace=namespace,
            mongodb_url=self.mongodb_url,
            db_name=self.db_name,
            collection_name=self.collection_name
        )
