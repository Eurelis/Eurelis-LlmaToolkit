import pymongo
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch


class VectorStoreFactory:
    @staticmethod
    def create_vector_store(config):
        provider = config["provider"]

        if provider == "MongoDB":
            client = pymongo.MongoClient(config["url"])

            # TODO : Bug
            # TODO : ValueError: Must specify MONGODB_URI via env variable if not directly passing in client.
            return MongoDBAtlasVectorSearch(
                client=client,
                db_name=config["db_name"],
                collection_name=config["collection_name"],
            )

        raise ValueError(f"Vector store provider {provider} non supporté.")
