class VectorStoreFactory:
    @staticmethod
    def create_vector_store(config: dict):
        provider = config["provider"]

        if provider == "CustomMongoDB":
            import pymongo
            from eurelis_llmatoolkit.llamaindex.vector_stores.custom_mongodb_atlas_vector_store import (
                CustomMongoDBAtlasVectorSearch,
            )

            client = pymongo.MongoClient(config["url"])

            # FIXME : ValueError: Must specify MONGODB_URI via env variable if not directly passing in client.
            return CustomMongoDBAtlasVectorSearch(
                client,
                db_name=config["db_name"],
                collection_name=config["collection_name"],
                vector_index_name=config["vector_index_name"],
            )

        if provider == "MongoDB":
            import pymongo
            from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

            client = pymongo.MongoClient(config["url"])

            # FIXME : ValueError: Must specify MONGODB_URI via env variable if not directly passing in client.
            return MongoDBAtlasVectorSearch(
                client,
                db_name=config["db_name"],
                collection_name=config["collection_name"],
                vector_index_name=config["vector_index_name"],
            )

        if provider == "Chroma":
            import chromadb
            from llama_index.vector_stores.chroma import ChromaVectorStore

            mode = config.get("mode", "ephemeral")

            if mode == "ephemeral":
                client = chromadb.Client()
            elif mode == "persistent":
                client = chromadb.PersistentClient(config["path"])

            # check if the collection exists
            if config["collection_name"] in [c.name for c in client.list_collections()]:
                collection = client.get_collection(config["collection_name"])
            else:
                collection = client.create_collection(config["collection_name"])

            return ChromaVectorStore(client=client, chroma_collection=collection)

        raise ValueError(f"Vector store provider {provider} is not supported.")
