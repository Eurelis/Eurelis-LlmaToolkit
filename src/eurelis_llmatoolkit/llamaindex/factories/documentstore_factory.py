class DocumentStoreFactory:
    @staticmethod
    def create_document_store(config: dict):
        provider = config["provider"]

        if provider == "MongoDB":
            from llama_index.storage.docstore.mongodb import MongoDocumentStore

            return MongoDocumentStore.from_uri(
                uri=config["url"],
                db_name=config["db_name"],
            )

        raise ValueError(f"Document store provider {provider} is not supported.")
