class EmbeddingFactory:
    @staticmethod
    def create_embedding(config: dict):
        provider = config["provider"]

        if provider == "OpenAI":
            from llama_index.embeddings.openai import OpenAIEmbedding

            return OpenAIEmbedding(
                model=config.get("model", "text-embedding-3-small"),
                api_key=config["openai_api_key"],
            )
        if provider == "HuggingFace":
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            return HuggingFaceEmbedding(
                model_name=config.get(
                    "model", "antoinelouis/biencoder-electra-base-french-mmarcoFR"
                ),
                cache_folder=config["cache_folder"],
            )

        raise ValueError(f"Embedding provider {provider} is not supported.")
