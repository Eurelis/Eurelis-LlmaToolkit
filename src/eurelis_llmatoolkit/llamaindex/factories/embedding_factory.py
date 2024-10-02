from llama_index.embeddings.openai import OpenAIEmbedding


class EmbeddingFactory:
    @staticmethod
    def create_embedding(config):
        provider = config["provider"]

        if provider == "OpenAI":
            return OpenAIEmbedding(
                model=config.get("model", "text-embedding-3-small"),
                api_key=config["openai_api_key"],
            )

        raise ValueError(f"Embedding provider {provider} non supporté.")
