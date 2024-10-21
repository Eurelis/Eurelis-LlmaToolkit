from llama_index.core.retrievers import VectorIndexRetriever


class RetrieverFactory:
    @staticmethod
    def create_retriever(
        config: dict,
        index=None,
        filter=None,
        embedding_model=None,
    ):
        provider = config.get("provider")

        #
        # Check for built-in Retriever
        #
        if provider == "VectorIndexRetriever":
            if index is None:
                raise ValueError("VectorIndexRetriever requires a valid 'index'.")

            return VectorIndexRetriever(
                index=index,
                similarity_top_k=config.get("similarity_top_k", 10),  # Default to 10
                filters=filter,
                embed_model=embedding_model,
            )

        raise ValueError(f"Retriever provider '{provider}' is not supported.")
