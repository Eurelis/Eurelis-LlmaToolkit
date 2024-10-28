import importlib


class RetrieverFactory:
    @staticmethod
    def create_retriever(
        config: dict,
        index=None,
        filters=None,
        embedding_model=None,
    ):
        provider = config.get("provider")

        #
        # Check for built-in Retriever
        #
        if provider == "VectorIndexRetriever":
            from llama_index.core.retrievers import VectorIndexRetriever

            if index is None:
                raise ValueError("VectorIndexRetriever requires a valid 'index'.")

            return VectorIndexRetriever(
                index=index,
                similarity_top_k=config.get("similarity_top_k", 10),  # Default to 10
                filters=filters,
                embed_model=embedding_model,
            )

        #
        # If the provider is a custom retriever
        #
        if provider.count(".") == 0:
            raise ValueError(
                "Provider attribute must reference a standard Retriever short name or a fully qualified class path"
            )

        module_name, class_name = provider.rsplit(".", 1)

        module = importlib.import_module(module_name)

        retriever_class = getattr(module, class_name)

        return retriever_class(
            index=index,
            similarity_top_k=config.get("similarity_top_k", 10),
            filters=filters,
            embed_model=embedding_model,
            config=config,
        )
