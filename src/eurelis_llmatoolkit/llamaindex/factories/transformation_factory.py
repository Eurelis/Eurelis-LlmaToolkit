from llama_index.core.node_parser import SentenceSplitter


class TransformationFactory:
    @staticmethod
    def create_transformation(config: dict):
        provider = config["provider"]

        if provider == "SentenceSplitter":
            return SentenceSplitter(
                chunk_size=config.get("chunk_size", 768),
                chunk_overlap=config.get("chunk_overlap", 56),
            )
        raise ValueError(f"Transformation provider {provider} non supporté.")
