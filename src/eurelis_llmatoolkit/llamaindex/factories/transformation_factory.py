class TransformationFactory:
    @staticmethod
    def create_transformation(config: dict):
        provider = config["provider"]

        if provider == "SentenceSplitter":
            from llama_index.core.node_parser import SentenceSplitter

            return SentenceSplitter(
                chunk_size=config.get("chunk_size", 768),
                chunk_overlap=config.get("chunk_overlap", 56),
            )
        if provider == "JSONFileAcronymTransformer":
            from eurelis_llmatoolkit.llamaindex.transformers.json_file_acronym_transformer import (
                JSONFileAcronymTransformer,
            )

            return JSONFileAcronymTransformer(config)
        if provider == "MetadataTransformer":
            from eurelis_llmatoolkit.llamaindex.transformers.metadata_transformer import (
                MetadataTransformer,
            )

            return MetadataTransformer(config)
        if provider == "LLMNodeTransformer":
            from eurelis_llmatoolkit.llamaindex.transformers.llm_node_transformer import (
                LLMNodeTransformer,
            )

            return LLMNodeTransformer(config)
        raise ValueError(f"Transformation provider {provider} is not supported.")
