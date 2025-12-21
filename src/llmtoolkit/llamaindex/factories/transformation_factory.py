from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from llama_index.core.callbacks import CallbackManager


class TransformationFactory:
    @staticmethod
    def create_transformation(
        config: dict, callback_manager: Optional["CallbackManager"] = None
    ):
        provider = config["provider"]

        if provider == "SentenceSplitter":
            from llama_index.core.node_parser import SentenceSplitter

            return SentenceSplitter(
                chunk_size=config.get("chunk_size", 768),
                chunk_overlap=config.get("chunk_overlap", 56),
                callback_manager=callback_manager,
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
