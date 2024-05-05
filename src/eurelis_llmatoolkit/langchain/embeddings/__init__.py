from typing import Mapping, Union

from langchain.schema.embeddings import Embeddings

from eurelis_llmatoolkit.utils.base_factory import ProviderFactory


class GenericEmbeddingsFactory(ProviderFactory[Embeddings]):
    """
    Generic embeddings factory, will delegate embeddings construction
    to another factory given a provider name
    """

    ALLOWED_PROVIDERS: Mapping[str, Union[type, str]] = {
        "openai": "eurelis_llmatoolkit.langchain.embeddings.openai.OpenAIEmbeddingsFactory",
        "huggingface": "eurelis_llmatoolkit.langchain.embeddings.huggingface.HuggingFaceEmbeddingsFactory",
    }
