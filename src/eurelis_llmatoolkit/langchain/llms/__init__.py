from langchain.llms.base import BaseLLM

from eurelis_llmatoolkit.utils.base_factory import ProviderFactory
from eurelis_llmatoolkit.langchain.llms.openai import GenericOpenAIFactory


class GenericLLMFactory(ProviderFactory[BaseLLM]):
    ALLOWED_PROVIDERS = {
        **GenericOpenAIFactory.ALLOWED_PROVIDERS,
        "huggingface-pipeline": "eurelis_llmatoolkit.langchain.llms.huggingface.HuggingFaceFactory",
    }
