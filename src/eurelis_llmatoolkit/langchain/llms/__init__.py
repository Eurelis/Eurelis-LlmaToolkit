
from eurelis_llmatoolkit.utils.base_factory import ProviderFactory
from eurelis_llmatoolkit.langchain.llms.openai import GenericOpenAIFactory
from langchain_core.language_models import BaseLLM


class GenericLLMFactory(ProviderFactory[BaseLLM]):
    ALLOWED_PROVIDERS = {
        **GenericOpenAIFactory.ALLOWED_PROVIDERS,
        "huggingface-pipeline": "eurelis_llmatoolkit.langchain.llms.huggingface.HuggingFaceFactory",
    }
