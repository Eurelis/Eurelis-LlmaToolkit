from langchain.schema import BaseRetriever

from eurelis_llmatoolkit.utils.base_factory import ProviderFactory


class GenericRetrieverFactory(ProviderFactory[BaseRetriever]):
    ALLOWED_PROVIDERS = {
        "vectorstore": "eurelis_llmatoolkit.langchain.retrievers.vectorstore.VectorStoreRetrieverFactory",
        "selfcheck": "eurelis_llmatoolkit.langchain.retrievers.selfquery.SelfQueryRetrieverFactory",
    }

    def __init__(self):
        super().__init__()
        self.params["provider"] = "vectorstore"
