
from eurelis_llmatoolkit.utils.base_factory import ProviderFactory
from langchain_core.vectorstores import VectorStore


class GenericVectorStoreFactory(ProviderFactory[VectorStore]):
    """
    Generic factory for vector store, delegate to another factory under the hood
    """

    ALLOWED_PROVIDERS = {
        "chroma": "eurelis_llmatoolkit.langchain.vectorstores.chroma.ChromaFactory",
        "solr": "eurelis_llmatoolkit.langchain.vectorstores.solr.SolrFactory",
        "mongodb": "eurelis_llmatoolkit.langchain.vectorstores.mongodb.MongoDBVectorStoreFactory",
    }
