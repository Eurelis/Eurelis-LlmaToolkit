from langchain.document_loaders.base import BaseLoader

from eurelis_llmatoolkit.utils.base_factory import ProviderFactory


class GenericLoaderFactory(ProviderFactory[BaseLoader]):
    """
    Generic loader factory
    """

    ALLOWED_PROVIDERS = {
        "url": "eurelis_llmatoolkit.langchain.document_loaders.url.UrlLoaderFactory",
        "fs": "eurelis_llmatoolkit.langchain.document_loaders.fs.FSLoaderFactory",
        "list": "eurelis_llmatoolkit.langchain.document_loaders.list.ListLoaderFactory",
        "sitemap": "eurelis_llmatoolkit.langchain.document_loaders.sitemap.SitemapDocumentLoaderFactory",
    }
