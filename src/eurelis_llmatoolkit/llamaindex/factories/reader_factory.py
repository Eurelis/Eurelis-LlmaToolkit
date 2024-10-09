from eurelis_llmatoolkit.llamaindex.readers.advanced_sitemap_reader_adapter import (
    AdvancedSitemapReader,
)
from eurelis_llmatoolkit.llamaindex.readers.sitemap_reader_adapter import (
    SitemapReaderAdapter,
)
from eurelis_llmatoolkit.llamaindex.readers.simple_webpage_reader_adapter import (
    SimpleWebPageReaderAdapter,
)


class ReaderFactory:
    @staticmethod
    def create_reader(config: dict, namespace: str):
        provider = config["provider"]
        if provider == "SitemapReader":
            return SitemapReaderAdapter(config)
        if provider == "SimpleWebPageReader":
            return SimpleWebPageReaderAdapter(config)
        if provider == "AdvancedSitemapReader":
            return AdvancedSitemapReader(config, namespace)

        raise ValueError(f"Reader provider {provider} non supporté.")
