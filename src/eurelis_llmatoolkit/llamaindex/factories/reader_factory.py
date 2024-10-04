from ..readers.advanced_sitemap_reader_adapter import AdvancedSitemapReader
from ..readers.sitemap_reader_adapter import SitemapReaderAdapter
from ..readers.simple_webpage_reader_adapter import SimpleWebPageReaderAdapter


class ReaderFactory:
    @staticmethod
    def create_reader(config: dict):
        provider = config["provider"]
        if provider == "SitemapReader":
            return SitemapReaderAdapter(config)
        if provider == "SimpleWebPageReader":
            return SimpleWebPageReaderAdapter(config)
        if provider == "AdvancedSitemapReader":
            return AdvancedSitemapReader(config)

        raise ValueError(f"Reader provider {provider} non supporté.")
