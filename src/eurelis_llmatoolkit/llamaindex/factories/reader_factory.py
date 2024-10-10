from eurelis_llmatoolkit.llamaindex.readers.advanced_sitemap_reader_adapter import (
    AdvancedSitemapReader,
)
from eurelis_llmatoolkit.llamaindex.readers.pdf_file_reader import PDFFileReader
from eurelis_llmatoolkit.llamaindex.readers.sitemap_reader_adapter import (
    SitemapReaderAdapter,
)
from eurelis_llmatoolkit.llamaindex.readers.simple_webpage_reader_adapter import (
    SimpleWebPageReaderAdapter,
)
from eurelis_llmatoolkit.llamaindex.readers.txt_file_reader import TXTFileReader


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
        if provider == "TXTFileReader":
            return TXTFileReader(config, namespace)
        if provider == "PDFFileReader":
            return PDFFileReader(config, namespace)

        raise ValueError(f"Reader provider {provider} non supporté.")
