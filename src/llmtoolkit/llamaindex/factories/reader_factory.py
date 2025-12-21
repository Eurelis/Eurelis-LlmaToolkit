import importlib
import inspect


class ReaderFactory:
    @staticmethod
    def create_reader(namespace: str, config: dict):

        provider = config["provider"]

        #
        # Check for built-in readers
        #
        if provider == "CommunitySitemapReader":
            from ..readers.community_sitemap_reader import CommunitySitemapReader

            return CommunitySitemapReader(config)

        if provider == "CommunitySimpleWebPageReader":
            from ..readers.community_simple_webpage_reader import (
                CommunitySimpleWebPageReader,
            )

            return CommunitySimpleWebPageReader(config)

        if provider == "AdvancedSitemapReader":
            from ..readers.advanced_sitemap_reader import AdvancedSitemapReader

            return AdvancedSitemapReader(config, namespace)

        if provider == "TXTFileReader":
            from ..readers.txt_file_reader import TXTFileReader

            return TXTFileReader(config, namespace)

        if provider == "PDFFileReader":
            from ..readers.pdf_file_reader import PDFFileReader

            return PDFFileReader(config, namespace)

        #
        # If the provider is a custom reader
        #
        if provider.count(".") == 0:
            raise ValueError(
                "Provider attribute must reference a standard Reader short name or a fully qualified class path"
            )

        module_name, class_name = provider.rsplit(".", 1)

        module = importlib.import_module(module_name)

        reader_class = getattr(module, class_name)

        init_params = inspect.signature(reader_class.__init__)

        if "namespace" in init_params.parameters:
            return reader_class(config, namespace)
        return reader_class(config)
