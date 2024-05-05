from typing import TYPE_CHECKING, Iterator

from langchain.document_loaders.base import BaseBlobParser
from langchain.schema import Document
from langchain_community.document_loaders import Blob

from eurelis_llmatoolkit.utils.base_factory import BaseFactory
from eurelis_llmatoolkit.langchain.dataset.dataset import Dataset

if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext


class DocumentCacheParser(BaseBlobParser):
    """
    Document cache parser
    """

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """
        Override of lazy_parse method
        Args:
            blob: file blob representation

        Yields:
            an iterator over documents
        """
        yield Dataset.load_document_from_cache(blob.path)


class DocumentCacheParserFactory(BaseFactory[BaseBlobParser]):
    """
    Factory for the document cache parser
    """

    def build(self, context: "BaseContext") -> BaseBlobParser:
        """
        Construct the document cache parser instance
        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a document cache parser instance
        """
        return DocumentCacheParser()
