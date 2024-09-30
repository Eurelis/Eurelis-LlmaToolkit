from typing import TYPE_CHECKING, Iterator

from eurelis_llmatoolkit.utils.base_factory import BaseFactory
from eurelis_llmatoolkit.langchain.dataset.dataset import Dataset
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document

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
