from typing import TYPE_CHECKING

from eurelis_llmatoolkit.utils.base_factory import BaseFactory
from langchain_core.documents import BaseDocumentTransformer

if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext


class Html2TextTransformerFactory(BaseFactory[BaseDocumentTransformer]):
    """
    Factory for the Html2TextTransformer
    """

    def build(self, context: "BaseContext") -> BaseDocumentTransformer:
        """Construct the Html2textTransformer

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a Html2TextTransformer instance
        """
        from langchain_community.document_transformers import Html2TextTransformer

        return Html2TextTransformer()
