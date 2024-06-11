from typing import TYPE_CHECKING

from eurelis_llmatoolkit.utils.base_factory import ParamsDictFactory
from langchain_core.vectorstores import VectorStore

if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext


class SolrFactory(ParamsDictFactory[VectorStore]):
    """
    Factory to get a solr based vector store
    """

    OPTIONAL_PARAMS = {
        "page_content_field",
        "vector_field",
        "core_name",
        "url_base",
        "query_handler",
        "update_handler",
        "metadata_fields",
    }

    def build(self, context: "BaseContext") -> VectorStore:
        """
        Construct a solr based vector store

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a Solr vector store object
        """
        try:
            from eurelis_langchain_solr_vectorstore import Solr  # type: ignore

            context.console.verbose_print(f"Getting solr vector store")

            return Solr(context.embeddings, core_kwargs=self.get_optional_params())
        except ImportError:
            raise ImportError(
                "Please install eurelis_langchain_solr_vectorstore with the option solr, (pip install eurelis_llmatoolkit[solr]"
            )
