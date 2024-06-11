from typing import TYPE_CHECKING

from eurelis_llmatoolkit.utils.base_factory import ParamsDictFactory
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext


class HuggingFaceEmbeddingsFactory(ParamsDictFactory[Embeddings]):
    OPTIONAL_PARAMS = {"cache_folder", "encode_kwargs", "model_name", "multi_process"}

    def build(self, context: "BaseContext") -> Embeddings:
        """
        Construct the embeddings object

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            embeddings
        """
        from langchain_community.embeddings import HuggingFaceEmbeddings

        arguments = self.get_optional_params()

        return HuggingFaceEmbeddings(**arguments)
