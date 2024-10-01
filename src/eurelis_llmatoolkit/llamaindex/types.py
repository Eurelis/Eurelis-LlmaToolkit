from typing import TYPE_CHECKING, Callable, Mapping, Sequence, TypeAlias, Union

# TODO : Faire valider types dans llamaindex

if TYPE_CHECKING:
    from llama_index.core import Document
    from llama_index.core.base.embeddings.base import BaseEmbedding

JSON: TypeAlias = Union[Mapping[str, "JSON"], list["JSON"], str, int, float, bool, None]
PARAMS: TypeAlias = Mapping[str, "JSON"]
FACTORY: TypeAlias = Union[str, PARAMS]
CLASS: TypeAlias = Union[str, PARAMS]

EMBEDDING: TypeAlias = Sequence[float]

DOCUMENT_MEAN_EMBEDDING: TypeAlias = Union[
    str, Callable[["BaseEmbedding", Sequence["Document"]], EMBEDDING]
]
