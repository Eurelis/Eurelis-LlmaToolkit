from typing import List

from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.retrievers import VectorIndexRetriever


class QAndaRetriever(VectorIndexRetriever):
    """QAnda retriever that processes nodes based on metadata.

    Inherits from VectorIndexRetriever.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _process_node_metadata(self, node_with_score: NodeWithScore) -> NodeWithScore:
        node = node_with_score.node
        metadata = node.metadata or {}

        if (
            metadata.get("generated_content") is True
            and "generated_content_mode" in metadata
        ):
            generated_content_mode = metadata["generated_content_mode"]
            if generated_content_mode == "QAndA":
                metadata["generated_question"] = node.text
                node.text = metadata.get("generated_content_origin", node.text)

                node.metadata = metadata
                node_with_score.node = node
        return node_with_score

    def _process_nodes(
        self, nodes_with_scores: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        return [self._process_node_metadata(node) for node in nodes_with_scores]

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        nodes_with_scores = super()._retrieve(query_bundle)
        return self._process_nodes(nodes_with_scores)

    async def _aretrieve(
        self,
        query_bundle: QueryBundle,
    ) -> List[NodeWithScore]:
        nodes_with_scores = await super()._aretrieve(query_bundle)
        return self._process_nodes(nodes_with_scores)
