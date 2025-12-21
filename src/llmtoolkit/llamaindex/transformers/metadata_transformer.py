from typing import List, Any, Sequence
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode


class MetadataTransformer(NodeParser):
    def __init__(self, config: dict):
        """
        Initialize class with metadata from configuration file.
        """
        super().__init__()
        self._metadata: dict[str, Any] = config.get("metadata", {})

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse nodes by adding metadata from the configuration file.

        Args:
            nodes (Sequence[BaseNode]): List of nodes to be processed.
            show_progress (bool, optional): Show progress. Defaults to False.
        Returns:
            List[BaseNode]: List of modified nodes.
        """
        updated_nodes = []

        for node in nodes:
            if hasattr(node, "metadata"):
                node.metadata.update(self._metadata)
            else:
                node.metadata = self._metadata.copy()

            updated_nodes.append(node)

        return updated_nodes
