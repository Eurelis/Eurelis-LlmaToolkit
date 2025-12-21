import json
import re
from typing import List, Any, Sequence
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode


class JSONFileAcronymTransformer(NodeParser):
    def __init__(self, config: dict):
        """
        Initialisation de la classe avec le chemin vers le fichier JSON contenant les acronymes.

        Args:
            path (str): Le chemin vers le fichier JSON contenant les acronymes.
        """
        super().__init__()
        self._acronyms = self._load_acronyms(config["path"])

    def _load_acronyms(self, path: str) -> dict:
        """Charge le fichier JSON contenant les acronymes."""
        with open(path, "r") as f:
            return json.load(f).get("default", {})

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """Parse les noeuds en remplaçant les acronymes par leur signification.

        Args:
            nodes (Sequence[BaseNode]): Liste des noeuds à traiter.
            show_progress (bool, optional): Affiche la progression. Par défaut à False.

        Returns:
            List[BaseNode]: Liste des noeuds modifiés.
        """
        updated_nodes = []
        for node in nodes:
            updated_content = self._replace_acronyms(node.get_content())
            node.set_content(value=updated_content)
            updated_nodes.append(node)

        return updated_nodes

    def _replace_acronyms(self, text: str) -> str:
        """Remplace les acronymes dans le texte par leur signification.

        Args:
            text (str): Le texte à modifier.

        Returns:
            str: Le texte avec les acronymes remplacés.
        """
        for key, value in self._acronyms.items():
            text = re.sub(rf"\b{key}\b", f"{key} ({value})", text)
        return text
