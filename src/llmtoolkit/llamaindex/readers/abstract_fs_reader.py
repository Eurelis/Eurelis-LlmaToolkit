from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from llama_index.core import Document
from llama_index.core.readers.base import BaseReader


class AbstractFSReader(BaseReader):
    """Classe abstraite pour les readers de fichiers sur le système de fichiers."""

    def __init__(self, config: dict, namespace: str = None):
        self._namespace = namespace
        self._config = config
        self._file_dir = config["base_dir"]
        self._unsuccessful_docs: list[str] = (
            []
        )  # Liste des docs non récupérés(ex:pages en timeout ou fichiers non trouvés)

    def _get_files(self, path: str, glob: str) -> Generator:
        """Récupère les fichiers à partir du path et du glob.

        Args:
            path (str): Le chemin vers le répertoire
            glob (str): Le motif de recherche des fichiers

        Returns:
            generator: Un générateur de fichiers.
        """

        file_path = Path(path)

        return file_path.rglob(glob)

    def _get_metadatas(self, path: Path, relative_path: str) -> dict:
        """Récupère les métadonnées d'un fichier.

        Args:
            path (Path): Le chemin du fichier.

        Returns:
            dict: Les métadonnées du fichier.
        """
        last_modified = path.stat().st_mtime
        last_modified_formatted = datetime.fromtimestamp(last_modified).isoformat()

        metadata = {
            "source": relative_path,
            "lastmod": last_modified_formatted,
        }

        if self._namespace:
            metadata["namespace"] = self._namespace

        return metadata

    @abstractmethod
    def _process_file(self, path: Path) -> Document:
        """Traite un fichier et retourne un objet Document.

        Args:
            file (Path): Le fichier à traiter.

        Returns:
            Document: Un objet Document.
        """

    def load_data(self, *args: Any, **kwargs: Any) -> list:
        """Charge les données à partir d'un path et d'un glob.

        Returns:
            list: Liste des documents.
        """
        glob = self._config.get("glob", "*.json")

        files = self._get_files(self._file_dir, glob)

        documents = []

        for file in files:
            document = self._process_file(file)
            documents.append(document)

        return documents

    def get_unsuccessful_docs(self) -> list[str]:
        """Retourne une liste vide par défaut pour les fichiers/Docs échoués."""
        return self._unsuccessful_docs
