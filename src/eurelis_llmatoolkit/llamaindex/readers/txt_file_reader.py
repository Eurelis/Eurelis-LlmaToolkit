from datetime import datetime
import os
from pathlib import Path
from typing import Generator
from eurelis_llmatoolkit.llamaindex.readers.reader_adapter import ReaderAdapter
from llama_index.core.schema import Document


class TXTFileReader(ReaderAdapter):
    required_params = ["path", "glob"]

    def __init__(self, config: dict, namespace: str = None):
        super().__init__(config)
        self.namespace = namespace

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

    def load_data(self, *args, **kwargs):
        """Charge les données à partir d'un path et d'un glob."""

        params = self._get_load_data_params()

        files = self._get_files(**params)

        documents = []

        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()

                # chemin relatif par rapport au répertoire de base
                relative_path = os.path.relpath(file, params["path"])

                # dernière date de modification du fichier
                last_modified = file.stat().st_mtime
                last_modified_formatted = datetime.fromtimestamp(
                    last_modified
                ).isoformat()

                metadata = {
                    "namespace": self.namespace,
                    "source": relative_path,
                    "lastmod": last_modified_formatted,
                }

                document = Document(
                    text=content, metadata=metadata, doc_id=relative_path
                )

                documents.append(document)

        return documents
