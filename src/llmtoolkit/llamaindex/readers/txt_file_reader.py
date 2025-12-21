import os
from pathlib import Path

from llama_index.core.schema import Document

from eurelis_llmatoolkit.llamaindex.readers.abstract_fs_reader import AbstractFSReader


class TXTFileReader(AbstractFSReader):
    def __init__(self, config: dict, namespace: str = None):
        super().__init__(config)
        self._namespace = namespace

    def _process_file(self, path: Path) -> Document:
        """Traite un fichier et retourne un objet Document.

        Args:
            file (Path): Le fichier à traiter.

        Returns:
            Document: Un objet Document.
        """

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

            relative_path = os.path.relpath(path, self._config["base_dir"])

            document = Document(
                text=content,
                metadata=self._get_metadatas(path, relative_path),
                doc_id=relative_path,
            )

        return document
