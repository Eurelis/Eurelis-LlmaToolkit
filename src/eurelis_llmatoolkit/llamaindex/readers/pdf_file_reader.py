import os
from pathlib import Path

from llama_index.core.schema import Document
from pypdf import PdfReader

from eurelis_llmatoolkit.llamaindex.readers.abstract_fs_reader import AbstractFSReader


class PDFFileReader(AbstractFSReader):
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

        with open(path, "rb") as f:
            pdf = PdfReader(f)

            content = ""

            for page in pdf.pages:
                content += f"{page.extract_text()}\n\n"

            relative_path = os.path.relpath(path, self._config["base_dir"])

            document = Document(
                text=content,
                metadata=self._get_metadatas(path, relative_path),
                doc_id=relative_path,
            )

        return document
