import json
from pathlib import Path
from typing import List
from urllib.parse import urlparse

from llama_index.core import Document

from eurelis_llmatoolkit.llamaindex.readers.abstract_fs_reader import AbstractFSReader


class FSCacheMarshaller(AbstractFSReader):
    def __init__(self, cache_config):
        super().__init__(cache_config)
        self.base_dir = Path(cache_config["base_dir"]).resolve()

    def to_cache(self, dataset_name: str, documents: List[Document]):
        """
        Sérialise les documents dans des fichiers JSON en utilisant un chemin de cache
        basé sur l'URL source des documents.

        Args:
            dataset_name (str): Le nom du dataset.
            documents (list): La liste des documents à sérialiser.
        """
        if dataset_name is None:
            raise ValueError("dataset_name cannot be None")

        cache_base_path = self.base_dir / dataset_name

        for document in documents:
            # Définit le chemin de cache en fonction du doc_id
            cache_path = self._define_cache_path(document.doc_id)

            # Retirer un slash au début du cache_path si présent car provoque un l'écrasement des autres Path lors d'un concaténation
            if cache_path.parts and cache_path.parts[0].startswith("/"):
                cache_path = Path(*cache_path.parts[1:])

            # Combine base_dir, dataset_name et cache_path pour construire le chemin complet
            full_cache_path = (cache_base_path / cache_path).resolve()

            # Crée les répertoires si nécessaire
            full_cache_path.parent.mkdir(parents=True, exist_ok=True)

            # Sérialiser le document dans un fichier JSON
            document_data = document.to_embedchain_format()
            with open(full_cache_path.with_suffix(".json"), "w", encoding="utf-8") as f:
                json.dump(document_data, f, ensure_ascii=False, indent=4)

    def _process_file(self, path: Path) -> Document:
        with open(path, "r", encoding="utf-8") as f:
            content = json.load(f)
            return Document.from_embedchain_format(content)

    def load_data(self, dataset_name: str = None, *args, **kwargs) -> List[Document]:
        self._file_dir = f"{self._config['base_dir']}/{dataset_name}"
        return super().load_data(*args, **kwargs)

    @staticmethod
    def _define_cache_path(doc_id: str) -> Path:
        """
        Définit le chemin du cache en fonction de l'URL ou du chemin du document.

        Args:
            doc_id (str): L'URL ou le chemin du document.

        Returns:
            Path: Le chemin relatif où le document sera mis en cache.
        """
        # Vérifie si le doc_id est une URL ou un chemin de fichier local
        if "://" in doc_id:
            # Analyser l'URL pour obtenir le domaine et le chemin
            parsed_result = urlparse(doc_id)

            # Créer le chemin relatif à partir du netloc et du path de l'URL
            relative_path = (
                Path(parsed_result.path) if parsed_result.path else Path("index")
            )
            return Path(parsed_result.netloc) / Path(*relative_path.parts[1:])
        else:
            # Pour les chemins locaux, on utilise Path normalement
            return Path(doc_id)
