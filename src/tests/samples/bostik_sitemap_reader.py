from typing import List

import requests
import requests.compat
import json
from bs4 import BeautifulSoup

from eurelis_llmatoolkit.llamaindex.readers.advanced_sitemap_reader import (
    AdvancedSitemapReader,
)


class BostikSitemapReader(AdvancedSitemapReader):
    def _find_pdf_urls(self, page: BeautifulSoup, url: str) -> List[str]:
        """Récupère les URLs des PDFs inclus dans une page Bostik

        Args:
            page (BeautifulSoup): Page à traiter

        Returns:
            List[str]: Liste des URLs des PDFs ciblés
        """
        pdf_links = set()
        for link in page.find_all("a", href=True):
            # Vérifie si l'URL commence par /files/live/sites
            if link["href"].startswith("/files/live/sites"):
                pdf_links.add(requests.compat.urljoin(url, link["href"]))
        return list(pdf_links)

    def _get_metadata(self, url: str, page: BeautifulSoup) -> dict:
        """
        Génère un dictionnaire de métadonnées

        Args:
            url (str): L'URL de la page à traiter.
            page (BeautifulSoup): Le contenu de la page.

        Returns:
            dict: Dictionnaire contenant les métadonnées extraites.
        """
        metadata = super()._get_metadata(url=url, page=page)

        exclude_keys = {"event", "isLogged"}  # Clés à exclure du datalayer

        datalayer_div = page.find("div", class_="js-datalayer")

        if datalayer_div and datalayer_div.has_attr("data-layer"):
            # Charger le JSON du data-layer
            datalayer_json = datalayer_div["data-layer"]
            try:
                datalayer_dict = json.loads(datalayer_json)

                # Filtrer les clés et ajouter le préfixe 'c_'
                for key, value in datalayer_dict.items():
                    if key not in exclude_keys:
                        metadata[f"c_{key}"] = value

            except json.JSONDecodeError:
                print("Erreur : lors de la génération des métadonnées")

        return metadata
