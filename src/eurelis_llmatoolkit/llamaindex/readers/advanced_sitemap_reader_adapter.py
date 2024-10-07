import requests
import xml.etree.ElementTree as ET
from eurelis_llmatoolkit.llamaindex.readers.sitemap_reader_adapter import (
    SitemapReaderAdapter,
)


class AdvancedSitemapReader(SitemapReaderAdapter):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

    def load_data(self, sitemap_url: str) -> list:
        """Charge les données d'un sitemap

        Args:
            sitemap_url (str): URL du sitemap

        Returns:
            list: Liste des données du sitemap
        """
        sitemap_content = self._fetch_sitemap(sitemap_url)
        root = ET.fromstring(sitemap_content)

        # on vérifie si le sitemap est un sitemap index ou un sitemap
        if root.tag.endswith("sitemapindex"):
            return self._process_sitemap_index(root)
        if root.tag.endswith("urlset"):
            return super().load_data(sitemap_url)
        raise ValueError(f"Format de sitemap non supporté pour l'URL: {sitemap_url}")

    def _fetch_sitemap(self, sitemap_url):
        """Récupère le contenu du sitemap de base

        Args:
            sitemap_url (str): URL du sitemap

        Returns:
            bytes: Contenu du sitemap
        """

        response = requests.get(sitemap_url, timeout=10)
        if response.status_code == 200:
            return response.content
        raise requests.exceptions.RequestException(
            f"Impossible de récupérer le sitemap depuis {sitemap_url}"
        )

    def _process_sitemap_index(self, root: ET.Element) -> list:
        """Récupère les données de tous les sitemaps référencés dans un sitemap index

        Args:
            root (ET.Element): ElementTree root de l'index de sitemap

        Returns:
            list: Liste des données de tous les sitemaps référencés
        """
        all_data = []
        for sitemap in root.findall(".//{*}sitemap"):
            loc = sitemap.find("{*}loc").text
            data = self.load_data(loc)
            all_data.extend(data)
        return all_data
