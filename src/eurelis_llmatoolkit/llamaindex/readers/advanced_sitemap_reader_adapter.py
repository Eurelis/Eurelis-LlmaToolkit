import io
import logging
import re
import xml.etree.ElementTree as ET
from typing import Optional

import requests
import requests.compat
from bs4 import BeautifulSoup
from llama_index.core.schema import Document
from pypdf import PdfReader

from eurelis_llmatoolkit.llamaindex.readers.reader_adapter import ReaderAdapter


class AdvancedSitemapReader(ReaderAdapter):
    required_params = ["sitemap_url"]  # Liste des paramètres requis

    def __init__(self, config: dict, namespace: str = "default"):
        super().__init__(config)
        self._headers = {"User-Agent": config.get("user_agent", "EurelisLLMATK/0.1")}
        self._namespace = namespace

    def load_data(self, url: Optional[str] = None) -> list:
        """Charge les données d'un sitemap

        Args:

        Returns:
            list: Liste des données du sitemap
        """
        if url is None:
            load_params = self._get_load_data_params()
            url = load_params["sitemap_url"]

        sitemap_content = self._fetch_url(url)
        root = ET.fromstring(sitemap_content)

        # on vérifie si le sitemap est un sitemap index ou un sitemap
        if root.tag.endswith("sitemapindex"):
            return self._process_sitemap_index(root)
        if root.tag.endswith("urlset"):
            return self._process_urlset(root)
        raise ValueError(f"Format de sitemap non supporté pour l'URL: {url}")

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

    def _process_urlset(self, root: ET.Element) -> list:
        """Récupère les données d'un sitemap

        Args:
            root (ET.Element): ElementTree root du sitemap

        Returns:
            list: Liste des données du sitemap
        """
        all_data = []

        url_include_filters = self.config.get("url_include_filters", None)

        for url in root.findall(".//{*}url"):
            loc = url.find("{*}loc").text
            if url_include_filters and not any(
                re.match(regexp_pattern, loc) for regexp_pattern in url_include_filters
            ):
                continue
            page_data = self._process_page(loc)
            if page_data:
                metadatas = {
                    "source": loc,
                    "namespace": self._namespace,
                    "lastmod": url.find("{*}lastmod").text,
                }
                doc = Document(text=page_data, metadata=metadatas, doc_id=loc)
                all_data.append(doc)

        return all_data

    def _process_page(self, url: str) -> str:
        """Récupère les données d'une page en incluant les PDFs dans la page

        Args:
            url (str): URL de la page

        Returns:
            str: Contenu de la page
        """
        try:
            response = self._fetch_url(url)

            page = BeautifulSoup(response, "html.parser")

            if self.config.get("parser_remove", None):
                self._remove_excluded_elements(page, self.config["parser_remove"])

            page_text = page.get_text()

            if self.config.get("include_pdfs", False):
                page_text += self._process_pdfs(page, url)

            if self.config.get("html_to_text", True):
                import html2text

                page_text = html2text.html2text(page_text)

            return page_text

        except Exception as e:
            print(f"Erreur lors de la récupération de {url}: {e}")
            return None

    def _remove_excluded_elements(self, page: BeautifulSoup, remove_list: list):
        """Supprime les éléments dans parser_remove du contenu HTML

        Args:
            page (BeautifulSoup): Page à traiter
            remove_list (list): Liste des éléments à supprimer
        """
        for remove in remove_list:
            nodes = []
            if isinstance(remove, str):
                nodes = page.find_all(remove)
            elif isinstance(remove, dict):
                nodes = page.find_all(**remove)

            for node in nodes:
                node.extract()

    def _process_pdfs(self, page: BeautifulSoup, url: str) -> str:
        """Récupère le texte des PDFs inclus dans une page

        Args:
            page (BeautifulSoup): Page à traiter
            url (str): URL de la page
            page_text (str): Texte de la page

        Returns:
            str: Texte de la page avec les PDFs inclus
        """
        pdf_links = set()
        page_text = ""
        for link in page.find_all("a", href=True):
            if link["href"].endswith(".pdf"):
                pdf_links.add(requests.compat.urljoin(url, link["href"]))

        for pdf_link in pdf_links:
            try:
                pdf_response = self._fetch_url(pdf_link)

                temp_obj = io.BytesIO(pdf_response)
                pdf_file = PdfReader(temp_obj)

                for pdf_page in pdf_file.pages:
                    page_text += f"{pdf_page.extract_text()}\n\n"

            except Exception as e:
                print(f"Erreur lors de la récupération de {pdf_link}: {e}")

        return page_text

    def _fetch_url(self, url: str) -> Optional[str]:
        """Récupère le contenu d'une URL

        Args:
            url (str): URL de la page

        Returns:
            str: Contenu de la page
        """
        response = requests.get(url, timeout=10, headers=self._headers)
        if response.status_code == 200:
            return response.content
        print(f"Erreur lors de la récupération de {url}: {response.status_code}")
        return None
