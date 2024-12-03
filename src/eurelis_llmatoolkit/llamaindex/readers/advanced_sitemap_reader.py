import re
import time
import xml.etree.ElementTree as ET
from typing import List, Optional

import requests
import requests.compat
from bs4 import BeautifulSoup
from llama_index.core.schema import Document


from eurelis_llmatoolkit.llamaindex.readers.abstract_reader_adapter import (
    AbstractReaderAdapter,
)


class AdvancedSitemapReader(AbstractReaderAdapter):
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

        requests_per_second = self.config.get("requests_per_second", -1)

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
                page_data.metadata.update(metadatas)
                all_data.append(page_data)

            if requests_per_second > 0:
                time.sleep(1 / requests_per_second)

        return all_data

    def _process_page(self, url: str) -> Optional[Document]:
        """Récupère les données d'une page en incluant les PDFs dans la page

        Args:
            url (str): URL de la page

        Returns:
            Document: Contenu de la page
        """
        try:
            response = self._fetch_url(url)

            page = BeautifulSoup(response, "html.parser")

            if self.config.get("parser_remove", None):
                self._remove_excluded_elements(page, self.config["parser_remove"])

            page_text = page.get_text()

            if self.config.get("embed_pdf", False):
                pdf_urls = self._find_pdf_urls(page, url)
                for pdf_url in pdf_urls:
                    page_text += f"\n{self._process_pdf(pdf_url)}"

            if self.config.get("html_to_text", True):
                import html2text

                page_text = html2text.html2text(page_text)

            return Document(
                text=page_text,
                metadata=self._get_metadata(url, page),
                doc_id=url,
            )

        except Exception as e:
            print(f"Erreur lors de la récupération de {url}: {e}")
            return None

    def _find_pdf_urls(self, page: BeautifulSoup, url: str) -> List[str]:
        """Récupère les URLs des PDFs inclus dans une page

        Args:
            page (BeautifulSoup): Page à traiter

        Returns:
            List[str]: Liste des URLs des PDFs
        """
        pdf_links = set()
        for link in page.find_all("a", href=True):
            if link["href"].endswith(".pdf"):
                pdf_links.add(requests.compat.urljoin(url, link["href"]))
        return list(pdf_links)

    def _process_pdf(self, pdf_url: str) -> str:
        """Récupère le contenu d'un PDF

        Args:
            pdf_url (str): URL du PDF

        Returns:
            str: Contenu du PDF
        """
        import pymupdf
        import pymupdf4llm

        try:
            pdf_response = self._fetch_url(pdf_url)
            pdf_file = pymupdf.open(stream=pdf_response)

            title = (
                pdf_file.metadata.get("title", "PDF Document")
                if pdf_file.metadata
                else None
            )  # Titre du PDF ou titre par défaut
            if isinstance(title, bytes):
                title = title.decode("utf-8")

            # Vérifie si le titre est vide ou ne contient que des espaces
            if title is None or not title.strip():
                title = "PDF Document"  # Valeur par défaut

            # Extraction au format MD
            pdf_md_text = pymupdf4llm.to_markdown(pdf_file)

            return f"---------- {title} ----------\n{pdf_md_text}"

        except Exception as e:
            print(f"Erreur lors de la récupération de {pdf_url}: {e}")
            return ""

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

    def _get_metadata(self, url: str, page: BeautifulSoup) -> dict:
        """
        Génère un dictionnaire de métadonnées

        Args:
            url (str): L'URL de la page à traiter.
            page (BeautifulSoup): Le contenu de la page.

        Returns:
            dict: Dictionnaire contenant les métadonnées extraites.
        """
        h1 = page.find("h1")
        title = h1.get_text(strip=True) if h1 else None

        return {"title": title}

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
