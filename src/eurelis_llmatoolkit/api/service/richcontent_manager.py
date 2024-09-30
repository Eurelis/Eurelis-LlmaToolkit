import datetime
from urllib.parse import urljoin, urlparse

import requests
from eurelis_llmatoolkit.api.model.dao import DAOFactory
from eurelis_llmatoolkit.api.misc.singleton import Singleton
from bs4 import BeautifulSoup


class RichContentManager(metaclass=Singleton):
    """La classe RichContentManager permet de récupérer les métadonnées des pages web pour en extraire les informations utiles à l'affichage."""

    # 30 days
    CACHE_TTL = 3600 * 24 * 30

    def get_page_metadata(self, url: str) -> dict:
        """Retourne les métadonnées de la page web.

        Args:
            url (str): URL de la page web

        Returns:
            dict: Métadonnées de la page web
        """
        cache_data = DAOFactory().get_cache_dao().get(url)
        if cache_data is not None and "timestamp" in cache_data:
            # Convert isoformat to datetime
            timestamp = datetime.datetime.fromisoformat(cache_data["timestamp"])
            timedelta = datetime.timedelta(days=30)
            if (timestamp + timedelta) > datetime.datetime.now(datetime.timezone.utc):
                return cache_data

        new_cache_data = self._compute_cache(url)
        DAOFactory().get_cache_dao().save(url, new_cache_data, self.CACHE_TTL)
        return new_cache_data

    @staticmethod
    def _compute_cache(url: str) -> dict:
        try:
            response = requests.get(url)
            if response.status_code != 200:
                return None
            soup = BeautifulSoup(response.text, "html.parser")
        except:
            # if we can't fetch or parse the source we return an empty dict
            return None

        og_title = soup.find("meta", property="og:title")
        og_description = soup.find("meta", property="og:description")
        og_image = soup.find("meta", property="og:image")

        #
        # Title
        if og_title is not None:
            title = og_title["content"]
        else:
            title = soup.find("title").string

        #
        # Description
        if og_description is not None:
            description = og_description["content"]
        else:
            description = soup.find("meta", property="description")
            if description is not None:
                description = description["content"]

        # Image
        if og_image is not None:
            image = og_image["content"]
        else:
            figure = soup.find("figure")
            if figure is None:
                image = None
            else:
                new_src = figure.find("img")["src"]
                parsed_url = urlparse(url)
                base_url = parsed_url.scheme + "://" + parsed_url.netloc
                image = urljoin(base_url, new_src)

        # Timestamp
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        return {
            "target": url,
            "title": title,
            "description": description,
            "image": image,
            "timestamp": timestamp,
        }
