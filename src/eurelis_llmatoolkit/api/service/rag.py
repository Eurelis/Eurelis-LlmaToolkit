import datetime
import os
from typing import TYPE_CHECKING, cast

import requests
from bs4 import BeautifulSoup
from eurelis_llmatoolkit.api.misc.base_config import BaseConfig
from eurelis_llmatoolkit.api.misc.console_manager import ConsoleManager
from eurelis_llmatoolkit.utils.output import Verbosity
from eurelis_llmatoolkit.langchain import LangchainWrapper, LangchainWrapperFactory
from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext
from eurelis_llmatoolkit.api.misc.base_config import config

from flashrank import Ranker, RerankRequest

logger = ConsoleManager().get_output()

def get_wrapper(llmatk_config: str) -> LangchainWrapper:
    base_config = BaseConfig()
    factory = LangchainWrapperFactory()
    factory.set_verbose(Verbosity.CONSOLE_DEBUG)
    factory.set_config_path(f"config/{llmatk_config}")
    factory.set_logger_config(base_config.get("LANGCHAIN_LOGGER_CONFIG", None), base_config.get("LANGCHAIN_LOGGER_NAME", None))
    
    instance = factory.build(
        cast(BaseContext, None)
    )  # casting None to BaseContext is wanted
    setattr(factory, "instance", instance)
    wrapper=instance
    
    # wrapper = factory.build(None)

    return wrapper


def format_documents(documents, prefix_url_img, max_results):
    result_dict = []
    urls = set()
    for item in documents:
        if len(result_dict) >= max_results:
            break

        doc = item[0]
        score = item[1]
        source = doc["metadata"].get("source")

        if source not in urls and "Erreur 404" not in doc["metadata"].get("title", ""):
            urls.add(source)

            img = None
            if config.get("IMAGE_SRC_SCRAPPING", False) == "True":
                img = get_image(source, prefix_url_img)

            result_dict.append(
                {
                    "id": datetime.datetime.now().timestamp(),
                    "type": "autres",
                    "url": source,
                    "tags": "Site Internet",
                    "distance": abs(score),
                    "title": doc["metadata"].get("title"),
                    "baseline": tronquer_texte(doc["metadata"].get("description")),
                    "image": {
                        "title": img["title"] if img else "",
                        "alt": img["alt"] if img else "",
                        "src": img["src"] if img else "",
                    },
                }
            )

    # Order results by distance
    result_dict = sorted(result_dict, key=lambda k: k["distance"])

    return result_dict


def tronquer_texte(texte: str, longueur_max: int = 100) -> str:
    if texte is None:
        return ""

    if len(texte) <= longueur_max:
        return texte

    texte_tronque = texte[: longueur_max - 3]
    return f"{texte_tronque}..."


def get_image(url, starts_with):
    """
    Fonction pour récupérer le src d'une balise img avec timeout

    Args:
      url (str): L'URL de la page web
      starts_with (list): Un tableau de préfixes que le src de la balise img doit avoir

    Returns:
      dict: Un dictionnaire contenant les informations de l'image (src, alt, title), ou None si aucune balise img n'est trouvée, ne correspond pas aux préfixes ou si le timeout est atteint
    """
    try:
        with requests.get(
            url, timeout=2
        ) as response:  # Définir le timeout à 2 secondes

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")

                # Recherche de toutes les balises img
                images = soup.find_all("img")

                for img in images:
                    # Vérification si le src commence par l'un des préfixes spécifiés
                    for prefix in starts_with:
                        if img["src"].startswith(prefix):
                            img_data = {
                                "src": img.get("src", ""),
                                "alt": img.get("alt", ""),
                                "title": img.get("title", ""),
                            }

                            return img_data

    except requests.exceptions.Timeout:
        logger.warning(f"**Timeout atteint pour l'URL : {url}")

    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'image : {e}")

    # Retourne None si aucune balise img n'est trouvée, ne correspond pas aux préfixes ou si le timeout est atteint
    return None


def rerank(query, documents, score_threshold=0.0):
    """
    Fonction pour réordonner les documents en fonction de la pertinence de la requête

    Args:
        query (str): La requête de l'utilisateur
        documents (list): Liste des documents à réordonner
        score_threshold (float): Seuil de score pour filtrer les documents - Défault: 0.0

    Returns:
        list: Liste des documents réordonnés
    """
    # Init Ranker
    base_cache_dir = os.getenv("CACHE_BASE_PATH", "./data/cache")
    cache_dir = os.path.join(base_cache_dir, "flashrank")
    model_name = "ms-marco-MiniLM-L-12-v2"
    ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    # Prepare RerankRequest
    to_rank = []
    for result in documents:
        item = {}
        item["id"] = result["metadata"].get("_uid", "")
        item["meta"] = {"object": result}
        item["text"] = result["content"]
        to_rank.append(item)

    # Run rerank
    rerankrequest = RerankRequest(query=query, passages=to_rank)
    rerank_data = ranker.rerank(rerankrequest)

    # Return reranked documents with distance
    return [
        (item["meta"]["object"], 1.0 - item["score"])
        for item in rerank_data
        if item["score"] >= score_threshold
    ]
