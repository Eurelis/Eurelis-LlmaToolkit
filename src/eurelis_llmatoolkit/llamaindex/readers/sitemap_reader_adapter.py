from llama_index.readers.web import SitemapReader


# TODO: Pour plus de clareté créer une classe abstraite ReaderAdapter
class SitemapReaderAdapter:
    def __init__(self, config):
        self.reader = SitemapReader(
            html_to_text=config.get("html_to_text", True),
            limit=config.get("limit", 10),
        )

    def load_data(self, sitemap_url):
        # TODO: Décompacter les paramètres pour factoriser la méthode au niveau de la classe abstraite
        return self.reader.load_data(sitemap_url=sitemap_url)

    # TODO: Généraliser la méthode pour tous les readers au niveau de la classe abstraite
    @staticmethod
    def get_load_data_params(dataset_config):
        return {
            "sitemap_url": dataset_config["reader"]["sitemap_url"],
        }
