from llama_index.readers.web import SitemapReader


class SitemapReaderAdapter:
    def __init__(self, config):
        self.reader = SitemapReader(
            html_to_text=config.get("html_to_text", True),
            limit=config.get("limit", 10),
        )

    def load_data(self, sitemap_url):
        return self.reader.load_data(sitemap_url=sitemap_url)

    @staticmethod
    def get_load_data_params(dataset_config):
        return {
            "sitemap_url": dataset_config["reader"]["sitemap_url"],
        }
