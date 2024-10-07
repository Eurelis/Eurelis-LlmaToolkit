from eurelis_llmatoolkit.llamaindex.readers.reader_adapter import ReaderAdapter
from llama_index.readers.web import SimpleWebPageReader


class SimpleWebPageReaderAdapter(ReaderAdapter):
    def __init__(self, config):
        self.reader = SimpleWebPageReader(
            html_to_text=config.get("html_to_text", True),
        )

    def load_data(self, urls):
        return self.reader.load_data(urls=urls)

    # TODO: Généraliser la méthode pour tous les readers au niveau de la classe abstraite
    @staticmethod
    def get_load_data_params(dataset_config):
        return {
            "urls": dataset_config["reader"]["urls"],
        }
