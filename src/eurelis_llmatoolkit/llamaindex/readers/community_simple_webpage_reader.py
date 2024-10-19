from llama_index.readers.web import SimpleWebPageReader

from eurelis_llmatoolkit.llamaindex.readers.abstract_reader_adapter import (
    AbstractReaderAdapter,
)


class CommunitySimpleWebPageReader(AbstractReaderAdapter):
    required_params = ["urls"]  # Liste des paramètres requis

    def __init__(self, config):
        super().__init__(config)
        self.reader = SimpleWebPageReader(
            html_to_text=config.get("html_to_text", True),
        )

    def load_data(self):
        """Charge les données en passant les paramètres nécessaires au reader."""
        load_params = self._get_load_data_params()
        return self.reader.load_data(**load_params)
