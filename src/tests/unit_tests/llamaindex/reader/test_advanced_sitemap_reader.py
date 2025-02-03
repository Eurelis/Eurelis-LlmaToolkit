from eurelis_llmatoolkit.llamaindex.readers.advanced_sitemap_reader import (
    AdvancedSitemapReader,
)


def test_instantiation(advanced_sitemap_readers):
    for reader in advanced_sitemap_readers:
        assert isinstance(reader, AdvancedSitemapReader)


def test_load_data(advanced_sitemap_readers):
    for reader in advanced_sitemap_readers:
        data = reader.load_data()
        assert data is not None
        assert len(data) > 0


def test_load_data_sitemap_hs(advanced_sitemap_readers_hs):
    for reader_hs in advanced_sitemap_readers_hs:
        data = reader_hs.load_data()
        assert data is None
