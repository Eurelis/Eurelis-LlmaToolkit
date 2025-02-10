import pytest
from eurelis_llmatoolkit.llamaindex.readers.advanced_sitemap_reader import (
    AdvancedSitemapReader,
)


@pytest.mark.parametrize(
    "advanced_sitemap_readers",
    ["reader/test_advanced_sitemap_reader.json"],
    indirect=True,
)
def test_instantiation(advanced_sitemap_readers):
    for reader in advanced_sitemap_readers:
        assert isinstance(reader, AdvancedSitemapReader)


@pytest.mark.parametrize(
    "advanced_sitemap_readers",
    ["reader/test_advanced_sitemap_reader.json"],
    indirect=True,
)
def test_load_data(advanced_sitemap_readers):
    for reader in advanced_sitemap_readers:
        data = reader.load_data()
        assert data is not None
        assert len(data) > 0


@pytest.mark.parametrize(
    "advanced_sitemap_readers",
    [
        "reader/test_advanced_sitemap_reader_sitemap_hs.json",
    ],
    indirect=True,
)
def test_load_data_sitemap_hs(advanced_sitemap_readers):
    for reader in advanced_sitemap_readers:
        data = reader.load_data()
        assert data is None
