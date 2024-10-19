from eurelis_llmatoolkit.llamaindex.factories.reader_factory import ReaderFactory


def test_load_community_simple_web_page_reader():
    config = {
        "provider": "CommunitySimpleWebPageReader",
        "urls": ["https://example.com/page1", "https://example.com/page2"],
    }
    reader = ReaderFactory.create_reader("eurelis", config)
    assert reader is not None
    assert reader.__class__.__name__ == "CommunitySimpleWebPageReader"


def test_load_community_sitemap_reader():
    config = {
        "provider": "CommunitySitemapReader",
        "url": "https://www.eurelis.com/sitemap.xml",
    }
    reader = ReaderFactory.create_reader("eurelis", config)
    assert reader is not None
    assert reader.__class__.__name__ == "CommunitySitemapReader"


def test_load_advanced_sitemap_reader():
    config = {
        "provider": "AdvancedSitemapReader",
        "sitemap_url": "<https://example.com/sitemap.xml>",
        "user_agent": "EurelisLLMATK/0.1",
        "requests_per_second": 4,
        "embed_pdf": True,
    }
    reader = ReaderFactory.create_reader("eurelis", config)
    assert reader is not None
    assert reader.__class__.__name__ == "AdvancedSitemapReader"


def test_load_txt_file_reader():
    config = {
        "provider": "TXTFileReader",
        "base_dir": "../etc/data_sample/",
        "glob": "*.txt",
    }
    reader = ReaderFactory.create_reader("eurelis", config)
    assert reader is not None
    assert reader.__class__.__name__ == "TXTFileReader"


def test_load_pdf_file_reader():
    config = {
        "provider": "PDFFileReader",
        "base_dir": "../etc/data_sample/",
        "glob": "*.pdf",
    }
    reader = ReaderFactory.create_reader("eurelis", config)
    assert reader is not None
    assert reader.__class__.__name__ == "PDFFileReader"


def test_load_custom_reader():
    config = {
        "provider": "eurelis_llmatoolkit.llamaindex.readers.community_sitemap_reader.CommunitySitemapReader",
        "url": "https://www.eurelis.com/sitemap.xml",
    }
    reader = ReaderFactory.create_reader("eurelis", config)
    assert reader is not None
    assert reader.__class__.__name__ == "CommunitySitemapReader"
