# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Fix in `AdvancedSitemapReader` for when the metadata `lastmod` does not exist

### Changed

### Removed

## [2.0.0dev9] - 2024-02-24

### Added

- Manage timeout during sitemap ingestion
- Environment variable to set Sentry environment tag

## [2.0.0dev8] - 2024-02-13

### Changed

- Fix ingestion wrapper

## [2.0.0dev7] - 2024-01-29

### Added

- Data cleaning in ingestion wrapper (delete obsolete data)
- Sentry support in console mode
- Add timeout to sitemap reader configuration

### Changed

- Logging strategy improvements

### Removed

- Legacy code source

## [2.0.0dev6] - 2024-01-11

- No exception raised if .env file not found
- Limit pymupdf4llm.to_markdown verbosity

## [2.0.0dev5] - 2024-01-09

- Add node post-processor support
- Increaseed timeout for requests in AdvancedSitemaReader

## [2.0.0dev4] - 2024-12-31

- Fix motor & pymongo version

## [2.0.0dev3] - 2024-12-24

- Improved chatbot wrapper and memory management
- LLM Node Transformer

## [2.0.0dev2] - 2024-10-29

- Improved ingestion pipeline & chatbot wrapper
- Custom retriever

## [2.0.0dev1] - 2024-10-22

- 1st integration with LlmaIndex

## [1.1.0rc2] - 2024-09-30

- MongoDb RecordManager
- On demand indexation
- Search API

## [1.1.0rc1] - 2024-06-17

- Langchain 0.2.2 update
- Add CITATION.cff
- Add SECURITY.md

## [1.0.1] - 2024-05-22

- Factory update for Solr VectorStore
- Variabilization of the limit on the number of requests per second when scrapping via sitemap

## [1.0.0] - 2024-05-20

- First release

## [1.0.0rc1] - 2024-05-16

- First release candidate
