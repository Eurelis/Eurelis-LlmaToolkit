# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Removed

## [2.0.0dev15] - 2025-08-27

### Added

- ReAct wrapper

### Changed

- Update `llama-index` dependencies from `0.12.25` to `0.12.41`

## [2.0.0dev14] - 2025-07-02

### Added

- New `retrieve_similar_docs` method in `ChatbotWrapper` to retrieve documents similar to a given text using the retriever.

### Changed

### Removed

## [2.0.0dev13] - 2025-06-23

### Added

- Enhanced OpenAI embedding creation with configurable batch size and retry parameters
- Improved error logging for failed URL fetches in AdvancedSitemapReader
- Added detailed progress logging for node transformation in LLMNodeTransformer
- New `VerboseErrorLoggingHandler` callback (llamaindex) and its factory for detailed error and event logging in the ingestion pipeline

### Changed

- Update dependancies

### Removed

## [2.0.0dev12] - 2025-03-24

### Changed

- Update `llama-index` dependencies from `0.11.14` to `0.12.25`

## [2.0.0dev11] - 2025-03-19

### Added

- New VectorSearch CustomMongoDBAtlasVectorSearch with filter tree compatibility
- Update dépendancies

### Changed

### Removed

## [2.0.0dev10] - 2025-03-10

### Added

- Fix in `AdvancedSitemapReader` for when the metadata `lastmod` does not exist
- Fix in `IngestionWrapper` with cache

### Changed

### Removed

## [2.0.0dev9] - 2025-02-24

### Added

- Manage timeout during sitemap ingestion
- Environment variable to set Sentry environment tag

## [2.0.0dev8] - 2025-02-13

### Changed

- Fix ingestion wrapper

## [2.0.0dev7] - 2025-01-29

### Added

- Data cleaning in ingestion wrapper (delete obsolete data)
- Sentry support in console mode
- Add timeout to sitemap reader configuration

### Changed

- Logging strategy improvements

### Removed

- Legacy code source

## [2.0.0dev6] - 2025-01-11

- No exception raised if .env file not found
- Limit pymupdf4llm.to_markdown verbosity

## [2.0.0dev5] - 2025-01-09

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
