# LlmToolkit

![Python : 11](https://img.shields.io/badge/Python-=3.11-green)
[![PyPI version](https://img.shields.io/pypi/v/llmtoolkit.svg)](https://pypi.org/project/llmtoolkit/)
[![PyPI downloads](https://img.shields.io/pypi/dm/llmtoolkit.svg)](https://pypi.org/project/llmtoolkit/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![Code style : black](https://img.shields.io/badge/Code_style-black-black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[![Open Issues](https://img.shields.io/github/issues-raw/Eurelis/Eurelis-LlmaToolkit)](https://github.com/Eurelis/Eurelis-LlmaToolkit/issues)
[![GitHub star chart](https://img.shields.io/github/stars/Eurelis/Eurelis-LlmaToolkit?style=social)](https://star-history.com/#Eurelis/Eurelis-LlmaToolkit)
[![X (formerly Twitter) URL](https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2FAgence_Eurelis&label=Follow%20%40Eurelis)](https://x.com/Agence_Eurelis)

**LlmToolkit** is a comprehensive Python framework for building and managing AI-based applications. Built on top of LlamaIndex, it provides a streamlined interface for creating chatbots, search engines, and document processing pipelines.

## 🚀 Quick Start

### Installation

Install from PyPI:

```bash
pip install llmtoolkit
```

Install with specific features:

```bash
# For OpenAI integration
pip install llmtoolkit[openai]

# For local embeddings with Hugging Face
pip install llmtoolkit[huggingface]

# For MongoDB vector storage
pip install llmtoolkit[mongodb]

# For ChromaDB vector storage
pip install llmtoolkit[chroma]

# For web scraping capabilities
pip install llmtoolkit[sitemap]

# For PDF processing
pip install llmtoolkit[pdf]

# Install all features
pip install llmtoolkit[llamaindex,openai,huggingface,mongodb,chroma,sitemap,pdf,markdown,sentry]
```

### Basic Usage

To be done...

## 🛠️ Features

- **🤖 Chat Engines**: Build conversational AI applications with memory persistence
- **🔍 Search Engines**: Implement semantic search across your document collections
- **📚 Document Ingestion**: Process various document formats (PDF, TXT, Web pages)
- **🗄️ Vector Stores**: Support for multiple vector databases (MongoDB Atlas, ChromaDB)
- **🔗 LLM Integration**: Compatible with OpenAI, Anthropic, and local models
- **🧠 Memory Management**: Persistent chat memory with JSON storage
- **🌐 Web Scraping**: Advanced sitemap and webpage readers
- **⚙️ Configurable**: JSON-based configuration for easy customization
- **🔄 Transformations**: Built-in text transformations and metadata enrichment
- **📊 Monitoring**: Sentry integration for error tracking

## 🎯 Use Cases

- **Knowledge Base Systems**: Build intelligent document search and Q&A systems
- **Customer Support Chatbots**: Create context-aware conversational agents
- **Document Analysis**: Process and analyze large document collections
- **Research Assistance**: Develop AI assistants for research and discovery
- **Content Management**: Implement semantic search for content repositories

## 📝 Changelog

See [CHANGELOG.md](CHANGELOG.md) for detailed release notes.

## 🔒 Security

For security concerns, please see our [Security Policy](SECURITY.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [LlamaIndex](https://www.llamaindex.ai/)
- Inspired by the open-source AI community
- Developed with ❤️ by the [Eurelis](https://www.eurelis.com/) team

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/Eurelis/Eurelis-LlmaToolkit/issues)
- **Email**: Contact us at [support@eurelis.com](mailto:support@eurelis.com)
- **Twitter**: Follow [@Agence_Eurelis](https://x.com/Agence_Eurelis) for updates

## 📚 Citation

If you use LlmToolkit in your research, please cite our work. See [CITATION.cff](CITATION.cff) for details.
