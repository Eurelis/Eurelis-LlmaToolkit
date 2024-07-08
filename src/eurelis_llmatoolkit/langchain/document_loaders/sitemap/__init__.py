import types
from typing import TYPE_CHECKING, Any
from bs4 import BeautifulSoup
from eurelis_llmatoolkit.utils.base_factory import ParamsDictFactory
from langchain_core.document_loaders import BaseLoader

if TYPE_CHECKING:
    from eurelis_llmatoolkit.langchain.langchain_wrapper import BaseContext


def _parsing_function_factory(parsing_config: list):
    if not parsing_config:
        return lambda content: content.get_text()

    def parse_include_exclude(content: Any) -> str:
        def parsing_exclude_function(content: Any) -> str:
            for remove in parsing_list:
                if isinstance(remove, str):  # tagname
                    nodes = content.find_all(remove)
                    for node in nodes:
                        node.extract()
                elif isinstance(remove, dict):
                    nodes = content.find_all(**remove)
                    for node in nodes:
                        node.extract()

            return content
      
        def parsing_include_function(content: Any) -> str:
            # Créer une nouvelle structure BeautifulSoup pour les éléments conservés
            soup = BeautifulSoup("<div></div>", "html.parser")
            container = soup.div

            # Trouver et ajouter les éléments à conserver
            for include in parsing_list:
                if isinstance(include, str):  # tagname
                    nodes = content.find_all(include)
                elif isinstance(include, dict):
                    nodes = content.find_all(**include)

                for node in nodes:
                    container.append(node)

            return container
      
        for operation in parsing_config:
            op_type = operation.get("operation")
            parsing_list = operation.get("elements", [])

            if op_type == "parser_include":
                content = parsing_include_function(content)
            elif op_type == "parser_exclude":
                parsing_exclude_function(content)

        return content.get_text("\n")

    return parse_include_exclude


def _meta_function(meta: dict, content: Any) -> dict:
    metadata = {"source": meta["loc"], **meta}

    title = content.find("title")
    description = content.find("meta", attrs={"name": "description"})
    html = content.find("html")

    if title:
        metadata["title"] = title.get_text()
    if description:
        metadata["description"] = description.get("content", "")
    if html:
        metadata["language"] = html.get("lang", "")

    return metadata


class SitemapDocumentLoaderFactory(ParamsDictFactory[BaseLoader]):
    OPTIONAL_PARAMS = {
        "filter_urls",
        "blocksize",
        "is_local",
        "continue_on_failure",
        "restrict_to_same_domain",
        "requests_per_second",
    }

    def build(self, context: "BaseContext") -> BaseLoader:
        """
        Construct the sitemap document loader

        Args:
            context: the context object, usually the current langchain wrapper instance

        Returns:
            a document loader
        """
        from langchain_community.document_loaders import SitemapLoader
        from langchain_community.document_loaders.web_base import (
            default_header_template,
        )

        web_path = self.params.get("web_path")

        if not web_path:
            raise ValueError("Missing required web_path parameter")

        header_template = default_header_template.copy()
        # default user agent is "EurelisLlmaToolkit/0.1"
        header_template["User-Agent"] = self.params.get(
            "user_agent", "EurelisLlmaToolkit/1.0"
        )

        parameters = self.get_optional_params()

        loader = SitemapLoader(
            web_path,
            parsing_function=_parsing_function_factory(
                self.params.get("parser")
            ),
            meta_function=_meta_function,
            header_template=header_template,
            **parameters,  # type: ignore[arg-type]
        )

        # hack as SitemapLoader does not override lazy_load from WebBaseLoader, we will force
        # the dataset to use the load method instead
        def lazy_load(instance):
            raise NotImplementedError("")

        loader.lazy_load = types.MethodType(lazy_load, loader)  # type: ignore[method-assign]

        return loader
