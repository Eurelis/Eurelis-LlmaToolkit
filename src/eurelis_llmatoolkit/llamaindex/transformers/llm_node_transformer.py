import copy
from typing import List, Any, Sequence, Union
import uuid
from llama_index.core.node_parser import NodeParser
from llama_index.core.schema import BaseNode
from pydantic import BaseModel
from eurelis_llmatoolkit.llamaindex.factories.llm_factory import LLMFactory


class LLMResponseSchema(BaseModel):
    responses: List[str]


class LLMNodeTransformer(NodeParser):
    def __init__(self, config: dict):
        """
        Initializes the transformer with configuration settings.

        Args:
            config (dict): Configuration for the transformer, including LLM information
            and transformation options.
        """
        super().__init__()
        self._config = config
        self._llm = None
        self._prompt = config.get("prompt", "")
        self._keep_origin_node = config.get("keep_origin_node", True)
        self._mode = config.get("mode")
        self._llm_provider = config.get("llm_provider")
        self._llm_model = config.get("llm_model")

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        """
        Transforms nodes by generating new nodes based on their content.

        Args:
            nodes (Sequence[BaseNode]): List of nodes to process.

        Returns:
            List[BaseNode]: List of transformed nodes with generated content.
        """
        updated_nodes = []
        for node in nodes:
            original_content = node.get_content()

            generated_content = self._generate_content_from_llm(original_content)

            if generated_content:
                # Créer un ou plusieurs nouveaux noeuds avec la réponse du LLM
                new_nodes = self._create_transformed_nodes(
                    node, generated_content, original_content
                )
                updated_nodes.extend(new_nodes)

        return updated_nodes

    def _generate_content_from_llm(self, content: str) -> List[str]:
        """
        Calls the LLM to generate content based on the original node's content.

        Args:
            content (str): Original content of the node.

        Returns:
            List[str]: Content generated by the LLM.
        """
        prompt = self._format_prompt(content)
        llm = self._get_llm()
        sllm = llm.as_structured_llm(output_cls=LLMResponseSchema)

        try:
            return sllm.complete(prompt).raw.responses
        except Exception as e:
            print(f"Error generating content via LLM: {e}")
            return []

    def _get_llm(self):
        """
        Initializes and returns the LLM instance if not already initialized.

        Returns:
            LLM: LLM instance configured based on provided settings.
        """
        if self._llm is None:
            llm_config = {
                "provider": self._llm_provider,
                "model": self._llm_model,
                "api_key": self._config.get("llm_api_key"),
            }
            self._llm = LLMFactory.create_llm(llm_config)
        return self._llm

    def _format_prompt(self, content: str) -> str:
        """
        Formats the prompt by combining the initial prompt message with the node's content.

        Args:
            content (str): The content of the node to be passed to the LLM.

        Returns:
            str: The formatted prompt.
        """
        return f"{self._prompt}\nMessage:\n{content}\nResponse:"

    def _create_transformed_nodes(
        self,
        original_node: BaseNode,
        generated_content: Union[str, List[str]],
        original_content: str,
    ) -> List[BaseNode]:
        """
        Creates one or more new nodes with the generated content.

        Args:
            original_node (BaseNode): The original node.
            generated_content (Union[str, List[str]]): Content generated by the LLM.
            original_content (str): Original content of the node.

        Returns:
            List[BaseNode]: List of new nodes created with generated content.
        """
        # Convertir en liste pour traiter les deux formats (str ou list)
        contents = (
            [generated_content]
            if isinstance(generated_content, str)
            else generated_content
        )
        transformed_nodes = []

        # Conserver le noeud d'origine si True
        if self._keep_origin_node:
            transformed_nodes.append(original_node)

        # Crée un nouveau noeud pour chaque élément de contenu généré
        for content in contents:
            new_node: BaseNode = copy.deepcopy(original_node)
            new_node.id_ = str(uuid.uuid4())

            new_node.set_content(value=content)

            # Ajouter les métadonnées spécifiques sur le nouveau noeud
            self._add_generated_metadata(new_node, original_content)
            transformed_nodes.append(new_node)

        return transformed_nodes

    def _add_generated_metadata(self, node: BaseNode, original_content: str):
        """
        Adds metadata to indicate that the content is generated.

        Args:
            node (BaseNode): Node to which metadata will be added.
            original_content (str): The original content used for generation.
        """
        node.metadata["generated_content"] = True
        node.metadata["generated_content_mode"] = self._mode
        node.metadata["generated_content_origin"] = original_content
