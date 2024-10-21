from typing import TYPE_CHECKING, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.storage import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine

from eurelis_llmatoolkit.llamaindex.abstract_wrapper import AbstractWrapper
from eurelis_llmatoolkit.llamaindex.factories.embedding_factory import EmbeddingFactory
from eurelis_llmatoolkit.llamaindex.factories.memory_factory import MemoryFactory
from eurelis_llmatoolkit.llamaindex.factories.retriever_factory import RetrieverFactory
from eurelis_llmatoolkit.llamaindex.factories.llm_factory import (
    LLMFactory,
)

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.base.llms.base import BaseLLM
    from llama_index.core.memory import BaseMemory
    from llama_index.core.retrievers import BaseRetriever


class ChatbotWrapper(AbstractWrapper):
    def __init__(self, config: dict):
        super().__init__(config)
        self._storage_context: Optional[StorageContext] = None
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: "BaseRetriever" = None
        self._embedding_model: "BaseEmbedding" = None
        self._llm: "BaseLLM" = None
        self._query_engine: Optional[RetrieverQueryEngine] = None
        self._memory: "BaseMemory" = None
        self._chat_engine = None

    def run(self, dataset_id: Optional[str] = None, use_cache: bool = False):
        """
        Runs the chatbot by initializing the vector store, storage context,
        index, retriever, query engine, and memory.

        Args:
            dataset_id: Optional, ID of the dataset to process.
            use_cache: Optional, flag to indicate if caching should be used.
        """
        vector_store = self._get_vector_store()
        storage_context = self._get_storage_context()
        index = self._get_index()
        retriever = self._get_retriever()
        llm = self._get_llm()
        memory = self._get_memory()
        query_engine = self._get_query_engine(retriever, llm)
        chat_engine = self._get_chat_engine()

        # Debug prints
        # print(f"VectorStore initialized: {vector_store}")
        # print(f"StorageContext initialized: {storage_context}")
        # print(f"Index initialized: {index}")
        # print(f"Retriever initialized: {retriever}")
        # print(f"LLM initialized: {llm}")
        # print(f"Memory initialized: {memory}")
        # print(f"QueryEngine initialized: {query_engine}")
        # print(f"ChatEngine initialized: {chat_engine}")

        # FIXME : Empty Response
        response = chat_engine.chat("Hello!")
        print(response)

        response = chat_engine.chat("What is Drupal about?")
        print(response)

        # response = chat_engine.chat("Can you tell me more?")
        # print(response)

    def _get_memory(self):
        """
        Creates a BaseMemory.

        Returns:
            BaseMemory: The configured memory.
        """
        if self._memory is not None:
            return self._memory

        memory_config = self._config.get("memory")
        if memory_config:
            self._memory = MemoryFactory.create_memory(memory_config)
        return self._memory

    def _get_storage_context(self):
        """
        Creates a StorageContext using the vector store and document store.

        Returns:
            StorageContext: The configured storage context.
        """
        if self._storage_context is not None:
            return self._storage_context

        # Get the vector store and document store
        vector_store = self._get_vector_store()
        document_store = self._get_document_store()

        # Create the StorageContext
        self._storage_context = StorageContext.from_defaults(
            vector_store=vector_store, docstore=document_store
        )

        return self._storage_context

    def _get_index(self):
        """
        Creates a VectorStoreIndex using the vector store and storage context.

        Returns:
            VectorStoreIndex: The configured index.
        """
        if self._index is not None:
            return self._index

        storage_context = self._get_storage_context()
        vector_store = self._get_vector_store()

        # Create the index using the vector store and storage context
        self._index = VectorStoreIndex.from_vector_store(
            vector_store, storage_context=storage_context
        )

        return self._index

    def _get_embedding_model(self):
        """
        Retrieves the embedding model from the configuration.

        Returns:
            BaseEmbedding: The configured embedding model.
        """
        if self._embedding_model is not None:
            return self._embedding_model

        embedding_config = self._config.get("embeddings")
        if embedding_config:
            self._embedding_model = EmbeddingFactory.create_embedding(embedding_config)

        return self._embedding_model

    def _get_retriever(self):
        """
        Creates a Retriever using the index and embedding model.

        Returns:
            Retriever: The configured retriever.
        """
        if self._retriever is not None:
            return self._retriever

        index = self._get_index()
        embedding_model = self._get_embedding_model()

        retriever_config = self._config.get("retriever")
        if retriever_config:
            self._retriever = RetrieverFactory.create_retriever(
                retriever_config, index=index, embedding_model=embedding_model
            )

        return self._retriever

    def _get_llm(self):
        """
        Retrieves the language model (LLM) from the configuration.

        Returns:
            BaseLLM: The configured language model.
        """
        if self._llm is not None:
            return self._llm

        llm_config = self._config.get("llm")
        if llm_config:
            self._llm = LLMFactory.create_llm(llm_config)

        return self._llm

    def _get_query_engine(self, retriever, llm):
        """
        Creates a query engine using the retriever and language model.

        Args:
            retriever: The retriever to be used for querying.
            llm: The language model to be used for querying.

        Returns:
            RetrieverQueryEngine: The configured query engine.
        """
        if self._query_engine is not None:
            return self._query_engine

        self._query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever, llm=llm
        )

        return self._query_engine

    def _get_chat_engine(self):
        """
        Creates a chat engine using the index, chat mode, memory, and system prompt.

        Returns:
            ChatEngine: The configured chat engine.
        """
        if self._chat_engine is not None:
            return self._chat_engine

        index = self._get_index()
        memory = self._get_memory()

        # Create the chat engine with the specified configuration
        chat_mode = self._config.get("chat_mode")
        system_prompt_list = self._config.get("system_prompt")

        if isinstance(system_prompt_list, list):
            system_prompt = "\n".join(system_prompt_list)
        elif isinstance(system_prompt_list, str):
            system_prompt = system_prompt_list
        else:
            raise ValueError(
                "The 'system_prompt' should be either a list of strings or a single string."
            )

        self._chat_engine = index.as_chat_engine(
            chat_mode=chat_mode,
            memory=memory,
            system_prompt=system_prompt,
        )

        return self._chat_engine
