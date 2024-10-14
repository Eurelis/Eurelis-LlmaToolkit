from typing import TYPE_CHECKING, Optional
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage import StorageContext
from llama_index.core.query_engine import RetrieverQueryEngine

from eurelis_llmatoolkit.llamaindex.factories.documentstore_factory import (
    DocumentStoreFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.embedding_factory import EmbeddingFactory
from eurelis_llmatoolkit.llamaindex.factories.vectorstore_factory import (
    VectorStoreFactory,
)
from eurelis_llmatoolkit.llamaindex.factories.llm_factory import (
    LLMFactory,
)

if TYPE_CHECKING:
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.vector_stores.types import BasePydanticVectorStore
    from llama_index.core.base.llms.base import BaseLLM


class ChatbotWrapper:
    def __init__(self, config: dict):
        self._config: dict = config
        self._vector_store: "BasePydanticVectorStore" = None
        self._document_store = None
        self._storage_context: Optional[StorageContext] = None
        self._index: Optional[VectorStoreIndex] = None
        self._retriever: Optional[VectorIndexRetriever] = None
        self._embedding_model: "BaseEmbedding" = None
        self._llm: "BaseLLM" = None
        self._query_engine: Optional[RetrieverQueryEngine] = None
        self._memory: Optional[ChatMemoryBuffer] = None
        self._chat_engine = None

    def run(self):
        """
        Runs the chatbot by initializing the vector store, storage context,
        index, retriever, query engine, and memory.
        """
        vector_store = self._get_vector_store()
        storage_context = self._get_storage_context()
        index = self._get_index()
        retriever = self._get_retriever()
        llm = self._get_llm()
        memory = self._get_memory()
        query_engine = self._get_query_engine(retriever, llm)
        chat_engine = self._get_chat_engine()

        print(f"VectorStore initialized: {vector_store}")
        print(f"StorageContext initialized: {storage_context}")
        print(f"Index initialized: {index}")
        print(f"Retriever initialized: {retriever}")
        print(f"LLM initialized: {llm}")
        print(f"Memory initialized: {memory}")
        print(f"QueryEngine initialized: {query_engine}")
        print(f"ChatEngine initialized: {chat_engine}")

        # FIXME : Empty Response
        response = chat_engine.chat("Hello!")
        print(response)

        response = chat_engine.chat("What is Drupal about?")
        print(response)

        response = chat_engine.chat("Can you tell me more?")
        print(response)

    def _get_memory(self):
        """
        Creates a ChatMemoryBuffer with a token limit.

        Returns:
            ChatMemoryBuffer: The configured memory buffer.
        """
        if self._memory is not None:
            return self._memory

        # Initialize ChatMemoryBuffer with a token limit of 1500
        self._memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

        return self._memory

    def _get_vector_store(self):
        """
        Retrieves the vector store from the configuration.

        Returns:
            BasePydanticVectorStore: The configured vector store.
        """
        if self._vector_store is not None:
            return self._vector_store

        vectorstore_config = self._config.get("vectorstore")
        if vectorstore_config:
            self._vector_store = VectorStoreFactory.create_vector_store(
                vectorstore_config
            )
        return self._vector_store

    def _get_document_store(self):
        """
        Retrieves the document store from the configuration.

        Returns:
            DocumentStore: The configured document store.
        """
        if self._document_store is not None:
            return self._document_store

        documentstore_config = self._config.get("documentstore")
        if documentstore_config:
            self._document_store = DocumentStoreFactory.create_document_store(
                documentstore_config
            )

        return self._document_store

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
        Creates a VectorIndexRetriever using the index and embedding model.

        Returns:
            VectorIndexRetriever: The configured retriever.
        """
        if self._retriever is not None:
            return self._retriever

        index = self._get_index()
        embedding_model = self._get_embedding_model()

        # Create the retriever with a specified number of results
        self._retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,  # TODO Utiliser une VAR d'ENV
            filter=None,
            embedding_model=embedding_model,
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
        self._chat_engine = index.as_chat_engine(
            chat_mode=ChatMode.CONTEXT,
            memory=memory,
            system_prompt=(
                "You are a chatbot, able to have normal interactions, as well as talk "
                "about an essay discussing Paul Graham's life."
            ),
        )

        return self._chat_engine
