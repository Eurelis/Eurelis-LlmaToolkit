import logging
from typing import Any, Dict, Optional, Set, List
from dataclasses import dataclass, field
from collections import defaultdict

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

logger = logging.getLogger(__name__)


@dataclass
class DocumentInfo:
    source: str
    content_id: str
    error: Optional[str] = None


@dataclass
class TraceStats:
    transform_input_docs: int = 0
    transform_input_null: int = 0
    transform_output_docs: int = 0
    transform_output_null: int = 0
    embedding_input_docs: int = 0
    embedding_input_null: int = 0
    embedding_success: int = 0
    embedding_null: int = 0
    vectorstore_saved: int = 0
    failed_docs: List[DocumentInfo] = field(default_factory=list)
    success_docs: List[DocumentInfo] = field(default_factory=list)
    all_success_sources: Set[str] = field(default_factory=set)
    current_stage: Optional[str] = None


class VerboseErrorLoggingHandler(BaseCallbackHandler):
    """Handler that logs detailed information about errors and specific events."""

    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._trace_data: Dict[str, TraceStats] = defaultdict(TraceStats)

        logger.info("[CALLBACK] Initializing VerboseErrorLoggingHandler")
        try:
            logger.debug("[CALLBACK] Info logging test")
            logger.debug("[CALLBACK] Callback handler successfully initialized")
        except Exception as e:
            logger.error(
                f"[CALLBACK][CRITICAL] Failed to initialize callback handler: {e}"
            )
            raise RuntimeError("Callback handler initialization failed") from e

    def _log_stats(self, prefix: str, stats: Dict[str, int]) -> None:
        for key, value in stats.items():
            logger.debug(f"[CALLBACK] {prefix} {key}: {value}")

    def _extract_chunk_info(self, chunk: str) -> DocumentInfo:
        metadata = chunk.split("\n")
        return DocumentInfo(
            source=next(
                (
                    line.split(": ")[1]
                    for line in metadata
                    if line.startswith("source: ")
                ),
                "unknown",
            ),
            content_id=next(
                (
                    line.split(": ")[1]
                    for line in metadata
                    if line.startswith("c_contentId: ")
                ),
                "unknown",
            ),
        )

    def _update_trace_stats(self, event_id: str, key: str, value: int) -> None:
        stats = self._trace_data[event_id]
        setattr(stats, key, getattr(stats, key, 0) + value)

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        """Start a trace with the given trace ID.

        Args:
            trace_id: Optional ID for this trace
        """
        if trace_id is None:
            trace_id = "default"
        logger.debug(f"[CALLBACK][TRACE START] {trace_id}")
        initial_state = TraceStats()
        self._trace_data[trace_id] = initial_state
        logger.debug(f"[CALLBACK][TRACE INITIALIZED] {trace_id}")

    def _log_payload_counts(self, payload: Optional[Dict[str, Any]]) -> None:
        """Log counts for each key in the payload."""
        if not payload:
            return
        for k, v in payload.items():
            try:
                count = (
                    len(v)
                    if hasattr(v, "__len__") and not isinstance(v, str)
                    else "N/A"
                )
            except Exception:
                count = "N/A"
            logger.debug(f"[CALLBACK][PAYLOAD COUNT] {k}: {count}")

    def _log_sources_summary(self, stats: TraceStats, context: str = "") -> None:
        """Log summary of successful and failed sources."""
        failed_sources = sorted(list({doc.source for doc in stats.failed_docs}))
        success_sources = sorted(list(stats.all_success_sources))
        logger.debug(f"=== RÉSUMÉ DES SOURCES {context}===")
        logger.debug("SOURCES EN ÉCHEC:")
        logger.debug(str(failed_sources))
        logger.debug("SOURCES EN SUCCÈS:")
        logger.debug(str(success_sources))
        logger.debug("=" * 50)

    def _log_document_details(self, stats: TraceStats) -> None:
        """Log details about failed and successful documents."""
        if stats.failed_docs:
            logger.error("[CALLBACK][FAILED DOCUMENTS]")
            for doc in stats.failed_docs:
                logger.error(f"  - Source: {doc.source}, ID: {doc.content_id}")

        if stats.success_docs:
            logger.debug("[CALLBACK][SUCCESSFUL DOCUMENTS (sample)]")
            for doc in stats.success_docs:
                logger.debug(f"  - Source: {doc.source}, ID: {doc.content_id}")

    def on_event_start(
        self, event_type: CBEventType, payload: Optional[Dict[str, Any]], **kwargs: Any
    ) -> str:
        event_id = kwargs.get("event_id", "default")
        logger.debug(f"[CALLBACK][START] -> {event_type.name}")
        logger.debug(f"[CALLBACK][START] -> {event_type.name}: {payload}")
        self._log_payload_counts(payload)

        if event_type == CBEventType.NODE_PARSING:
            docs = payload.get("documents", []) if payload else []
            if docs:
                null_docs = sum(1 for doc in docs if doc is None)
                valid_docs = len(docs) - null_docs
                self._update_trace_stats(event_id, "transform_input_docs", valid_docs)
                self._update_trace_stats(event_id, "transform_input_null", null_docs)
                logger.debug(
                    f"[CALLBACK][TRANSFORM START] Processing {valid_docs} documents"
                )

        elif event_type == CBEventType.EMBEDDING:
            if payload and EventPayload.SERIALIZED in payload:
                logger.debug("[CALLBACK][EMBEDDING START] Starting embedding process")

        return event_id

    def _ensure_trace_exists(self, event_id: str) -> None:
        """Ensure that a TraceStats object exists for the given event_id."""
        if event_id not in self._trace_data:
            self._trace_data[event_id] = TraceStats()

    def on_event_end(
        self, event_type: CBEventType, payload: Optional[Dict[str, Any]], **kwargs: Any
    ) -> None:
        event_id = kwargs.get("event_id", "default")
        self._ensure_trace_exists(event_id)
        logger.debug(f"[CALLBACK][END] {event_type.name}")
        logger.debug(f"[CALLBACK][END] -> {event_type.name}: {payload}")
        self._log_payload_counts(payload)

        if not payload:
            logger.error("[CALLBACK][ERROR] Empty payload")
            return

        if event_type == CBEventType.NODE_PARSING:
            self._handle_node_parsing_end(event_id, payload)
        elif event_type == CBEventType.EMBEDDING:
            self._handle_embedding_end(event_id, payload)

    def _handle_node_parsing_end(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle node parsing end event."""
        nodes = payload.get("nodes", [])
        if not nodes:
            return

        self._update_trace_stats(event_id, "transform_output_docs", len(nodes))
        logger.debug(f"[CALLBACK][TRANSFORM END] Created {len(nodes)} nodes")

        empty_nodes = [n for n in nodes if not getattr(n, "text", "").strip()]
        if empty_nodes:
            logger.warning(
                f"[CALLBACK][TRANSFORM WARNING] Found {len(empty_nodes)} empty nodes"
            )

    def _handle_embedding_end(self, event_id: str, payload: Dict[str, Any]) -> None:
        """Handle embedding end event."""
        embeddings = payload.get(EventPayload.EMBEDDINGS, [])
        chunks = payload.get(EventPayload.CHUNKS, [])

        if not isinstance(embeddings, list):
            logger.error(
                f"[CALLBACK][EMBEDDING ERROR] Invalid embeddings type: {type(embeddings)}"
            )
            return

        if len(embeddings) == 1 and chunks and len(chunks) > 1:
            self._handle_single_embedding_error(event_id, chunks)
            return

        if embeddings and chunks:
            self._process_embeddings_and_chunks(event_id, embeddings, chunks)
        else:
            logger.error(
                f"[CALLBACK][EMBEDDING DEBUG] Missing data - E: {bool(embeddings)}, C: {bool(chunks)}"
            )
            logger.error(
                f"[CALLBACK][EMBEDDING DEBUG] Payload keys: {list(payload.keys())}"
            )

    def _handle_single_embedding_error(self, event_id: str, chunks: List[str]) -> None:
        """Handle error case of single embedding for multiple chunks."""
        logger.error(
            "[CALLBACK][EMBEDDING ERROR] Single embedding returned for multiple chunks"
        )
        logger.error(f"[CALLBACK][EMBEDDING DEBUG] Number of chunks: {len(chunks)}")

        for chunk in chunks:
            doc_info = self._extract_chunk_info(chunk)
            doc_info.error = (
                "Invalid embedding format - single embedding for multiple chunks"
            )
            self._trace_data[event_id].failed_docs.append(doc_info)

        self._update_trace_stats(event_id, "embedding_null", len(chunks))

    def _process_embeddings_and_chunks(
        self, event_id: str, embeddings: List[Any], chunks: List[str]
    ) -> None:
        """Process embeddings and chunks, updating statistics and logging results."""
        success_details = []
        failed_details = []

        for chunk, embedding in zip(chunks, embeddings):
            doc_info = self._extract_chunk_info(chunk)
            if embedding is None:
                doc_info.error = "Null embedding"
                self._trace_data[event_id].failed_docs.append(doc_info)
                failed_details.append(doc_info)
            else:
                self._trace_data[event_id].all_success_sources.add(doc_info.source)
                if len(self._trace_data[event_id].success_docs) < 5:
                    self._trace_data[event_id].success_docs.append(doc_info)
                    success_details.append(doc_info)

        null_count = sum(1 for emb in embeddings if emb is None)
        success_count = len(embeddings) - null_count

        self._update_trace_stats(event_id, "embedding_success", success_count)
        self._update_trace_stats(event_id, "embedding_null", null_count)

        logger.info(
            f"[CALLBACK][EMBEDDING END] Successful: {success_count}, Null: {null_count}"
        )

        if failed_details:
            logger.error(
                "[CALLBACK][EMBEDDING FAILURES] Documents with null embeddings:"
            )
            for doc in failed_details:
                logger.error(f"  - Source: {doc.source}, ID: {doc.content_id}")

        if success_details:
            logger.debug(
                "[CALLBACK][EMBEDDING SUCCESS DETAILS] First 5 successful embeddings:"
            )
            for doc in success_details:
                logger.debug(f"  - Source: {doc.source}, ID: {doc.content_id}")

        stats = self._trace_data[event_id]
        self._log_sources_summary(stats)

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Run when an overall trace is exited.

        Args:
            trace_id: Optional ID for the trace
            trace_map: Optional mapping of trace information
        """
        if trace_id is None:
            trace_id = "default"

        if trace_id in self._trace_data:
            stats = self._trace_data[trace_id]
            # Log summary statistics
            logger.debug(f"[CALLBACK][TRACE SUMMARY] {trace_id}")
            logger.debug(
                f"[CALLBACK] Transform input documents: {stats.transform_input_docs} (Null: {stats.transform_input_null})"
            )
            logger.debug(
                f"[CALLBACK] Transform output nodes: {stats.transform_output_docs}"
            )
            logger.debug(
                f"[CALLBACK] Embedding input nodes: {stats.embedding_input_docs}"
            )
            logger.debug(f"[CALLBACK] Successful embeddings: {stats.embedding_success}")
            logger.debug(f"[CALLBACK] Null embeddings: {stats.embedding_null}")

            # Log documents and sources details
            self._log_document_details(stats)
            self._log_sources_summary(stats, "FINAL ")

            # Log alerts
            if stats.embedding_null > 0:
                logger.error(
                    f"[CALLBACK][ALERT] {stats.embedding_null} documents have null embeddings!"
                )

            if stats.embedding_input_docs != (
                stats.embedding_success + stats.embedding_null
            ):
                logger.error("[CALLBACK][ALERT] Mismatch in embedding counts!")

            # Clean up trace data
            del self._trace_data[trace_id]
            # Compute sources before logging
            failed_sources = sorted(list({doc.source for doc in stats.failed_docs}))
            success_sources = sorted(list(stats.all_success_sources))

            logger.debug("=== RÉSUMÉ DES SOURCES ===")
            logger.debug("SOURCES EN ÉCHEC:")
            logger.debug(str(failed_sources))
            logger.debug("SOURCES EN SUCCÈS:")
            logger.debug(str(success_sources))
            logger.debug("=" * 50)

            # Clean up trace data
            del self._trace_data[trace_id]
            for doc in stats.failed_docs:
                logger.error(f"  - Source: {doc.source}, ID: {doc.content_id}")

            if stats.success_docs:
                logger.debug("[CALLBACK][SUCCESSFUL DOCUMENTS (sample)]")
                for doc in stats.success_docs:
                    logger.debug(f"  - Source: {doc.source}, ID: {doc.content_id}")

            # Log alerts
            if stats.embedding_null > 0:
                logger.error(
                    f"[CALLBACK][ALERT] {stats.embedding_null} documents have null embeddings!"
                )
