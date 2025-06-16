import logging
from typing import Any, Dict, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType

logger = logging.getLogger(__name__)


class VerboseErrorLoggingHandler(BaseCallbackHandler):
    """Handler that logs detailed information about errors and specific events."""

    def __init__(self) -> None:
        super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
        self._trace_data = {}  # Pour stocker les données de traçage

        # Vérification précoce que le callback est fonctionnel
        logger.info("[CALLBACK] Initializing VerboseErrorLoggingHandler")
        try:
            # Test simple du système de log
            logger.debug("[CALLBACK] Debug logging test")
            logger.info("[CALLBACK] Info logging test")
            logger.info("[CALLBACK] Callback handler successfully initialized")
        except Exception as e:
            logger.error(
                f"[CALLBACK][CRITICAL] Failed to initialize callback handler: {e}"
            )
            raise RuntimeError("Callback handler initialization failed") from e

    def on_event_start(
        self, event_type: CBEventType, payload: Optional[Dict[str, Any]], **kwargs: Any
    ) -> None:
        logger.debug(f"[CALLBACK][START] {event_type.name}: {payload}")

    def on_event_end(
        self, event_type: CBEventType, payload: Optional[Dict[str, Any]], **kwargs: Any
    ) -> None:
        logger.debug(f"[CALLBACK][END] {event_type.name}")
        if event_type == CBEventType.EMBEDDING:
            embeddings = payload.get("embeddings", [])
            for i, emb in enumerate(embeddings):
                if emb is None:
                    logger.error(f"[CALLBACK][ERROR] Embedding {i} is None.")

    def on_event_error(
        self,
        event_type: CBEventType,
        payload: Optional[Dict[str, Any]],
        error: Exception,
        **kwargs: Any,
    ) -> None:
        logger.error(f"[CALLBACK][EXCEPTION] during {event_type.name}: {error}")
        logger.error(f"[CALLBACK][EXCEPTION PAYLOAD]: {payload}")

    def start_trace(self, trace_id: str) -> None:
        """Initialize trace data and verify initial state."""
        logger.info(f"[CALLBACK][TRACE START] {trace_id}")

        # Vérifie si une trace existe déjà
        existing_data = self._trace_data.get(trace_id, None)
        if existing_data:
            logger.warning(
                f"[CALLBACK][TRACE WARNING] Existing trace found for {trace_id}:"
            )
            for key, value in existing_data.items():
                logger.warning(f"[CALLBACK]  → {key}: {value}")

        # Initialise avec vérification explicite
        initial_state = {"docs_processed": 0, "docs_embedded": 0, "embedding_errors": 0}
        self._trace_data[trace_id] = initial_state

        # Log l'état initial avec confirmation explicite
        logger.info(f"[CALLBACK][TRACE VERIFICATION] Initial state for {trace_id}:")
        for key, value in initial_state.items():
            logger.info(f"[CALLBACK]  → {key}: {value} (confirmed empty)")

    def end_trace(self, trace_id: str) -> None:
        if trace_id in self._trace_data:
            stats = self._trace_data[trace_id]
            logger.info(f"[CALLBACK][TRACE END] {trace_id}")
            logger.info(f"[CALLBACK] Documents processed: {stats['docs_processed']}")
            logger.info(
                f"[CALLBACK] Documents successfully embedded: {stats['docs_embedded']}"
            )
            logger.info(f"[CALLBACK] Embedding errors: {stats['embedding_errors']}")

            if stats["docs_processed"] != stats["docs_embedded"]:
                logger.warning(
                    f"[CALLBACK] Mismatch detected: {stats['docs_processed']} docs processed "
                    f"but only {stats['docs_embedded']} embedded successfully"
                )

            del self._trace_data[trace_id]
