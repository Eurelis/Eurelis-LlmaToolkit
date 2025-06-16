import logging
from typing import Any, Dict, Optional

from llama_index.core.callbacks.base_handler import BaseCallbackHandler
from llama_index.core.callbacks.schema import CBEventType, EventPayload

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
            logger.info("[CALLBACK] Info logging test")
            logger.info("[CALLBACK] Callback handler successfully initialized")
        except Exception as e:
            logger.error(
                f"[CALLBACK][CRITICAL] Failed to initialize callback handler: {e}"
            )
            raise RuntimeError("Callback handler initialization failed") from e

    def start_trace(self, trace_id: str) -> None:
        logger.info(f"[CALLBACK][TRACE START] {trace_id}")

        initial_state = {
            "transform_input_docs": 0,
            "transform_input_null": 0,
            "transform_output_docs": 0,
            "transform_output_null": 0,
            "embedding_input_docs": 0,
            "embedding_input_null": 0,
            "embedding_success": 0,
            "embedding_null": 0,
            "vectorstore_saved": 0,
            "current_stage": None,
            "failed_docs": [],  # Liste des documents échoués
            "success_docs": [],  # Liste des 5 premiers documents réussis (pour exemple)
            "all_success_sources": set(),  # Nouveau: stockage de toutes les sources réussies
        }
        self._trace_data[trace_id] = initial_state
        logger.info(f"[CALLBACK][TRACE INITIALIZED] {trace_id}")

    def on_event_start(
        self, event_type: CBEventType, payload: Optional[Dict[str, Any]], **kwargs: Any
    ) -> None:
        logger.info(f"[CALLBACK][START] {event_type.name}: {payload}")

        if event_type == CBEventType.NODE_PARSING:
            docs = payload.get("documents", [])
            if docs:
                null_docs = sum(1 for doc in docs if doc is None)
                valid_docs = len(docs) - null_docs
                self._update_trace_stats("transform_input_docs", valid_docs)
                self._update_trace_stats("transform_input_null", null_docs)
                logger.info(
                    f"[CALLBACK][TRANSFORM START] Processing {valid_docs} documents "
                    f"({null_docs} null documents found)"
                )
                if null_docs > 0:
                    logger.warning(
                        f"[CALLBACK][TRANSFORM WARNING] Found {null_docs} null documents!"
                    )

        elif event_type == CBEventType.EMBEDDING:
            chunks = payload.get(EventPayload.CHUNKS, [])
            if chunks:
                # Afficher toutes les sources au début
                initial_sources = sorted(
                    list(
                        {
                            self._extract_chunk_info(chunk).get("source", "unknown")
                            for chunk in chunks
                            if chunk is not None
                        }
                    )
                )

                logger.info("=== SOURCES À TRAITER ===")
                logger.info(str(initial_sources))
                logger.info("=" * 50)

                null_chunks = sum(1 for chunk in chunks if chunk is None)
                valid_chunks = len(chunks) - null_chunks
                self._update_trace_stats("embedding_input_docs", valid_chunks)
                self._update_trace_stats("embedding_input_null", null_chunks)
                logger.info(
                    f"[CALLBACK][EMBEDDING START] Processing {valid_chunks} chunks "
                    f"({null_chunks} null chunks found)"
                )
                if null_chunks > 0:
                    logger.warning(
                        f"[CALLBACK][EMBEDDING WARNING] Found {null_chunks} null chunks!"
                    )

    def _ensure_trace_exists(self, trace_id: str) -> None:
        """Ensure that a trace exists for the given ID."""
        if trace_id not in self._trace_data:
            logger.warning(f"[CALLBACK] Initializing missing trace {trace_id}")
            self.start_trace(trace_id)

    def on_event_end(
        self, event_type: CBEventType, payload: Optional[Dict[str, Any]], **kwargs: Any
    ) -> None:
        trace_id = kwargs.get("trace_id", "default")
        self._ensure_trace_exists(trace_id)

        logger.info(f"[CALLBACK][END] {event_type.name}")

        if not payload:
            logger.error("[CALLBACK][ERROR] Empty payload")
            return

        if event_type == CBEventType.NODE_PARSING:
            nodes = payload.get("nodes", [])
            if nodes:
                self._update_trace_stats("transform_output_docs", len(nodes))
                logger.info(f"[CALLBACK][TRANSFORM END] Created {len(nodes)} nodes")

        elif event_type == CBEventType.EMBEDDING:
            embeddings = payload.get(EventPayload.EMBEDDINGS, [])
            chunks = payload.get(EventPayload.CHUNKS, [])

            if not isinstance(embeddings, list):
                logger.error(
                    f"[CALLBACK][EMBEDDING ERROR] Invalid embeddings type: {type(embeddings)}"
                )
                return

            if len(embeddings) == 1 and chunks and len(chunks) > 1:
                logger.error(
                    "[CALLBACK][EMBEDDING ERROR] Single embedding returned for multiple chunks"
                )
                logger.error(
                    f"[CALLBACK][EMBEDDING DEBUG] Number of chunks: {len(chunks)}"
                )

                # Marquer tous les chunks comme échoués
                for chunk in chunks:
                    doc_info = self._extract_chunk_info(chunk)
                    doc_info["error"] = (
                        "Invalid embedding format - single embedding for multiple chunks"
                    )
                    self._trace_data[trace_id]["failed_docs"].append(doc_info)

                self._update_trace_stats("embedding_null", len(chunks))
                return

            if embeddings and chunks:
                success_details = []
                failed_details = []
                for chunk, embedding in zip(chunks, embeddings):
                    doc_info = self._extract_chunk_info(chunk)
                    if embedding is None:
                        doc_info["error"] = "Null embedding"
                        self._trace_data[trace_id]["failed_docs"].append(doc_info)
                        failed_details.append(doc_info)
                    else:
                        # Stocker toutes les sources réussies
                        self._trace_data[trace_id]["all_success_sources"].add(
                            doc_info["source"]
                        )
                        # Garder seulement 5 exemples pour l'affichage détaillé
                        if len(self._trace_data[trace_id]["success_docs"]) < 5:
                            self._trace_data[trace_id]["success_docs"].append(doc_info)
                            success_details.append(doc_info)

                null_count = sum(1 for emb in embeddings if emb is None)
                success_count = len(embeddings) - null_count

                self._update_trace_stats("embedding_success", success_count)
                self._update_trace_stats("embedding_null", null_count)

                logger.info(
                    f"[CALLBACK][EMBEDDING END] Successful: {success_count}, Null: {null_count}"
                )

                if failed_details:
                    logger.error(
                        "[CALLBACK][EMBEDDING FAILURES] Documents with null embeddings:"
                    )
                    for doc in failed_details:
                        logger.error(
                            f"  - Source: {doc['source']}, ID: {doc['content_id']}"
                        )

                if success_details:
                    logger.info(
                        "[CALLBACK][EMBEDDING SUCCESS DETAILS] First 5 successful embeddings:"
                    )
                    for doc in success_details:
                        logger.info(
                            f"  - Source: {doc['source']}, ID: {doc['content_id']}"
                        )

                # Ajout du récapitulatif des sources après le traitement des embeddings
                stats = self._trace_data[trace_id]

                # Collecter toutes les sources uniques
                failed_sources = sorted(
                    list({doc["source"] for doc in stats["failed_docs"]})
                )
                success_sources = sorted(
                    list(
                        stats["all_success_sources"]
                    )  # Utiliser toutes les sources réussies
                )

                # Log des sources en format liste
                logger.info("=== RÉSUMÉ DES SOURCES ===")
                logger.info("SOURCES EN ÉCHEC:")
                logger.info(str(failed_sources))
                logger.info("SOURCES EN SUCCÈS:")
                logger.info(str(success_sources))
                logger.info("=" * 50)

            else:
                logger.error(
                    f"[CALLBACK][EMBEDDING DEBUG] Missing data - Embeddings: {bool(embeddings)}, Chunks: {bool(chunks)}"
                )
                logger.error(
                    f"[CALLBACK][EMBEDDING DEBUG] Payload keys: {list(payload.keys())}"
                )

    def _extract_chunk_info(self, chunk: str) -> Dict[str, str]:
        """Helper to extract metadata from a chunk."""
        metadata = chunk.split("\n")
        return {
            "source": next(
                (
                    line.split(": ")[1]
                    for line in metadata
                    if line.startswith("source: ")
                ),
                "unknown",
            ),
            "content_id": next(
                (
                    line.split(": ")[1]
                    for line in metadata
                    if line.startswith("c_contentId: ")
                ),
                "unknown",
            ),
        }

    def end_trace(self, trace_id: str) -> None:
        if trace_id in self._trace_data:
            stats = self._trace_data[trace_id]
            logger.info(f"[CALLBACK][TRACE SUMMARY] {trace_id}")
            logger.info(
                f"[CALLBACK] Transform input documents: {stats['transform_input_docs']} (Null: {stats['transform_input_null']})"
            )
            logger.info(
                f"[CALLBACK] Transform output nodes: {stats['transform_output_docs']} (Null: {stats['transform_output_null']})"
            )
            logger.info(
                f"[CALLBACK] Embedding input nodes: {stats['embedding_input_docs']} (Null: {stats['embedding_input_null']})"
            )
            logger.info(
                f"[CALLBACK] Successful embeddings: {stats['embedding_success']}"
            )
            logger.info(f"[CALLBACK] Null embeddings: {stats['embedding_null']}")

            # Ajout des logs pour les documents échoués
            if stats["failed_docs"]:
                logger.error("[CALLBACK][FAILED DOCUMENTS]")
                for doc in stats["failed_docs"]:
                    logger.error(
                        f"  - Source: {doc['source']}, ID: {doc['content_id']}"
                    )

            # Ajout des logs pour les documents réussis
            if stats["success_docs"]:
                logger.info("[CALLBACK][SUCCESSFUL DOCUMENTS (sample)]")
                for doc in stats["success_docs"]:
                    logger.info(f"  - Source: {doc['source']}, ID: {doc['content_id']}")

            if any(
                stats[k] > 0
                for k in [
                    "transform_input_null",
                    "transform_output_null",
                    "embedding_input_null",
                    "embedding_null",
                ]
            ):
                logger.error(
                    "[CALLBACK][ALERT] Null documents/nodes detected in pipeline!"
                )

            if stats["embedding_null"] > 0:
                logger.error(
                    f"[CALLBACK][ALERT] {stats['embedding_null']} documents have null embeddings!"
                )

            if stats["embedding_input_docs"] != (
                stats["embedding_success"] + stats["embedding_null"]
            ):
                logger.error("[CALLBACK][ALERT] Mismatch in embedding counts!")

            del self._trace_data[trace_id]

    def _update_trace_stats(self, key: str, value: int) -> None:
        """Helper to update trace statistics."""
        for trace_id in self._trace_data:
            self._trace_data[trace_id][key] = (
                self._trace_data[trace_id].get(key, 0) + value
            )
