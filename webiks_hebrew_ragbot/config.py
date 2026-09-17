import logging
import os

EMBEDDING_INDEX = os.getenv("ES_EMBEDDING_INDEX", "embedded_index")
ES_EMBEDDING_INDEX_LENGTH = int(os.getenv("ES_EMBEDDING_INDEX_LENGTH", "1000"))
MODEL_LOCATION = os.getenv("MODEL_LOCATION", "model")

# "dense" (default, original behavior) or "hybrid" (BM25 + dense fusion via RRF)
RETRIEVAL_MODE_DENSE = "dense"
RETRIEVAL_MODE_HYBRID = "hybrid"


def warn_if_unknown_retrieval_mode(mode: str) -> None:
    """Logs a warning if mode isn't a recognized RETRIEVAL_MODE value. Engine.search_documents
    treats anything other than RETRIEVAL_MODE_HYBRID as dense, so an unrecognized value never
    crashes - but it would otherwise silently run dense search with no indication of the typo."""
    if mode not in (RETRIEVAL_MODE_DENSE, RETRIEVAL_MODE_HYBRID):
        logging.warning(
            f"Unknown RETRIEVAL_MODE={mode!r}, expected {RETRIEVAL_MODE_DENSE!r} "
            f"or {RETRIEVAL_MODE_HYBRID!r}; falling back to {RETRIEVAL_MODE_DENSE!r}."
        )


RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", RETRIEVAL_MODE_DENSE)
warn_if_unknown_retrieval_mode(RETRIEVAL_MODE)

