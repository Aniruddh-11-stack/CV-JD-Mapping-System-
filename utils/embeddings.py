"""
CV to JD Mapping System v2 — Embedding Utilities
=================================================
Handles embedding generation for CV and JD text.
Wraps LangChain embeddings with batching, retry logic, and caching.
"""

import logging
import time
import hashlib
import json
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory embedding cache (keyed by SHA256 of text)
# ---------------------------------------------------------------------------
_embedding_cache: dict[str, List[float]] = {}


def _text_hash(text: str) -> str:
    """Return a stable SHA256 hash for a text string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed_texts(
    texts: List[str],
    embeddings_client=None,
    batch_size: int = 16,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    use_cache: bool = True,
) -> List[List[float]]:
    """
    Embed a list of texts using the provided embeddings client.
    Handles batching, retry with exponential backoff, and in-memory caching.

    Args:
        texts: List of text strings to embed
        embeddings_client: LangChain-compatible embeddings instance.
                           If None, fetched from config.settings.
        batch_size: Number of texts per API call (avoids token limit errors)
        max_retries: Number of retry attempts on API failure
        retry_delay: Base delay in seconds (doubles on each retry)
        use_cache: If True, cache embeddings by text hash to avoid re-embedding

    Returns:
        List of embedding vectors (same order as input texts)
    """
    if embeddings_client is None:
        from config.settings import get_embeddings_client
        embeddings_client = get_embeddings_client()

    all_embeddings: List[Optional[List[float]]] = [None] * len(texts)
    uncached_indices: List[int] = []
    uncached_texts: List[str] = []

    # --- Check cache ---
    if use_cache:
        for i, text in enumerate(texts):
            key = _text_hash(text)
            if key in _embedding_cache:
                all_embeddings[i] = _embedding_cache[key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)
    else:
        uncached_indices = list(range(len(texts)))
        uncached_texts = texts

    if not uncached_texts:
        logger.debug("All %d embeddings served from cache.", len(texts))
        return all_embeddings  # type: ignore[return-value]

    logger.info(
        "Embedding %d texts (%d cached, %d new) in batches of %d",
        len(texts), len(texts) - len(uncached_texts), len(uncached_texts), batch_size
    )

    # --- Batch + retry ---
    new_embeddings: List[List[float]] = []
    for batch_start in range(0, len(uncached_texts), batch_size):
        batch = uncached_texts[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1

        for attempt in range(1, max_retries + 1):
            try:
                result = embeddings_client.embed_documents(batch)
                new_embeddings.extend(result)
                logger.debug("Batch %d embedded successfully (%d texts).", batch_num, len(batch))
                break
            except Exception as e:
                wait = retry_delay * (2 ** (attempt - 1))
                if attempt < max_retries:
                    logger.warning(
                        "Embedding batch %d failed (attempt %d/%d): %s. Retrying in %.1fs...",
                        batch_num, attempt, max_retries, e, wait
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        "Embedding batch %d failed after %d attempts: %s",
                        batch_num, max_retries, e
                    )
                    raise

    # --- Write back to cache and output list ---
    for idx, text, embedding in zip(uncached_indices, uncached_texts, new_embeddings):
        if use_cache:
            _embedding_cache[_text_hash(text)] = embedding
        all_embeddings[idx] = embedding

    return all_embeddings  # type: ignore[return-value]


def embed_single(
    text: str,
    embeddings_client=None,
    use_cache: bool = True,
) -> List[float]:
    """
    Embed a single text string.
    Wrapper around embed_texts for convenience.

    Args:
        text: Text to embed
        embeddings_client: LangChain-compatible embeddings instance
        use_cache: Use in-memory cache

    Returns:
        Embedding vector as a list of floats
    """
    results = embed_texts([text], embeddings_client=embeddings_client, use_cache=use_cache)
    return results[0]


def embed_query(text: str, embeddings_client=None) -> List[float]:
    """
    Embed a query string for similarity search.
    Uses embed_query() instead of embed_documents() — some providers
    apply different preprocessing for search queries vs. document bodies.

    Args:
        text: Query text (e.g. enriched CV text)
        embeddings_client: LangChain-compatible embeddings instance

    Returns:
        Query embedding vector
    """
    if embeddings_client is None:
        from config.settings import get_embeddings_client
        embeddings_client = get_embeddings_client()

    key = "query::" + _text_hash(text)
    if key in _embedding_cache:
        return _embedding_cache[key]

    for attempt in range(1, 4):
        try:
            result = embeddings_client.embed_query(text)
            _embedding_cache[key] = result
            return result
        except Exception as e:
            if attempt < 3:
                time.sleep(2.0 * attempt)
            else:
                logger.error("embed_query failed after 3 attempts: %s", e)
                raise


def get_embedding_dim(embeddings_client=None) -> int:
    """
    Probe the embedding dimension by embedding a short test string.
    Used to validate FAISS index dimensions at startup.

    Returns:
        Integer dimension of the embedding vector
    """
    try:
        test_vec = embed_single("test", embeddings_client=embeddings_client, use_cache=False)
        return len(test_vec)
    except Exception as e:
        logger.warning("Could not probe embedding dim: %s. Falling back to settings.", e)
        from config.settings import settings
        return settings.faiss_index_dim


def clear_embedding_cache() -> None:
    """Clear the in-memory embedding cache. Useful for testing."""
    global _embedding_cache
    _embedding_cache = {}
    logger.info("Embedding cache cleared.")


def save_embedding_cache(path: str) -> None:
    """
    Persist the in-memory embedding cache to a JSON file.
    Allows warm-starting embeddings across app restarts without an API call.

    Args:
        path: File path to write the cache JSON
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    with open(path, "w") as f:
        json.dump(_embedding_cache, f)
    logger.info("Embedding cache saved to %s (%d entries).", path, len(_embedding_cache))


def load_embedding_cache(path: str) -> int:
    """
    Load a previously saved embedding cache from a JSON file.

    Args:
        path: File path to read the cache JSON from

    Returns:
        Number of entries loaded
    """
    global _embedding_cache
    if not os.path.exists(path):
        logger.warning("Embedding cache file not found: %s", path)
        return 0
    try:
        with open(path) as f:
            loaded = json.load(f)
        _embedding_cache.update(loaded)
        logger.info("Embedding cache loaded from %s (%d entries).", path, len(loaded))
        return len(loaded)
    except Exception as e:
        logger.error("Failed to load embedding cache from %s: %s", path, e)
        return 0
