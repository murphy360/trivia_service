from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache
def _get_embedder() -> SentenceTransformer:
    # Loaded lazily and cached: only paid for by code paths that actually embed
    # something, and only once per process.
    return SentenceTransformer(_MODEL_NAME)


def embed(text: str) -> np.ndarray:
    vector = _get_embedder().encode(text, normalize_embeddings=True)
    return np.asarray(vector, dtype=np.float32)


def to_bytes(vector: np.ndarray) -> bytes:
    return vector.astype(np.float32).tobytes()


def from_bytes(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def is_novel(candidate_vector: np.ndarray, existing_vectors: list[np.ndarray], threshold: float) -> bool:
    """Vectors are pre-normalized (normalize_embeddings=True), so cosine similarity
    is just the dot product. Rejects the candidate if it's too close to anything
    already stored, i.e. not "significantly different" from an existing question."""
    if not existing_vectors:
        return True
    similarities = np.dot(np.stack(existing_vectors), candidate_vector)
    return bool(similarities.max() < threshold)
