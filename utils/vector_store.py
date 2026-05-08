"""
CV to JD Mapping System v2 — FAISS Vector Store
================================================
Manages the FAISS JD index: build, search, persist (local + Azure Blob).

Design decisions:
- IndexFlatIP (inner product) with L2-normalized vectors = cosine similarity
- JD-indexed approach: index JDs, query with CV text
- Azure Blob Storage used for durable index persistence
- Local fallback: works fully offline without Blob Storage
"""

import logging
import os
import pickle
import struct
import tempfile
from io import BytesIO
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FAISS Index Manager
# ---------------------------------------------------------------------------

class FAISSJDIndex:
    """
    Manages a FAISS IndexFlatIP index over JD embeddings.

    Attributes:
        index: The FAISS index object (IndexFlatIP)
        jd_metadata: List of dicts, one per indexed JD, with keys:
                     'filename', 'text', 'job_title', 'department'
        dim: Embedding dimension (3072 for text-embedding-3-large)
    """

    def __init__(self, dim: Optional[int] = None):
        import faiss

        if dim is None:
            from config.settings import settings
            dim = settings.faiss_index_dim

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # Inner product = cosine (after L2 norm)
        self.jd_metadata: List[dict] = []
        logger.info("FAISSJDIndex initialized (dim=%d, IndexFlatIP)", dim)

    # ------------------------------------------------------------------
    # Build / Add
    # ------------------------------------------------------------------

    def add_jds(
        self,
        jd_texts: List[str],
        jd_metadata_list: List[dict],
        embeddings_client=None,
    ) -> None:
        """
        Embed and add a batch of JD texts to the index.

        Args:
            jd_texts: Raw JD text strings
            jd_metadata_list: Dicts with at minimum {'filename': str, 'text': str}
            embeddings_client: LangChain embeddings instance (or None = auto)
        """
        from utils.embeddings import embed_texts

        if len(jd_texts) != len(jd_metadata_list):
            raise ValueError(
                f"jd_texts ({len(jd_texts)}) and jd_metadata_list ({len(jd_metadata_list)}) "
                "must have the same length."
            )

        logger.info("Embedding %d JDs...", len(jd_texts))
        embeddings = embed_texts(jd_texts, embeddings_client=embeddings_client)

        vectors = np.array(embeddings, dtype=np.float32)
        vectors = self._l2_normalize(vectors)
        self.index.add(vectors)
        self.jd_metadata.extend(jd_metadata_list)

        logger.info(
            "Index now contains %d JDs (added %d).",
            self.index.ntotal, len(jd_texts)
        )

    def reset(self) -> None:
        """Clear the index and metadata (used when re-indexing)."""
        import faiss
        self.index = faiss.IndexFlatIP(self.dim)
        self.jd_metadata = []
        logger.info("FAISSJDIndex reset.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_text: str,
        top_k: int = 25,
        embeddings_client=None,
    ) -> List[Tuple[float, dict]]:
        """
        Find the top_k most similar JDs for a given CV (query) text.

        Args:
            query_text: Enriched CV text to use as the search query
            top_k: Number of results to return
            embeddings_client: LangChain embeddings instance (or None = auto)

        Returns:
            List of (cosine_similarity_score, jd_metadata_dict) tuples,
            sorted descending by score.
        """
        if self.index.ntotal == 0:
            logger.warning("search() called on empty index — returning []")
            return []

        from utils.embeddings import embed_query

        query_vec = np.array(
            [embed_query(query_text, embeddings_client=embeddings_client)],
            dtype=np.float32
        )
        query_vec = self._l2_normalize(query_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for empty slots
            results.append((float(score), self.jd_metadata[idx]))

        logger.debug("search() returned %d results (top score: %.3f)", len(results), results[0][0] if results else 0)
        return results

    # ------------------------------------------------------------------
    # Persistence — Local
    # ------------------------------------------------------------------

    def save_local(self, path: str) -> None:
        """
        Save the FAISS index + metadata to disk.

        Args:
            path: Directory path. Creates two files:
                  {path}/index.faiss  — FAISS binary index
                  {path}/metadata.pkl — JD metadata list
        """
        import faiss

        os.makedirs(path, exist_ok=True)
        index_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.pkl")

        faiss.write_index(self.index, index_path)
        with open(meta_path, "wb") as f:
            pickle.dump({"jd_metadata": self.jd_metadata, "dim": self.dim}, f)

        logger.info(
            "Index saved locally to %s (%d JDs).", path, self.index.ntotal
        )

    @classmethod
    def load_local(cls, path: str) -> "FAISSJDIndex":
        """
        Load a FAISS index from disk.

        Args:
            path: Directory path with index.faiss + metadata.pkl

        Returns:
            Loaded FAISSJDIndex instance
        """
        import faiss

        index_path = os.path.join(path, "index.faiss")
        meta_path = os.path.join(path, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Index files not found at: {path}")

        raw_index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        instance = cls(dim=meta.get("dim", raw_index.d))
        instance.index = raw_index
        instance.jd_metadata = meta.get("jd_metadata", [])

        logger.info(
            "Index loaded from %s (%d JDs).", path, instance.index.ntotal
        )
        return instance

    # ------------------------------------------------------------------
    # Persistence — Azure Blob Storage
    # ------------------------------------------------------------------

    def save_to_blob(
        self,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
        blob_prefix: str = "faiss_index",
    ) -> None:
        """
        Upload the FAISS index + metadata to Azure Blob Storage.

        Uploads two blobs:
          {blob_prefix}/index.faiss
          {blob_prefix}/metadata.pkl

        Args:
            connection_string: Azure Storage connection string.
                               Falls back to settings.azure_storage_connection_string.
            container_name: Blob container name.
                            Falls back to settings.azure_storage_container.
            blob_prefix: Blob name prefix (like a folder path)
        """
        from azure.storage.blob import BlobServiceClient
        import faiss

        conn_str, container = self._resolve_blob_config(connection_string, container_name)
        client = BlobServiceClient.from_connection_string(conn_str)
        container_client = client.get_container_client(container)

        # Ensure container exists
        try:
            container_client.create_container()
            logger.info("Created blob container: %s", container)
        except Exception:
            pass  # Container already exists

        # --- Upload index.faiss ---
        with tempfile.TemporaryDirectory() as tmp:
            index_path = os.path.join(tmp, "index.faiss")
            faiss.write_index(self.index, index_path)
            with open(index_path, "rb") as f:
                blob_name = f"{blob_prefix}/index.faiss"
                container_client.upload_blob(blob_name, f, overwrite=True)
                logger.info("Uploaded %s to container '%s'", blob_name, container)

        # --- Upload metadata.pkl ---
        meta_bytes = pickle.dumps({"jd_metadata": self.jd_metadata, "dim": self.dim})
        blob_name = f"{blob_prefix}/metadata.pkl"
        container_client.upload_blob(blob_name, meta_bytes, overwrite=True)
        logger.info(
            "Uploaded %s to container '%s' (%d JDs).", blob_name, container, self.index.ntotal
        )

    @classmethod
    def load_from_blob(
        cls,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
        blob_prefix: str = "faiss_index",
    ) -> "FAISSJDIndex":
        """
        Download and load a FAISS index from Azure Blob Storage.

        Args:
            connection_string: Azure Storage connection string
            container_name: Blob container name
            blob_prefix: Blob name prefix

        Returns:
            Loaded FAISSJDIndex instance
        """
        from azure.storage.blob import BlobServiceClient
        import faiss

        conn_str, container = cls._resolve_blob_config_static(connection_string, container_name)
        client = BlobServiceClient.from_connection_string(conn_str)
        container_client = client.get_container_client(container)

        with tempfile.TemporaryDirectory() as tmp:
            # --- Download index.faiss ---
            index_path = os.path.join(tmp, "index.faiss")
            blob_client = container_client.get_blob_client(f"{blob_prefix}/index.faiss")
            with open(index_path, "wb") as f:
                f.write(blob_client.download_blob().readall())

            # --- Download metadata.pkl ---
            meta_client = container_client.get_blob_client(f"{blob_prefix}/metadata.pkl")
            meta_bytes = meta_client.download_blob().readall()
            meta = pickle.loads(meta_bytes)

            raw_index = faiss.read_index(index_path)

        instance = cls(dim=meta.get("dim", raw_index.d))
        instance.index = raw_index
        instance.jd_metadata = meta.get("jd_metadata", [])

        logger.info(
            "Index loaded from blob '%s/%s' (%d JDs).", container, blob_prefix, instance.index.ntotal
        )
        return instance

    # ------------------------------------------------------------------
    # Convenience: Auto-persist (local + optional blob)
    # ------------------------------------------------------------------

    def save(self, local_path: Optional[str] = None, upload_to_blob: bool = False) -> None:
        """
        Save index to local path and optionally upload to Azure Blob.

        Args:
            local_path: Local directory. Falls back to settings.faiss_index_path.
            upload_to_blob: If True, also upload to blob (requires blob settings).
        """
        from config.settings import settings

        path = local_path or settings.faiss_index_path
        self.save_local(path)

        if upload_to_blob:
            try:
                self.save_to_blob()
                logger.info("Index successfully uploaded to Azure Blob Storage.")
            except Exception as e:
                logger.warning(
                    "Blob upload failed (local save succeeded): %s", e
                )

    @classmethod
    def load(
        cls,
        local_path: Optional[str] = None,
        try_blob_first: bool = False,
    ) -> "FAISSJDIndex":
        """
        Load index from local disk or Azure Blob (whichever is available).

        Strategy:
          - try_blob_first=True  → try Blob, fallback to local
          - try_blob_first=False → try local first, fallback to Blob

        Args:
            local_path: Local directory path. Falls back to settings.faiss_index_path.
            try_blob_first: If True, prefer Azure Blob over local disk.

        Returns:
            Loaded FAISSJDIndex instance.

        Raises:
            FileNotFoundError: If neither source has the index.
        """
        from config.settings import settings

        path = local_path or settings.faiss_index_path

        sources = (
            [("blob", None), ("local", path)]
            if try_blob_first
            else [("local", path), ("blob", None)]
        )

        last_error: Optional[Exception] = None
        for source, src_path in sources:
            try:
                if source == "local":
                    return cls.load_local(src_path)
                else:
                    return cls.load_from_blob()
            except Exception as e:
                logger.debug("Load from %s failed: %s", source, e)
                last_error = e

        raise FileNotFoundError(
            f"Could not load FAISS index from local ({path}) or Azure Blob. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Azure Blob: CV / JD file storage helpers
    # ------------------------------------------------------------------

    @staticmethod
    def upload_file_to_blob(
        file_bytes: bytes,
        blob_name: str,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> str:
        """
        Upload a raw file (CV or JD) to Azure Blob Storage.

        Args:
            file_bytes: Raw file bytes
            blob_name: Blob name (e.g. "cvs/John_Doe_CV.pdf")
            connection_string: Azure Storage connection string
            container_name: Blob container name

        Returns:
            Blob URL string
        """
        from azure.storage.blob import BlobServiceClient

        conn_str, container = FAISSJDIndex._resolve_blob_config_static(
            connection_string, container_name
        )
        client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = client.get_blob_client(container=container, blob=blob_name)

        # Ensure container exists
        try:
            client.get_container_client(container).create_container()
        except Exception:
            pass

        blob_client.upload_blob(file_bytes, overwrite=True)
        url = blob_client.url
        logger.info("Uploaded file to blob: %s", url)
        return url

    @staticmethod
    def list_blobs_in_folder(
        folder_prefix: str,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> List[str]:
        """
        List all blob names under a given prefix (folder).

        Args:
            folder_prefix: e.g. "jds/" or "cvs/"
            connection_string: Azure Storage connection string
            container_name: Blob container name

        Returns:
            List of blob name strings
        """
        from azure.storage.blob import BlobServiceClient

        conn_str, container = FAISSJDIndex._resolve_blob_config_static(
            connection_string, container_name
        )
        client = BlobServiceClient.from_connection_string(conn_str)
        container_client = client.get_container_client(container)

        blobs = [b.name for b in container_client.list_blobs(name_starts_with=folder_prefix)]
        logger.debug("Found %d blobs under prefix '%s'", len(blobs), folder_prefix)
        return blobs

    @staticmethod
    def download_blob_bytes(
        blob_name: str,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
    ) -> bytes:
        """
        Download raw bytes from a blob.

        Args:
            blob_name: Blob name to download
            connection_string: Azure Storage connection string
            container_name: Blob container name

        Returns:
            Raw bytes of the blob content
        """
        from azure.storage.blob import BlobServiceClient

        conn_str, container = FAISSJDIndex._resolve_blob_config_static(
            connection_string, container_name
        )
        client = BlobServiceClient.from_connection_string(conn_str)
        blob_client = client.get_blob_client(container=container, blob=blob_name)
        data = blob_client.download_blob().readall()
        logger.debug("Downloaded %d bytes from blob '%s'", len(data), blob_name)
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        """L2-normalize rows of a 2D float32 array (in-place safe)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)  # avoid divide-by-zero
        return (vectors / norms).astype(np.float32)

    def _resolve_blob_config(
        self,
        connection_string: Optional[str],
        container_name: Optional[str],
    ) -> Tuple[str, str]:
        return self._resolve_blob_config_static(connection_string, container_name)

    @staticmethod
    def _resolve_blob_config_static(
        connection_string: Optional[str],
        container_name: Optional[str],
    ) -> Tuple[str, str]:
        from config.settings import settings

        conn_str = connection_string or getattr(settings, "azure_storage_connection_string", None)
        container = container_name or getattr(settings, "azure_storage_container", "cv-jd-index")

        if not conn_str:
            raise ValueError(
                "Azure Storage connection string not provided. "
                "Set AZURE_STORAGE_CONNECTION_STRING in your .env file."
            )

        return conn_str, container
