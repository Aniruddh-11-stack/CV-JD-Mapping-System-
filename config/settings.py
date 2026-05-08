"""
CV to JD Mapping System v2 — Configuration
==========================================
All settings loaded from environment variables via .env file.
No hardcoded credentials — ever.
"""

import os
from typing import Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Supports: OpenAI, Azure OpenAI, Google Gemini, Groq.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -----------------------------------------------------------------------
    # LLM Provider Selection
    # -----------------------------------------------------------------------
    llm_provider: Literal["openai", "azure_openai", "gemini", "groq"] = Field(
        default="azure_openai",
        description="Which LLM provider to use"
    )

    # -----------------------------------------------------------------------
    # OpenAI
    # -----------------------------------------------------------------------
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-ada-002")

    # -----------------------------------------------------------------------
    # Azure OpenAI  (your existing setup — just moved to env vars)
    # -----------------------------------------------------------------------
    azure_openai_endpoint: Optional[str] = Field(default=None)
    azure_openai_key: Optional[str] = Field(default=None)
    azure_openai_deployment: str = Field(default="gpt-4.1-mini")
    azure_openai_embedding_deployment: str = Field(default="text-embedding-ada-002")
    azure_openai_api_version: str = Field(default="2024-12-01-preview")
    azure_openai_embedding_api_version: str = Field(default="2023-05-15")

    # Custom SSL cert path (for corporate networks like UltraTech)
    ssl_cert_path: Optional[str] = Field(default=None)

    # -----------------------------------------------------------------------
    # Google Gemini
    # -----------------------------------------------------------------------
    google_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")

    # -----------------------------------------------------------------------
    # Groq
    # -----------------------------------------------------------------------
    groq_api_key: Optional[str] = Field(default=None)
    groq_model: str = Field(default="llama3-8b-8192")

    # -----------------------------------------------------------------------
    # Embedding fallback (local, no API key needed)
    # -----------------------------------------------------------------------
    use_local_embeddings: bool = Field(
        default=False,
        description="Use HuggingFace sentence-transformers for embeddings (no API cost)"
    )
    local_embedding_model: str = Field(default="all-MiniLM-L6-v2")

    # -----------------------------------------------------------------------
    # FAISS Index
    # -----------------------------------------------------------------------
    faiss_index_path: str = Field(default="data/faiss_jd_index")
    faiss_index_dim: int = Field(
        default=1536,
        description="Embedding dimension: 1536 for text-embedding-ada-002 or text-embedding-3-small"
    )

    # -----------------------------------------------------------------------
    # Confidence Scoring Weights  (from your confidence_score.py — now integrated)
    # -----------------------------------------------------------------------
    weight_semantic: float = Field(default=0.40, description="Cosine similarity weight")
    weight_skill: float = Field(default=0.30, description="Skill match weight")
    weight_experience: float = Field(default=0.20, description="Experience match weight")
    weight_education: float = Field(default=0.10, description="Education match weight")

    # -----------------------------------------------------------------------
    # Text Processing
    # -----------------------------------------------------------------------
    chunk_size: int = Field(default=1500)
    chunk_overlap: int = Field(default=200)
    min_text_length: int = Field(
        default=100,
        description="If extracted text < this, attempt OCR"
    )

    # -----------------------------------------------------------------------
    # Matching
    # -----------------------------------------------------------------------
    top_k_matches: int = Field(default=3, description="Top N JD matches per CV")
    retrieval_k: int = Field(
        default=25,
        description="Number of candidates retrieved before experience filtering"
    )
    gpt_temperature: float = Field(default=0.2)
    gpt_max_tokens: int = Field(default=2000)

    # -----------------------------------------------------------------------
    # Azure Blob Storage  (new — for FAISS index + CV/JD file persistence)
    # -----------------------------------------------------------------------
    azure_storage_connection_string: Optional[str] = Field(
        default=None,
        description="Azure Storage Account connection string"
    )
    azure_storage_container: str = Field(
        default="cv-jd-index",
        description="Blob container name for storing FAISS index and files"
    )
    azure_storage_cv_prefix: str = Field(default="cvs/")
    azure_storage_jd_prefix: str = Field(default="jds/")
    azure_storage_index_prefix: str = Field(default="faiss_index")

    # -----------------------------------------------------------------------
    # API Server
    # -----------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_key_header: Optional[str] = Field(
        default=None,
        description="If set, require this key in X-API-Key header"
    )

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default="logs/cv_jd_system.log")


# Singleton instance
settings = Settings()


def get_llm_client():
    """
    Factory: returns the right LLM client based on settings.llm_provider.
    Supports OpenAI, Azure OpenAI, Gemini (via OpenAI-compatible API), Groq.
    """
    import httpx

    if settings.llm_provider == "azure_openai":
        from openai import AzureOpenAI

        http_client = None
        if settings.ssl_cert_path and os.path.exists(settings.ssl_cert_path):
            http_client = httpx.Client(
                verify=settings.ssl_cert_path,
                timeout=httpx.Timeout(60.0, connect=15.0),
            )

        return AzureOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version=settings.azure_openai_api_version,
            http_client=http_client,
        )

    elif settings.llm_provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=settings.openai_api_key)

    elif settings.llm_provider == "groq":
        from openai import OpenAI  # Groq is OpenAI-compatible
        return OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )

    elif settings.llm_provider == "gemini":
        from openai import OpenAI  # Gemini is OpenAI-compatible via v1beta
        return OpenAI(
            api_key=settings.google_api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")


def get_model_name() -> str:
    """Returns the model deployment name for the current provider."""
    if settings.llm_provider == "azure_openai":
        return settings.azure_openai_deployment
    elif settings.llm_provider == "openai":
        return settings.openai_model
    elif settings.llm_provider == "groq":
        return settings.groq_model
    elif settings.llm_provider == "gemini":
        return settings.gemini_model
    return "gpt-4o-mini"


def get_embeddings_client():
    """
    Factory: returns the right LangChain embeddings client.
    Falls back to local HuggingFace if use_local_embeddings=True.
    """
    import httpx

    if settings.use_local_embeddings:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)

    if settings.llm_provider == "azure_openai":
        from langchain_openai import AzureOpenAIEmbeddings

        http_client = None
        if settings.ssl_cert_path and os.path.exists(settings.ssl_cert_path):
            http_client = httpx.Client(
                verify=settings.ssl_cert_path,
                timeout=httpx.Timeout(60.0, connect=15.0),
            )

        return AzureOpenAIEmbeddings(
            azure_deployment=settings.azure_openai_embedding_deployment,
            openai_api_version=settings.azure_openai_embedding_api_version,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            http_client=http_client,
        )

    elif settings.llm_provider in ("openai", "groq", "gemini"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    raise ValueError(f"No embeddings client for provider: {settings.llm_provider}")
