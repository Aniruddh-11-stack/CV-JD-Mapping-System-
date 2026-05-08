"""
CV to JD Mapping System v2 — Text Extraction Utilities
======================================================
Handles PDF, DOCX, TXT extraction with OCR fallback.
Works with both Streamlit uploaded files and raw file paths.
"""

import logging
import os
import tempfile
from io import BytesIO
from typing import Optional, Union

logger = logging.getLogger(__name__)


def extract_text_from_bytes(file_bytes: bytes, filename: str, min_length: int = 100) -> Optional[str]:
    """
    Extract text from file bytes. Supports PDF, DOCX, TXT.
    Falls back to OCR for scanned PDFs.

    Args:
        file_bytes: Raw file bytes
        filename: Original filename (used to detect extension)
        min_length: If extracted text < this, attempt OCR

    Returns:
        Extracted text string, or None on failure
    """
    ext = os.path.splitext(filename)[-1].lower()

    try:
        if ext == ".pdf":
            return _extract_pdf(file_bytes, filename, min_length)
        elif ext == ".docx":
            return _extract_docx(file_bytes)
        elif ext == ".txt":
            return file_bytes.decode("utf-8", errors="replace")
        else:
            logger.warning(f"Unsupported file type: {ext} for file {filename}")
            return None
    except Exception as e:
        logger.error(f"Text extraction failed for {filename}: {e}")
        return None


def extract_text_from_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Streamlit-compatible wrapper.
    Reads bytes from an UploadedFile and extracts text.

    NOTE: Does NOT use @st.cache_data — caching at this level caused
    the critical bug where all CVs returned identical metadata.
    Caching is handled at a higher level with proper cache keys.
    """
    try:
        file_bytes = uploaded_file.getvalue()
        return extract_text_from_bytes(file_bytes, uploaded_file.name)
    except Exception as e:
        logger.error(f"Failed to read uploaded file {uploaded_file.name}: {e}")
        return None


def extract_text_from_path(file_path: str, min_length: int = 100) -> Optional[str]:
    """Extract text from a file path on disk."""
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        return extract_text_from_bytes(file_bytes, os.path.basename(file_path), min_length)
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pdf(file_bytes: bytes, filename: str, min_length: int) -> Optional[str]:
    """Extract text from PDF bytes, with OCR fallback for scanned PDFs."""
    import fitz  # PyMuPDF

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = " ".join(page.get_text() for page in doc)

        if len(text.strip()) >= min_length:
            return text.strip()

        # --- OCR fallback for scanned PDFs ---
        logger.info(f"Low text in {filename} ({len(text.strip())} chars). Attempting OCR...")
        return _ocr_pdf(file_bytes, filename)

    except Exception as e:
        logger.error(f"PyMuPDF failed for {filename}: {e}")
        return _ocr_pdf(file_bytes, filename)


def _ocr_pdf(file_bytes: bytes, filename: str) -> Optional[str]:
    """
    OCR a PDF using ocrmypdf (preferred) or pytesseract+pdf2image fallback.
    """
    # Strategy 1: ocrmypdf (better quality)
    try:
        import ocrmypdf
        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, filename)
            output_path = os.path.join(tmp, "ocr_" + filename)
            with open(input_path, "wb") as f:
                f.write(file_bytes)
            ocrmypdf.ocr(
                input_path, output_path,
                language="eng",
                rotate_pages=True,
                deskew=True,
                force_ocr=True,
                progress_bar=False,
            )
            import fitz
            with fitz.open(output_path) as doc:
                text = " ".join(page.get_text() for page in doc)
            logger.info(f"OCR (ocrmypdf) successful for {filename}: {len(text)} chars")
            return text.strip() if text.strip() else None
    except ImportError:
        logger.debug("ocrmypdf not installed, trying pytesseract fallback")
    except Exception as e:
        logger.warning(f"ocrmypdf failed for {filename}: {e}. Trying pytesseract...")

    # Strategy 2: pytesseract + pdf2image
    try:
        import pytesseract
        from pdf2image import convert_from_bytes

        images = convert_from_bytes(file_bytes)
        ocr_text = "\n".join(pytesseract.image_to_string(img) for img in images)
        logger.info(f"OCR (pytesseract) successful for {filename}: {len(ocr_text)} chars")
        return ocr_text.strip() if ocr_text.strip() else None
    except ImportError:
        logger.error("Neither ocrmypdf nor pytesseract+pdf2image is installed. Cannot OCR.")
        return None
    except Exception as e:
        logger.error(f"pytesseract OCR also failed for {filename}: {e}")
        return None


def _extract_docx(file_bytes: bytes) -> Optional[str]:
    """Extract text from DOCX bytes using mammoth."""
    try:
        import mammoth
        result = mammoth.extract_raw_text(BytesIO(file_bytes))
        return result.value.strip() if result.value.strip() else None
    except Exception as e:
        logger.error(f"mammoth DOCX extraction failed: {e}")
        return None


# ---------------------------------------------------------------------------
# CV text enrichment (replaces enrich_cv_text from v1)
# ---------------------------------------------------------------------------

def enrich_cv_with_metadata(cv_text: str, parsed_metadata: dict) -> str:
    """
    Append structured metadata to raw CV text for better embedding quality.
    This improves cosine similarity matching by making skills/experience explicit.

    Args:
        cv_text: Raw CV text
        parsed_metadata: Dict with keys like 'skills', 'experience_years', 'education'

    Returns:
        Enriched text = original + structured summary
    """
    enriched_parts = []

    skills = parsed_metadata.get("skills", [])
    if skills:
        enriched_parts.append("Key Skills: " + ", ".join(skills))

    exp = parsed_metadata.get("experience_years", 0)
    if exp:
        enriched_parts.append(f"Total Experience: {exp} years")

    edu = parsed_metadata.get("education", "")
    if edu:
        enriched_parts.append(f"Education: {edu}")

    dept = parsed_metadata.get("department", "")
    if dept:
        enriched_parts.append(f"Domain/Department: {dept}")

    if enriched_parts:
        return cv_text + "\n\n[Structured Summary]\n" + "\n".join(enriched_parts)
    return cv_text
