# ============================================================
# CV to JD Mapping System v2 — Dockerfile
# ============================================================
# Multi-stage build:
#   Stage 1 (builder): install system deps + OCR tools
#   Stage 2 (runtime): copy only what's needed

# ------------------------------------
# Stage 1: Builder
# ------------------------------------
FROM python:3.11-slim AS builder

# Install system dependencies for OCR and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    ghostscript \
    ocrmypdf \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ------------------------------------
# Stage 2: Runtime
# ------------------------------------
FROM python:3.11-slim

# Copy system OCR tools from builder
COPY --from=builder /usr/bin/tesseract /usr/bin/tesseract
COPY --from=builder /usr/share/tesseract-ocr /usr/share/tesseract-ocr
COPY --from=builder /usr/bin/gs /usr/bin/gs
COPY --from=builder /usr/bin/ocrmypdf /usr/bin/ocrmypdf
COPY --from=builder /usr/lib/x86_64-linux-gnu/libpoppler*.so* /usr/lib/x86_64-linux-gnu/
COPY --from=builder /usr/bin/pdftoppm /usr/bin/pdftoppm
COPY --from=builder /usr/bin/pdfinfo /usr/bin/pdfinfo

# Copy installed Python packages
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p data logs

# Environment defaults (override via .env or docker-compose)
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    LOG_LEVEL=INFO

# Expose ports
EXPOSE 8501   # Streamlit
EXPOSE 8000   # FastAPI

# Default: run Streamlit UI
CMD ["streamlit", "run", "ui/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
