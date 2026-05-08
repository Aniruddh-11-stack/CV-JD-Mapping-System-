#!/usr/bin/env bash
# ============================================================
# CV to JD Mapping System v2 — Setup & Run Script
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"

echo ""
echo "========================================"
echo "  CV to JD Mapping System v2"
echo "========================================"
echo ""

# ── 1. Create virtual environment if needed ──────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "  ✅ Virtual environment created at .venv/"
else
    echo "→ Virtual environment already exists, skipping creation."
fi

# ── 2. Activate venv ─────────────────────────────────────────
echo "→ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# ── 3. Upgrade pip silently ──────────────────────────────────
echo "→ Upgrading pip..."
pip install --upgrade pip -q

# ── 4. Install requirements ──────────────────────────────────
echo "→ Installing requirements (this may take 2–3 minutes on first run)..."
pip install -r "$PROJECT_DIR/requirements.txt" -q
echo "  ✅ All packages installed."

# ── 5. Create necessary local directories ────────────────────
mkdir -p "$PROJECT_DIR/data"
mkdir -p "$PROJECT_DIR/logs"
echo "  ✅ data/ and logs/ directories ready."

# ── 6. Verify .env exists ────────────────────────────────────
if [ ! -f "$PROJECT_DIR/.env" ]; then
    echo ""
    echo "  ⚠️  No .env file found! Copy .env.example to .env and fill in your credentials."
    exit 1
else
    echo "  ✅ .env file found."
fi

# ── 7. Launch Streamlit ──────────────────────────────────────
echo ""
echo "→ Starting Streamlit app..."
echo "  Open your browser at: http://localhost:8501"
echo ""
cd "$PROJECT_DIR"
streamlit run ui/app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
