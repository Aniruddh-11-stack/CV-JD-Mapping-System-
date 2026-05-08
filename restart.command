#!/usr/bin/env bash
# Kill any existing streamlit process and restart cleanly

# Skip Streamlit's email prompt by writing credentials
mkdir -p "$HOME/.streamlit"
cat > "$HOME/.streamlit/credentials.toml" << 'EOF'
[general]
email = ""
EOF

# Kill existing streamlit on port 8501
pkill -f "streamlit run" 2>/dev/null || true
sleep 1

# Now relaunch
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/.venv/bin/activate"

echo ""
echo "→ Restarting Streamlit app (email prompt skipped)..."
echo "  Open your browser at: http://localhost:8501"
echo ""

cd "$PROJECT_DIR"
streamlit run ui/app.py \
    --server.port 8501 \
    --server.headless false \
    --browser.gatherUsageStats false
