#!/usr/bin/env bash
# ============================================================
# CV ↔ JD Mapping System v2 — GitHub Push Script
# ============================================================
# Double-click this file to push the project to GitHub.
# You'll be prompted for your GitHub repo URL.
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  CV ↔ JD Mapping System v2 — GitHub Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# ── Clean up any broken .git from previous attempts ──────────────────────────
if [ -d ".git" ]; then
    echo "→ Removing previous .git directory..."
    rm -rf .git
    echo "  ✅ Cleared."
fi

# ── Init fresh repo ───────────────────────────────────────────────────────────
echo "→ Initialising git repository..."
git init
git config user.email "anikulks@gmail.com"
git config user.name "Aniruddh"

# ── Stage all files (respects .gitignore — .env will be excluded) ─────────────
echo "→ Staging files (credentials excluded via .gitignore)..."
git add -A
echo ""
echo "  Files to be committed:"
git status --short
echo ""

# ── Verify .env is NOT staged ────────────────────────────────────────────────
if git diff --cached --name-only | grep -q "^\.env$"; then
    echo "  ❌  ERROR: .env is staged! Aborting to protect your credentials."
    echo "      Check .gitignore and try again."
    exit 1
fi
echo "  ✅ .env is excluded (credentials safe)."
echo ""

# ── Commit ────────────────────────────────────────────────────────────────────
echo "→ Creating initial commit..."
git commit -m "Initial commit: CV ↔ JD Mapping System v2

GenAI internship project — UltraTech Cement

Architecture:
- LangGraph multi-agent pipeline (parse → retrieve → score → report)
- Azure OpenAI GPT-4.1-mini + text-embedding-ada-002
- FAISS vector store with Azure Blob Storage persistence
- Streamlit 3-tab UI (JD Indexing / CV Matching / Analytics)
- FastAPI REST backend with 5 endpoints
- Pydantic v2 type-safe schemas (ParsedCV, AnalysisReport, CVJDState)
- Docker + docker-compose deployment ready

Test results (3 CVs × 3 JDs):
- Arjun Mehta (ML Eng)  → JD ML Engineer        75%  Recommended
- Neha Patel  (Analyst) → JD Senior Data Analyst 84%  Recommended
- Priya Sharma (DS)     → JD Senior Data Scientist 77% Recommended"

echo ""
echo "  ✅ Committed."
echo ""

# ── GitHub remote ─────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════"
echo "  GITHUB SETUP"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  1. Go to https://github.com/new"
echo "  2. Create a repo named: cv-jd-mapping-v2  (private or public)"
echo "  3. Do NOT initialise with README (keep it empty)"
echo "  4. Copy the repo URL (e.g. https://github.com/anikulks/cv-jd-mapping-v2.git)"
echo ""
read -p "  Paste your GitHub repo URL here: " REPO_URL

if [ -z "$REPO_URL" ]; then
    echo "  ⚠️  No URL entered. Skipping push."
    echo "  You can push manually later:"
    echo "    cd \"$PROJECT_DIR\""
    echo "    git remote add origin <your-repo-url>"
    echo "    git push -u origin main"
    exit 0
fi

# ── Push ──────────────────────────────────────────────────────────────────────
echo ""
echo "→ Adding remote origin: $REPO_URL"
git remote add origin "$REPO_URL"

echo "→ Pushing to GitHub..."
BRANCH=$(git rev-parse --abbrev-ref HEAD)
git push -u origin "$BRANCH"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ PROJECT PUSHED TO GITHUB!"
echo "  🔗 $REPO_URL"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "  Next steps:"
echo "  1. Add your real Azure credentials to .env (never commit it)"
echo "  2. Update the Azure OpenAI API key if the current one has expired"
echo "  3. Run: bash run.sh  to launch the Streamlit app"
echo ""
