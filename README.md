# CV ↔ JD Mapping System v2

> **GenAI Internship Project — UltraTech Cement**  
> An intelligent, multi-agent pipeline that matches candidate CVs to Job Descriptions using LLMs, semantic embeddings, and vector search.

---

## Overview

Traditional CV screening is slow and inconsistent. This system automates the process end-to-end — parsing CVs and JDs, building a semantic index, scoring candidates across multiple dimensions, and generating a structured hiring recommendation — all powered by Azure OpenAI and LangGraph.

---

## Architecture

```
CV (DOCX/PDF)
     │
     ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Multi-Agent Pipeline          │
│                                                     │
│  [Parse Agent] → [Retrieve Agent] → [Score Agent]  │
│                         │                          │
│                  FAISS Vector Store                 │
│               (Azure Blob persistent)               │
│                         │                          │
│                  [Report Agent]                     │
└─────────────────────────────────────────────────────┘
     │
     ▼
AnalysisReport (JSON + Streamlit UI)
```

**Key components:**

| Layer | Technology |
|---|---|
| LLM | Azure OpenAI GPT-4.1-mini |
| Embeddings | Azure OpenAI text-embedding-ada-002 (1536-dim) |
| Vector Store | FAISS (IndexFlatIP) + Azure Blob Storage |
| Orchestration | LangGraph stateful multi-agent graph |
| API | FastAPI (5 REST endpoints) |
| UI | Streamlit 3-tab interface |
| Schemas | Pydantic v2 (ParsedCV, AnalysisReport, CVJDState) |
| Deployment | Docker + docker-compose |

---

## Scoring Methodology

Each CV–JD pair is evaluated across four dimensions:

| Dimension | Weight | Method |
|---|---|---|
| Semantic similarity | 40% | Cosine similarity over Ada-002 embeddings |
| Skill overlap | 30% | Keyword matching against JD skill requirements |
| Experience fit | 20% | Years of experience vs. JD requirement |
| Education fit | 10% | Degree-level matching |

**Verdict thresholds:**
- ≥ 85% → Strongly Recommended
- ≥ 70% → Recommended
- ≥ 50% → Conditionally Recommended
- < 50% → Not Recommended

---

## Features

- **JD Indexing tab** — upload JDs (DOCX/PDF), embed and store in FAISS
- **CV Matching tab** — upload a CV, retrieve top-K JD matches, generate AI analysis report
- **Analytics tab** — view scoring breakdowns, skill gap analysis, match history
- **FastAPI backend** — REST API for programmatic access
- **Azure Blob persistence** — FAISS index survives restarts
- **Docker-ready** — single `docker-compose up` to deploy

---

## Project Structure

```
cv_jd_v2/
├── agents/                  # LangGraph agent nodes
│   ├── parse_agent.py       # CV/JD text extraction & structuring
│   ├── retrieve_agent.py    # FAISS vector retrieval
│   ├── score_agent.py       # Multi-dimensional scoring
│   └── report_agent.py      # Hiring recommendation generation
├── api/
│   └── main.py              # FastAPI application (5 endpoints)
├── ui/
│   └── app.py               # Streamlit 3-tab interface
├── utils/
│   ├── azure_client.py      # Azure OpenAI wrapper
│   ├── vector_store.py      # FAISS + Azure Blob management
│   └── schemas.py           # Pydantic v2 data models
├── data/
│   └── test_docs/           # Sample CVs and JDs for testing
├── tests/
│   ├── demo_results.py      # API-free TF-IDF demo (no Azure key needed)
│   └── run_pipeline_test.py # Full LangGraph pipeline test
├── .env.example             # Environment variable template
├── docker-compose.yml       # Container orchestration
├── run.sh                   # One-command startup script
└── requirements.txt         # Python dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Azure OpenAI resource with:
  - `gpt-4.1-mini` deployment
  - `text-embedding-ada-002` deployment
- Azure Blob Storage account

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/Aniruddh-11-stack/CV-JD-Mapping-System-.git
cd CV-JD-Mapping-System-

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure credentials
cp .env.example .env
# Edit .env with your Azure OpenAI and Blob Storage credentials

# 5. Launch the app
bash run.sh
```

The Streamlit UI will open at **http://localhost:8501**  
The FastAPI backend runs at **http://localhost:8000**

### Run Demo (no Azure key required)

```bash
python tests/demo_results.py
```

This runs the full pipeline using TF-IDF similarity — same output structure as production, no API calls needed.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/index-jd` | Upload and index a JD |
| `POST` | `/match-cv` | Match a CV against indexed JDs |
| `GET` | `/results/{cv_id}` | Retrieve analysis report |
| `GET` | `/jds` | List all indexed JDs |
| `DELETE` | `/jds/{jd_id}` | Remove a JD from the index |

---

## Sample Results

Test run: 3 CVs × 3 JDs

| Candidate | Best Match | Score | Verdict |
|---|---|---|---|
| Arjun Mehta (ML Engineer) | JD ML Engineer | 75% | ✅ Recommended |
| Neha Patel (Data Analyst) | JD Senior Data Analyst | 84% | ✅ Recommended |
| Priya Sharma (Data Scientist) | JD Senior Data Scientist | 77% | ✅ Recommended |

---

## Tech Stack

`Python 3.10` · `LangGraph` · `LangChain` · `Azure OpenAI` · `FAISS` · `Streamlit` · `FastAPI` · `Pydantic v2` · `Docker` · `Azure Blob Storage`

---

## Author

**Aniruddh** — GenAI Intern, UltraTech Cement  
[anikulks@gmail.com](mailto:anikulks@gmail.com)
