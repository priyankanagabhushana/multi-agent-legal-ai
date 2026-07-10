# SovereignLex — Explainable Multi-Agent Legal Reasoning Workbench

**A transparent, modular multi-agent reasoning architecture for Swiss Tenancy Law that improves traceability, verifiability, and reliability of legal AI while preserving full digital sovereignty.**

## Core Research Question

> Can a transparent multi-agent reasoning architecture with explicit task decomposition, evidence mapping, verification steps, uncertainty estimation, and counterargument generation improve the traceability and reliability of legal AI while preserving digital sovereignty?

## Architecture

```
User Claim
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │Coordinator│─▶│  Fact    │─▶│  Legal   │               │
│  │ (Plan)   │  │Extractor │  │Retriever │               │
│  └──────────┘  └──────────┘  └──────────┘               │
│                                    │                     │
│       ┌────────────────────────────┘                     │
│       ▼                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │ Evidence │─▶│ Verifier │─▶│Synthesizer│              │
│  │ Mapping  │  │(Audit)   │  │(Report)  │              │
│  └──────────┘  └──────────┘  └──────────┘              │
│                                                          │
│  Full Trace Log ────▶ Structured Report                  │
└──────────────────────────────────────────────────────────┘
```

### Agents (5 specialized + 1 coordinator)

| Agent | Role | Output |
|-------|------|--------|
| **Coordinator** | Decomposes claim into analysis plan | `AnalysisPlan` with legal issues, key statutes, steps |
| **FactExtractor** | Extracts structured facts, dates, parties, timeline | `ExtractedFacts` with parties, dates, events |
| **LegalRetriever** | Searches OpenCaseLaw for real cases & statutes | `RetrievalResult` with BGE cases, OR statutes |
| **EvidenceMapping** | Maps facts to legal elements, identifies gaps | `EvidenceMapping` with gaps, supporting/contradicting evidence |
| **Verifier** | Audits reasoning, checks citations, flags issues | `VerificationReport` with findings, confidence adjustment |
| **Synthesizer** | Generates final report with counterarguments | `SynthesisReport` with executive summary, confidence, gaps |

### Key Features

- **Full Reasoning Trace**: Every step logged with agent name, confidence, duration, input/output summaries
- **Evidence Gaps**: Deterministic + LLM-generated gap detection — identifies missing documents, dates, facts
- **Counterargument Generation**: Devil's advocate perspective from the opposing party
- **Confidence Scoring**: Weighted confidence with breakdown (fact quality, legal basis, precedent alignment)
- **Real Case Retrieval**: OpenCaseLaw API integration — 121 courts, real BGE/BGER decisions
- **Statute Lookup**: Direct OR article text via OpenCaseLaw (`/api/laws/OR?article=271`)
- **Citation Verification**: `/api/attest` endpoint for cross-checking citations against corpus
- **100% Local Processing**: Ollama (llama3.1:8b) for sovereign inference, no cloud LLM dependency
- **Pydantic Contracts**: Every agent input/output is a strictly typed Pydantic model
- **Deterministic Gap Detection**: Fallback gap generation when LLM output is incomplete

## Data Sources

- **OpenCaseLaw REST API** (29 endpoints, free, no API key): `https://opencaselaw.ch/api/`
  - 191k BGER decisions, 35k BGE decisions, 121 courts
  - Statute lookup (OR, ZGB, StGB), citation graph, doctrine
- **HuggingFace**: `voilaj/swiss-caselaw` — 996k cases (available for batch processing)

## Quick Start

```bash
# Setup
cd LegalAI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Option A: DeepSeek API (recommended, no GPU needed)
export DEEPSEEK_API_KEY="your_key_here"
PYTHONPATH=. python -m sovereignlex.cli -e 0 --provider deepseek --model deepseek/deepseek-chat

# Option B: Local Ollama (sovereign, no API key)
ollama pull llama3.1:8b
PYTHONPATH=. python -m sovereignlex.cli -e 0

# CLI: Analyze custom text
PYTHONPATH=. python -m sovereignlex.cli "Your claim text here..."

# CLI: Save full JSON output
PYTHONPATH=. python -m sovereignlex.cli -e 0 -o result.json

# Streamlit UI (auto-detects provider from env or sidebar selection)
PYTHONPATH=. streamlit run app.py

# Evaluation harness
PYTHONPATH=. python -m sovereignlex.eval_benchmark --cases 5
```

## Deployment

### Streamlit Community Cloud (Free — Recommended)
1. Push to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → select repo `priyankanagabhushana/multi-agent-legal-ai`
4. Main file path: `app.py`
5. Set secret: `DEEPSEEK_API_KEY` = your key (optional, falls back to Ollama)
6. Deploy — free, no credit card needed

### GCP Cloud Run (Near-Free)
```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT/sovereignlex
gcloud run deploy sovereignlex \
  --image gcr.io/YOUR_PROJECT/sovereignlex \
  --port 8080 \
  --set-env-vars DEEPSEEK_API_KEY=sk-... \
  --allow-unauthenticated
```
First 2M requests/month free — negligible costs for demo use.

## Example Cases

5 realistic Swiss tenancy disputes included in `examples/tenancy_cases.py`:
1. Termination Without Official Form (Art. 266l OR)
2. Rent Increase After Renovation (Art. 269 OR)
3. Defective Heating in Winter (Art. 259a ff. OR)
4. Retaliatory Termination (Art. 271a OR)
5. Deposit Not Returned (Art. 257e OR)

## Evaluation

Run `python -m sovereignlex.eval_benchmark` to measure:
- Citation correctness (are retrieved cases real?)
- Fact extraction accuracy (are expected legal issues/statutes found?)
- Pipeline completeness (do all agents complete?)
- Counterargument and evidence gap generation

See `evaluation_notes.md` for detailed results.

## Technical Stack

- **Python 3.12** + Pydantic v2 for all data contracts
- **Ollama** with llama3.1:8b for local LLM inference
- **httpx** for OpenCaseLaw API calls with caching
- **Streamlit** for interactive UI with reasoning trace visualization
- **Rich** for CLI output formatting

## Research Positioning

This workbench is **not a chatbot**. It is a demonstration of a broader methodology:
- **Task decomposition** into specialized, auditable agents
- **Transparency** through complete reasoning traces
- **Verifiability** through real citation lookup and attestation
- **Sovereignty** through fully local processing
- **Uncertainty awareness** through confidence scoring and evidence gap identification
- **Adversarial reasoning** through counterargument generation

The Swiss Tenancy Law domain serves as a concrete, bounded demonstration. The architecture generalizes to any domain requiring transparent, trustworthy expert-assistance systems.

## Disclaimer

⚠️ **Research Prototype.** This is a decision-support tool for exploring precedents and reasoning transparency. It is **NOT legal advice**. All conclusions must be reviewed by a qualified lawyer. Outputs may contain errors.

## Project Structure

```
LegalAI/
├── sovereignlex/
│   ├── __init__.py          # Package metadata
│   ├── models.py            # Pydantic models (all I/O contracts)
│   ├── agents.py            # 6 agent implementations
│   ├── orchestrator.py      # Pipeline runner with trace logging
│   ├── api_client.py        # OpenCaseLaw REST API wrapper
│   ├── llm.py               # Ollama LLM interface
│   ├── cli.py               # CLI interface
│   ├── app.py               # Streamlit UI
│   └── eval_benchmark.py    # Evaluation harness
├── examples/
│   └── tenancy_cases.py     # 5 example tenancy disputes
├── tests/
├── requirements.txt
├── README.md
├── evaluation_notes.md
└── PROPOSAL.md
```
