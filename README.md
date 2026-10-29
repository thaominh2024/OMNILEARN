# PDF-to-Story Generation — AI Pipeline Portfolio

A focused extraction of the **PDF-to-Story** functionality from the collaborative **OMILEARNING1** project. This is a portfolio-oriented subset, not a claim of sole authorship of the original application. Adjust the personal-contributions section below to match actual authorship before publishing.

## What the code currently does

```text
PDF document
  -> PyMuPDF text extraction
  -> character/word chunking
  -> SentenceTransformer embeddings (optional ONNX encoder)
  -> FAISS index saved to disk
  -> MMR representative excerpt selection
  -> LLM-assisted learning unit/topic proposal
  -> separate FastAPI story endpoint receives ONE unit title
  -> LLM-generated Vietnamese short story
```

**Important limitation:** The original story endpoint receives only `unit` and `style`. It does **not** retrieve or receive relevant PDF passages. Therefore the generated story is *topic-based*, not source-grounded RAG. The FAISS index is saved but is not queried by the current story service.

## Repository structure

```text
PDF-to-Story-Generation/
├── src/
│   ├── pipeline.py          # Original PDF -> learning units pipeline
│   └── story/app.py         # Original story FastAPI service (credential removed)
├── run_demo.py              # Standalone CLI coordinator added for portfolio
├── docs/architecture.md    # Detailed data flow and limitations
├── examples/README.md      # Instructions for non-copyrighted demo inputs
├── requirements.txt
├── .env.example
└── .gitignore
```

## Installation

Python 3.10+ recommended (actual environment compatibility depends on installed ML packages).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in your terminal environment. `.env.example` is a template, **not auto-loaded** by these scripts. Do not commit real keys. If the key is absent, topic generation falls back to heuristic titles and the story API returns clearly marked FAKE demo text.

## Quickstart

First terminal — story service:

```bash
uvicorn src.story.app:app --host 127.0.0.1 --port 8006
```

Second terminal — create topics and stories:

```bash
python run_demo.py --pdf path/to/your.pdf --out-dir outputs/demo --generate-stories
```

To generate topics without launching the story server:

```bash
python run_demo.py --pdf path/to/your.pdf --out-dir outputs/demo
```

The initial embedding model may download model weights. The code stores artifacts in `outputs/` (ignored by Git). No user PDFs are bundled.

## API

`POST http://127.0.0.1:8006/story`

```json
{"unit": "Gradient Descent", "style": "ngắn gọn, dễ hiểu"}
```

Response: `{"text": "..."}`. Health check: `GET /health`.

## My contribution / Attribution

**Original project:** [OMILEARNING1](https://github.com/thaominh2024/OMILEARNING1) (collaborative project).

**Personal scope:** PDF extraction -> learning unit generation -> story generation integration **[EDIT THIS TO MATCH YOUR ACTUAL WORK]**. The source files in `src/` are copied from the team repository with removal of a hard-coded credential. `run_demo.py`, documentation and packaging were added for this standalone presentation. Credit other contributors and confirm permission/license before publishing any team-owned code.

## Caveats / Next steps

- Text-layer PDFs only; scanned/image-only PDFs need OCR.
- The saved FAISS index is not connected to story generation.
- Add per-unit retrieval and citations for grounded stories.
- Add end-to-end integration tests and sample PDFs with reuse rights.
- Confirm package versions in your actual deployment environment.
- Rotate/revoke any API key previously exposed in a public repository or shared ZIP; removing it from this subset does not invalidate it.

## Validation status

Files were checked for Python syntax and packaged. A full run of PyMuPDF, model downloads, OpenAI requests, and the service integration was **not executed** during packaging.
