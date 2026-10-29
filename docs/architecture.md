# Architecture and source-code map

## Data flow

1. `src/pipeline.py:extract_pdf_text`: PyMuPDF reads text-bearing pages.
2. `chunk_text_chars` / `chunk_text_words`: produce page-indexed segments.
3. `STEncoder` / `ONNXEncoder`: produce normalized vectors.
4. `build_faiss`: save vector index; no retrieval consumer in story service.
5. `mmr_select`: choose representative, diverse excerpts.
6. `propose_topics_with_llm`: propose ordered learning units from excerpts.
7. `src/story/app.py:story`: separate endpoint generates one Vietnamese story using `unit` and `style` only.
8. `run_demo.py`: newly added coordinator connects pipeline output to story API; no additional retrieval or grounding.

## Input/output contracts

Pipeline: `python src/pipeline.py --pdf FILE --run_id ID --out_dir DIR` produces final JSON with `chunks_path`, `faiss_path`, `units`, `samples_used`. Without an API key, unit title generation uses heuristic fallback.

Story API: POST `/story` JSON `{"unit":"...","style":"..."}` -> `{"text":"..."}`. Without a key, produces FAKE placeholder.

## Differences from full-stack OMILEARNING1

The original project has Node.js upload orchestration and React node components. Those are intentionally excluded to keep the portfolio scope AI-focused. `run_demo.py` replaces them for a minimal local demo. The demo coordinator is not an original module of OMILEARNING1.

## Known technical limitations

- No OCR for scanned PDF.
- No factual source grounding or citation in story output.
- No production job queue/retries/observability.
- FAISS saved but not queried by story.
- Model/SDK compatibility should be tested on the target OS.
