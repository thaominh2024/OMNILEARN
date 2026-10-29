
"""
PDF → Chunk → Embed (ST / ONNX-INT8) → (optional FAISS) →
Select Diverse Samples (MMR) → LLM \"Professor SIMON\" proposes lesson topics → Units

Safe-by-default for server integration:
- Stdout prints ONE final JSON only (unless --emit_stream is set).
- Progress/logs go to stderr, or NDJSON to stdout when --emit_stream is enabled.

Requires: pymupdf, faiss-cpu, numpy, scikit-learn, sentence-transformers OR (onnxruntime + transformers)
Optional: openai (SDK v1) for LLM titles.

Example:
    python server/pipeline.py \
      --pdf /path/to/doc.pdf \
      --run_id run_001 \
      --out_dir /tmp/run_001 \
      --engine onnx --int8_path /models/minilm.int8.onnx --max_len 160 --batch 512 \
      --chunk_mode words --target_words 220 --overlap_words 40 --min_words 40 \
      --sample_max 180 --mmr_lambda 0.5 \
      --title_model gpt-4o-mini --title_lang vi --title_max_len 60
"""

import argparse, json, os, re, sys, math, time
from typing import Dict, List, Optional

import fitz  # PyMuPDF
import faiss
import numpy as np

# Flags
ST_AVAILABLE = False
ORT_AVAILABLE = False

# -------------------- logging / events --------------------
EMIT_ENABLED = False  # enable via --emit_stream to print NDJSON progress to stdout

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    sys.stderr.write("[{}] {}\n".format(ts, str(msg)))
    sys.stderr.flush()

def emit(event, **data):
    if not EMIT_ENABLED:
        return
    try:
        payload = {"event": event}
        payload.update(data)
        print(json.dumps(payload, ensure_ascii=False))
        sys.stdout.flush()
    except Exception:
        pass

# -------------------- text utils --------------------
def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

# -------------------- PDF extract --------------------
def extract_pdf_text(pdf_path: str) -> List[Dict]:
    log("Open PDF: {}".format(pdf_path))
    doc = fitz.open(pdf_path)
    pages: List[Dict] = []
    for i in range(len(doc)):
        text = clean_text(doc[i].get_text("text", sort=True))
        if text:
            pages.append({"page_idx": i, "text": text})
    doc.close()
    log("Extracted pages: {}".format(len(pages)))
    emit("progress", stage="extract", percent=100, pages=len(pages))
    return pages

# -------------------- Chunking --------------------
def chunk_text_chars(pages: List[Dict], target_chars=1200, overlap=150, min_words=40) -> List[Dict]:
    log("Chunking[chars] target={}, overlap={}, min_words={}".format(target_chars, overlap, min_words))
    chunks: List[Dict] = []
    for page in pages:
        words = clean_text(page["text"]).split()
        buf: List[str] = []
        cur_len = 0
        for w in words:
            wlen = len(w) + 1
            if cur_len + wlen <= target_chars:
                buf.append(w); cur_len += wlen
            else:
                if len(buf) >= min_words:
                    t = " ".join(buf)
                    chunks.append({"text": t, "page_idx": page["page_idx"]})
                    # overlap ~ convert char overlap to words
                    avg = max(5, round(cur_len / max(1, len(buf))))
                    keep = max(0, min(len(buf), overlap // avg))
                    buf = buf[-keep:]
                    cur_len = sum(len(x) + 1 for x in buf)
                buf.append(w); cur_len += wlen
        if len(buf) >= min_words:
            t = " ".join(buf)
            chunks.append({"text": t, "page_idx": page["page_idx"]})
    log("Chunks created: {}".format(len(chunks)))
    emit("progress", stage="chunk", percent=100, chunks=len(chunks))
    return chunks

def chunk_text_words(pages: List[Dict], target_words=220, overlap_words=40, min_words=40) -> List[Dict]:
    log("Chunking[words] target={}, overlap_words={}, min_words={}".format(target_words, overlap_words, min_words))
    chunks: List[Dict] = []
    for page in pages:
        words = clean_text(page["text"]).split()
        buf: List[str] = []
        for w in words:
            buf.append(w)
            if len(buf) >= target_words:
                if len(buf) >= min_words:
                    chunks.append({"text": " ".join(buf), "page_idx": page["page_idx"]})
                buf = buf[-overlap_words:]
        if len(buf) >= min_words:
            chunks.append({"text": " ".join(buf), "page_idx": page["page_idx"]})
    log("Chunks created: {}".format(len(chunks)))
    emit("progress", stage="chunk", percent=100, chunks=len(chunks))
    return chunks

# -------------------- Embedding backends --------------------
class STEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: Optional[str] = None, batch_size: int = 64):
        global ST_AVAILABLE
        try:
            from sentence_transformers import SentenceTransformer  # noqa: F401
            ST_AVAILABLE = True
        except Exception as e:
            raise RuntimeError("SentenceTransformer not available: {}".format(e))
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch = batch_size
        self.dim = int(self.model.get_sentence_embedding_dimension())
        log("Loaded ST model '{}' on {}, dim={}, batch={}".format(model_name, self.device, self.dim, self.batch))

    def encode_all(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        log("Embedding {} chunks via ST (batch={})".format(n, self.batch))
        out = np.zeros((n, self.dim), dtype="float32")
        for start in range(0, n, self.batch):
            end = min(start + self.batch, n)
            pct = math.floor((end / n) * 100)
            log("Embedding progress: {}%  (chunks {}–{}/{})".format(pct, start+1, end, n))
            emit("progress", stage="embed", percent=pct, start=start+1, end=end, total=n)
            vecs = self.model.encode(
                texts[start:end],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            out[start:end] = vecs.astype(np.float32)
        log("Embedding done (ST).")
        return out

class ONNXEncoder:
    def __init__(self, onnx_path: str, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 max_len: int = 160, batch_size: int = 512):
        global ORT_AVAILABLE
        try:
            import onnxruntime as ort  # noqa: F401
            from transformers import AutoTokenizer  # noqa: F401
            ORT_AVAILABLE = True
        except Exception as e:
            raise RuntimeError("onnxruntime/transformers not available: {}".format(e))
        import onnxruntime as ort
        from transformers import AutoTokenizer

        assert os.path.exists(onnx_path), "ONNX not found at {}".format(onnx_path)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = max(1, (os.cpu_count() or 2) // 2)
        so.inter_op_num_threads = 1

        self.sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.batch = batch_size
        self.max_len = max_len

        try:
            enc = self.tok(["test"], padding=True, truncation=True, max_length=self.max_len, return_tensors="np")
            last_hidden = self.sess.run(["last_hidden_state"], {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"]
            })[0]
            self.dim = int(last_hidden.shape[-1])
        except Exception:
            self.dim = 384
        log("Loaded ONNX INT8 '{}' (tok='{}'), dim={}, batch={}, max_len={}".format(onnx_path, tokenizer_name, self.dim, self.batch, self.max_len))

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="np")
        last_hidden = self.sess.run(["last_hidden_state"], {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"]
        })[0]
        mask = enc["attention_mask"][..., None]
        vec = (last_hidden * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
        vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-9)
        return vec.astype(np.float32)

    def encode_all(self, texts: List[str]) -> np.ndarray:
        n = len(texts)
        log("Embedding {} chunks via ONNX-INT8 (batch={}, max_len={})".format(n, self.batch, self.max_len))
        out = np.zeros((n, self.dim), dtype="float32")
        for start in range(0, n, self.batch):
            end = min(start + self.batch, n)
            pct = math.floor((end / n) * 100)
            log("Embedding progress: {}%  (chunks {}–{}/{})".format(pct, start+1, end, n))
            emit("progress", stage="embed", percent=pct, start=start+1, end=end, total=n)
            out[start:end] = self._encode_batch(texts[start:end])
        log("Embedding done (ONNX).")
        return out

# -------------------- Vector & FAISS --------------------
def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)

def build_faiss(vecs: np.ndarray, kind: str = "flat", nlist: int = 4096, nprobe: int = 16, hnsw_m: int = 32) -> faiss.Index:
    dim = int(vecs.shape[1])
    log("Building FAISS[{}] dim={}, vectors={}".format(kind, dim, vecs.shape[0]))
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m)
        try:
            faiss.ParameterSpace().set_index_parameter(index, "efSearch", 64)
        except Exception:
            pass
    else:
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        rs = np.random.RandomState(42)
        sample = vecs[rs.choice(len(vecs), size=min(100_000, len(vecs)), replace=False)]
        index.train(sample)
        index.nprobe = nprobe
    index.add(vecs)
    log("FAISS add() done.")
    emit("progress", stage="faiss", percent=100, vectors=int(vecs.shape[0]))
    return index

# -------------------- Diverse sampling (MMR) --------------------
def mmr_select(texts: List[str], vecs: np.ndarray, k: int = 200, lam: float = 0.5, max_chars: int = 320) -> List[Dict]:
    """Select k diverse, representative samples using Maximal Marginal Relevance."""
    n = len(texts)
    if n == 0:
        return []
    k = max(1, min(k, n))
    # centroid as pseudo-query
    centroid = l2_normalize(vecs.mean(axis=0, keepdims=True))[0]
    sims_to_query = (vecs @ centroid.reshape(-1,))  # cosine (vecs normalized)
    selected: List[int] = [int(np.argmax(sims_to_query))]
    candidates = set(range(n)) - set(selected)

    for _ in range(1, k):
        best_i, best_score = None, -1e9
        for i in candidates:
            rel = sims_to_query[i]
            div_pen = 0.0 if not selected else max(float(vecs[i] @ vecs[j]) for j in selected)
            score = lam * rel - (1 - lam) * div_pen
            if score > best_score:
                best_i, best_score = i, score
        if best_i is None:
            break
        selected.append(best_i)
        candidates.remove(best_i)

    out: List[Dict] = []
    for i in selected:
        t = texts[i].strip()
        if len(t) > max_chars:
            t = t[:max_chars].rstrip() + "..."
        out.append({"id": i, "text": t})
    return out

# -------------------- LLM topic generation (Professor SIMON) --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
client = None
try:
    if OPENAI_API_KEY:
        from openai import OpenAI  # type: ignore
        client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    log("OpenAI SDK not available: {}".format(e))

def _is_meaningless_title(s: str) -> bool:
    if not s or not s.strip():
        return True
    t = s.strip()
    if len(t.replace("-", "").replace("_", "").replace(" ", "")) == 0:
        return True
    if t.lower() in {"overview", "introduction", "summary", "general", "misc"}:
        return True
    return False

def propose_topics_with_llm(
    samples: List[Dict],
    lang: str = "vi",
    model: str = "gpt-4o-mini",
    max_len: int = 60,
    unit_limit: int = 12,
    temperature: float = 0.2,
) -> List[str]:
    """Return a list of topic titles (length ≤ unit_limit)."""
    if not client:
        log("OPENAI_API_KEY absent → heuristic titles from samples")
        titles: List[str] = []
        for s in samples[:unit_limit]:
            t = s.get("text", "")
            cand = re.split(r"[.?!]\s+", t, 1)[0].strip() or t[:max_len]
            if len(cand) > max_len: cand = cand[:max_len].rstrip() + "..."
            if not _is_meaningless_title(cand): titles.append(cand)
        return titles

    lang_str = "Vietnamese" if lang == "vi" else "English"
    sys_prompt = (
    f"""You are Professor SIMON, an education expert specializing in chunking materials into lesson units.
Given a set of short excerpts sampled across a document, propose up to {unit_limit} lesson topics that best cover the material.

Rules:
1) Output short, specific, content-bearing titles (noun phrases); no quotes, no emojis, no boilerplate.
2) Max length: {max_len} characters. Prefer precise domain terms and proper nouns.
3) Cover the whole document at a reasonable granularity (not too coarse, not too fine).
4) If an excerpt is noise (headers/footers, page furniture, figure/table captions, page numbers, dates) or off-topic, ignore it.
5) Keep consistent granularity across topics; avoid vague words like "Overview"/"Introduction" unless essential.
6) Language for titles: {lang_str}.
7) Style: concise noun phrase, Title Case if English; if Vietnamese, keep natural casing and diacritics.
8) Đánh số thứ tự cho mỗi topic , giống như thứ tự bài học cho dễ tiếp cận.
9) Sắp xếp các topic theo một trình tự logic: khái niệm cơ bản trước, nội dung chi tiết và ứng dụng sau, nâng cao cuối cùng.
Return a JSON object: {{"topics": ["1_title1", "2_title2", ...]}} only. No explanations or extra fields."""
)

    prompt_user = "Here are representative excerpts (id, text). Propose topics as instructed.\n\n" + \
                  json.dumps(samples, ensure_ascii=False, indent=2)

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_user},
            ],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        data = json.loads(raw)
        arr = data.get("topics", []) if isinstance(data, dict) else []
        titles: List[str] = []
        for v in arr:
            title = (v or "").strip()
            if not title or _is_meaningless_title(title):
                continue
            if len(title) > max_len:
                title = title[:max_len].rstrip() + "..."
            titles.append(title)
        return titles[:unit_limit]
    except Exception as e:
        log("LLM topics error: {}".format(e))
        # fallback
        titles: List[str] = []
        for s in samples[:unit_limit]:
            t = s.get("text", "")
            cand = re.split(r"[.?!]\s+", t, 1)[0].strip() or t[:max_len]
            if len(cand) > max_len: cand = cand[:max_len].rstrip() + "..."
            if not _is_meaningless_title(cand): titles.append(cand)
        return titles

# -------------------- Units fallback --------------------
def make_units(chunks: List[Dict], limit=12, max_len=60) -> List[str]:
    units: List[str] = []
    for i, c in enumerate(chunks[:limit]):
        t = c["text"] if isinstance(c, dict) else str(c)
        cand = next((ln.strip() for ln in t.split("\n") if 5 <= len(ln.strip()) <= max_len), None)
        if not cand: cand = re.split(r"[.?!]\s+", t, 1)[0].strip()
        if not cand: cand = "Topic {}".format(i + 1)
        if len(cand) > max_len: cand = cand[:max_len].rstrip() + "..."
        units.append(cand)
    return units

# -------------------- Save helpers --------------------
def save_chunks(chunks: List[Dict], out_dir: str, as_jsonl: bool = False) -> str:
    if as_jsonl:
        p = os.path.join(out_dir, "chunks.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for i, c in enumerate(chunks):
                rec = {"id": i}; rec.update(c)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    else:
        p = os.path.join(out_dir, "chunks.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump([c["text"] for c in chunks], f, ensure_ascii=False, indent=2)
    return p

# -------------------- main --------------------
def main(args):
    global EMIT_ENABLED
    EMIT_ENABLED = bool(args.emit_stream)

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    log("RunID={} → out_dir={}".format(args.run_id, args.out_dir))
    log("ENGINE={}".format(args.engine))

    # 1) Extract & Chunk
    pages = extract_pdf_text(args.pdf)
    if not pages:
        log("No text extracted from PDF.")
        print(json.dumps({"error": "Empty PDF"}, ensure_ascii=False))
        return

    if args.chunk_mode == "words":
        chunks = chunk_text_words(pages, target_words=args.target_words, overlap_words=args.overlap_words, min_words=args.min_words)
    else:
        chunks = chunk_text_chars(pages, target_chars=args.target_chars, overlap=args.overlap_chars, min_words=args.min_words)

    if not chunks:
        log("Fallback: using pages as chunks (chunker returned 0).")
        chunks = pages

    # 2) Embedding
    t1 = time.time()
    texts_for_embed = [c["text"] if isinstance(c, dict) else str(c) for c in chunks]
    if args.engine == "onnx":
        enc = ONNXEncoder(onnx_path=args.int8_path, tokenizer_name=args.tokenizer, max_len=args.max_len, batch_size=args.batch)
    else:
        enc = STEncoder(model_name=args.st_model, device=args.device, batch_size=args.batch)
    vecs = enc.encode_all(texts_for_embed)

    # 3) FAISS (optional kinds)
    index = build_faiss(vecs, kind=args.faiss, nlist=args.nlist, nprobe=args.nprobe, hnsw_m=args.hnsw_m)

    # 4) Diverse sampling → LLM proposes topics
    samples = mmr_select(texts_for_embed, vecs, k=args.sample_max, lam=args.mmr_lambda, max_chars=args.sample_max_chars)
    log("Selected {} samples for LLM.".format(len(samples)))

    titles = propose_topics_with_llm(
        samples,
        lang=args.title_lang,
        model=args.title_model,
        max_len=args.title_max_len,
        unit_limit=args.unit_limit,
    )
    log("Proposed {} topics via LLM.".format(len(titles)))

    # 5) Save artifacts
    chunks_path = save_chunks(chunks, args.out_dir, as_jsonl=args.chunks_jsonl)
    faiss_path = os.path.join(args.out_dir, "index.faiss")
    faiss.write_index(index, faiss_path)
    log("Saved artifacts: chunks → {}, index → {}".format(chunks_path, faiss_path))
    emit("artifact", kind="chunks", path=chunks_path)
    emit("artifact", kind="faiss", path=faiss_path)

    if args.save_vecs:
        vec_path = os.path.join(args.out_dir, "embeddings.float16.npy")
        np.save(vec_path, vecs.astype(np.float16))
        log("Saved embeddings: {}".format(vec_path))
        emit("artifact", kind="embeddings", path=vec_path)

    # 6) Units output
    units: List[str] = titles if titles else make_units(chunks, limit=args.unit_limit, max_len=args.unit_max_len)
    log("Units generated: {}".format(len(units)))

    # 7) Done
    t2 = time.time()
    log("Total time: {:.2f}s (embedding: {:.2f}s)".format(t2 - t0, t2 - t1))
    emit("progress", stage="done", percent=100)

    print(json.dumps({
        "chunks_path": chunks_path,
        "faiss_path": faiss_path,
        "units": units,
        "samples_used": len(samples),
    }, ensure_ascii=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Required I/O
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_dir", required=True)

    # Engine switch
    ap.add_argument("--engine", choices=["st", "onnx"], default="st",
                    help="st=SentenceTransformer (PyTorch); onnx=INT8 ONNXRuntime")
    # ST options
    ap.add_argument("--st_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None, help="cuda/cpu (auto if None)")

    # ONNX options
    ap.add_argument("--int8_path", default=os.environ.get("INT8_PATH", ""),
                    help="Path to quantized INT8 ONNX model")
    ap.add_argument("--tokenizer", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max_len", type=int, default=160)

    # Shared embedding options
    ap.add_argument("--batch", type=int, default=64)

    # Chunking options
    ap.add_argument("--chunk_mode", choices=["chars", "words"], default="chars")
    ap.add_argument("--target_chars", type=int, default=1200)
    ap.add_argument("--overlap_chars", type=int, default=150)
    ap.add_argument("--target_words", type=int, default=220)
    ap.add_argument("--overlap_words", type=int, default=40)
    ap.add_argument("--min_words", type=int, default=40)

    # FAISS index options
    ap.add_argument("--faiss", choices=["flat", "hnsw", "ivf"], default="flat")
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=16)
    ap.add_argument("--hnsw_m", type=int, default=32)

    # Diverse sampling for LLM
    ap.add_argument("--sample_max", type=int, default=180, help="Max diverse samples for LLM (MMR)")
    ap.add_argument("--mmr_lambda", type=float, default=0.5, help="MMR lambda: relevance vs. diversity [0..1]")
    ap.add_argument("--sample_max_chars", type=int, default=320, help="Max chars per sample text sent to LLM")

    # LLM topic generation
    ap.add_argument("--title_model", default="gpt-4o-mini")
    ap.add_argument("--title_lang", choices=["vi", "en"], default="vi")
    ap.add_argument("--title_max_len", type=int, default=60)

    # Units
    ap.add_argument("--unit_limit", type=int, default=12)
    ap.add_argument("--unit_max_len", type=int, default=60)

    # Artefacts
    ap.add_argument("--chunks_jsonl", action="store_true")
    ap.add_argument("--save_vecs", action="store_true")

    # Emission (progress NDJSON)
    ap.add_argument("--emit_stream", action="store_true", help="Print NDJSON progress/events to stdout; leave off if server parses stdout once as a single JSON")

    args = ap.parse_args()

    if args.engine == "onnx":
        if not args.int8_path or not os.path.exists(args.int8_path):
            raise SystemExit("INT8 ONNX model not found: {} (use --int8_path)".format(args.int8_path))

    try:
        main(args)
    except Exception as e:
        log("FATAL: {}".format(e))
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
