# Documentation

Technical reference for the local health-insurance RAG knowledge-base pipeline. For a short operator guide, see [README.md](README.md).

The app ingests **public policy HTML and PDFs** that you scrape or upload, lets you correct them, stores embeddings locally, and tests retrieval. It does **not** generate answers with an LLM, does **not** seed insurance content, and does **not** need API keys.

Do not paste real personal data. PII detection is a **flag only**.

---

## 1. Tech stack

| Layer | Choice | Role |
| --- | --- | --- |
| API | FastAPI 0.115 + Uvicorn | HTTP server, JSON/multipart endpoints |
| UI | Plain HTML, CSS, JS (`static/index.html`) | Single page, no bundler |
| HTTP fetch | `requests` | Download remote pages/PDFs |
| HTML parse | BeautifulSoup4 | Strip chrome, extract title and body |
| PDF parse | pdfplumber | Extract text from PDF bytes |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Local 384-dim vectors |
| Vector store | ChromaDB (persistent `./chroma_db`) | Cosine HNSW collection `health_insurance_kb` |
| Uploads | python-multipart | `/api/ingest-file` |
| Eval log | `results.json` | Append-only retrieval labels |

Pinned versions live in `requirements.txt`. Hugging Face model weights download on first **Approve** or **Search** (cached under the user profile, not this repo).

**Not in the stack:** OpenAI/Anthropic, LangChain, React/Vite, databases other than Chroma files, authentication, OpenAPI UI (`docs_url=None`).

---

## 2. Repository layout

```
.
├── main.py                 FastAPI app, scrape/chunk/embed/search
├── static/index.html       UI + inline CSS/JS
├── requirements.txt
├── README.md               Run + workflow
├── DOCUMENTATION.md        This file
├── .gitignore
├── chroma_db/              Created at runtime (gitignored)
├── results.json            Created on first Save Result (gitignored)
└── .venv/                  Local Python env (gitignored)
```

---

## 3. What is supported

### Ingest

- HTTP(S) URLs for HTML pages
- HTTP(S) URLs that are PDFs (path ends in `.pdf`, `Content-Type` is PDF, or body starts with `%PDF-`)
- Local **Load file**: `.pdf`, `.html`, `.htm`
- Browser-like User-Agent, `Referer` from site origin, optional homepage warmup (helps some hotlink rules; **does not** pass Cloudflare JS challenges)
- Redirects (30s timeout)
- HTML chrome removal: `nav`, `header`, `footer`, `script`, `style`, `noscript`, `iframe`, `svg`, `form`, `aside`, `button`, plus class/id/role containing `nav`, `footer`, `header`, `cookie`, `banner`, `sidebar`, `menu`
- Headings `h1`–`h6` rewritten as Markdown `#` lines so later chunking can split on them
- PDF heading-like ALL-CAPS / section lines prefixed with `##`
- Suggested title from `og:title`, `<title>`, first `h1`, or filename/path
- Explicit errors: 404 (moved URL), 403 (blocked — use Load file), empty extract (422)

### Correction (human in the loop)

- Editable **title**, **category**, **source**, **version** (default `1.0`)
- Full-body textarea of cleaned text
- **PII checkbox** auto-checked if patterns match; you can uncheck or check manually
- Patterns: formatted SSN, US-style phone, email, DOB-adjacent dates (`date of birth` / `DOB` / `born on`)
- **No auto-redaction**

### Knowledge base

- Chunk by headings when possible; else ~**400** whitespace-tokens with **50**-token overlap
- One UUID `record_id` per approval; chunk ids `{record_id}:{chunk_index}`
- Metadata on every chunk: `record_id`, `title`, `category`, `source`, `version`, `pii` (bool), `chunk_index`
- Cosine space; query embeddings normalized; score shown as `1 - distance` clipped to `[0, 1]`
- Persistence in `./chroma_db` across restarts

### Retrieval evaluation

- Top **3** chunks (or fewer if the collection is smaller)
- Each hit: source, title, score, chunk text
- Verdicts: `correct` | `partially correct` | `incorrect`
- Free-text explanation
- **Save Result** appends to `results.json`

### Operations

- Serve UI at `GET /`
- `--reload` during development
- Wipe KB by deleting `chroma_db/`
- Chroma anonymous telemetry disabled via `ANONYMIZED_TELEMETRY=False`

---

## 4. What is not supported

- No generative RAG (no “answer this question” LLM)
- No JavaScript-rendered SPA content (only static HTML from `requests`)
- No Cloudflare/bot-challenge bypass
- No authenticated / paywalled pages
- No delete, update, or list-all UI for existing records
- No OCR for scanned image-only PDFs (pdfplumber text layer only)
- No hardcoded health-insurance corpus
- No transmission of data to third-party LLM APIs

---

## 5. End-to-end workflow

```
                    ┌─────────────┐
                    │  URL Scrape │──403/404──► save in browser
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Load file  │  (PDF / HTML)
                    └──────┬──────┘
                           │
              title, text, source, pii_suspected
                           │
                    ┌──────▼──────┐
                    │   Correct   │  human edits
                    └──────┬──────┘
                           │
                    Approve & Add to KB
                           │
              chunk → MiniLM → Chroma add
                           │
                    ┌──────▼──────┐
                    │   Search    │  top-3 + scores
                    └──────┬──────┘
                           │
                    Save Result → results.json
```

---

## 6. HTTP API

Base URL: `http://127.0.0.1:8000`

| Method | Path | Body | Success |
| --- | --- | --- | --- |
| GET | `/` | — | `index.html` |
| POST | `/api/scrape` | JSON `{ "url": "https://..." }` | `{ title, text, source, version, pii_suspected }` |
| POST | `/api/ingest-file` | multipart `file`, optional form `source` | same payload shape |
| POST | `/api/kb/add` | JSON `{ title, category, source, version, pii, text }` | `{ record_id, chunks_added }` |
| POST | `/api/search` | JSON `{ "query": "..." }` | `{ results: [{ chunk, source, title, score, chunk_index, record_id }] }` |
| POST | `/api/results` | JSON `{ query, chunk, source, score, verdict, explanation }` | `{ saved, count }` |

Typical error: HTTP **400** with `{ "detail": "..." }`. Empty PDF/HTML extract: **422**. Failed scrapes also print `SCRAPE FAILED [url]: ...` in the Uvicorn terminal.

---

## 7. Code walkthrough (`main.py`)

### Constants and globals

- `CHROMA_DIR`, `RESULTS_PATH`, `COLLECTION_NAME`, `EMBED_MODEL_NAME`
- `CHUNK_TOKENS = 400`, `CHUNK_OVERLAP = 50`, `REQUEST_TIMEOUT = 30`
- `_model` / `_collection` lazy singletons (`get_model`, `get_collection`)

### Pydantic models

- `ScrapeRequest` — URL (min length 8)
- `ApproveRequest` — fields written into Chroma metadata plus body text
- `SearchRequest` — query string
- `SaveResultRequest` — evaluation row

### Cleaning helpers

- `clean_whitespace` — collapse runs of blank lines
- `detect_pii` — any `PII_PATTERNS` hit → `True`
- `looks_like_pdf` — path or content-type
- `extract_pdf_text` — page text + heading heuristics
- `html_title` / `extract_html_text` / `_attr_blob` — HTML pipeline
- `_origin` / `_request_headers` — fetch headers
- `scrape_url` — GET origin (best-effort), GET target, branch PDF vs HTML, map status codes
- `extract_from_bytes` — same extractors for uploads
- `document_payload` — shared scrape/upload response

### Chunking

- `HEADING_LINE` — Markdown headings, ALL-CAPS lines, Section/Article/…, numbered `1.2 Title`
- `split_by_headings` — sections that start at those lines
- `window_chunks` — sliding word windows
- `chunk_text` — heading split first; windows only if a section is over 400 tokens

Tokens here are **whitespace-separated words**, not a subword tokenizer.

### Persistence and search

- `api_kb_add` — encode chunks with `normalize_embeddings=True`, `collection.add`
- `api_search` — encode query, `collection.query`, convert cosine **distance** to similarity
- `api_save_result` — read JSON list (or start empty), append, write UTF-8

### App wiring

- `app.mount("/static", ...)`
- `GET /` → `FileResponse` of the SPA
- Routes listed in section 6

---

## 8. Frontend walkthrough (`static/index.html`)

One file, no build step.

1. **Scrape source** — URL + Scrape → `POST /api/scrape`. File picker + Load file → `POST /api/ingest-file` (`FormData` with optional source from the URL box).
2. **Correct & approve** — hidden until ingest succeeds. `applyDoc` fills title/source/version/body/PII. Approve → `POST /api/kb/add`.
3. **Test retrieval** — Search → `POST /api/search`. Each result: verdict select, explanation textarea, Save Result → `POST /api/results`.

`readError` unwraps FastAPI `detail` whether it is a string or a validation list.

---

## 9. Data on disk

### Chroma (`chroma_db/`)

Collection `health_insurance_kb`, cosine space. Each vector is one chunk document with the metadata in section 3.

### `results.json`

Array of objects:

```json
{
  "query": "...",
  "chunk": "...",
  "source": "...",
  "score": 0.0,
  "verdict": "correct",
  "explanation": "..."
}
```

---

## 10. Run, stop, reset

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — **Ctrl+C** stops the server — `deactivate` leaves the venv.

Uvicorn `CancelledError` on `--reload` after editing `main.py` is expected.

Reset store (server stopped):

```powershell
Remove-Item -Recurse -Force .\chroma_db
Remove-Item -Force .\results.json
```

`.gitignore` excludes `.venv/`, `chroma_db/`, `results.json`, and Python caches.

---

## 11. Scrape failure map

| Symptom | Cause | Action |
| --- | --- | --- |
| 404 in UI | Remote path removed | Use the live URL |
| 403 / Cloudflare | Bot/JS challenge | Browser save → Load file |
| Network error | DNS, TLS, timeout | Check URL and connectivity |
| No readable text | Empty HTML/PDF text layer | Different source or OCR outside this app |
| `GET /json/version` 404 | Browser/devtools probe | Ignore; not part of the app |
