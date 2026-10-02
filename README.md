# Health Insurance RAG Knowledge Base

Local single-page app for ingesting **public** health-insurance policy pages and PDFs, correcting them, storing chunks in ChromaDB, and testing retrieval. No API keys. Nothing is preloaded — the KB only contains what you approve.

Do not paste real personal data. The PII checkbox is a **flag only** (SSN / phone / email / DOB-like patterns). It does not redact.

## Stack

- FastAPI backend (`main.py`)
- Plain HTML/JS frontend (`static/index.html`)
- ChromaDB at `./chroma_db`
- Local embeddings: `sentence-transformers` (`all-MiniLM-L6-v2`)
- HTML scrape: `requests` + BeautifulSoup
- PDF text: `pdfplumber`

## Run

From this folder, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**

The first `pip install` and the first **Approve** (model download) need network. After that, embeddings and the vector store stay local.

Stop the server with **Ctrl+C**. Leave the venv with `deactivate`.

## Workflow

```
URL or saved file
        ↓
   Scrape / Load file  →  cleaned text
        ↓
   Correct metadata + body  (title, category, source, version, PII flag)
        ↓
   Approve & Add to KB  →  chunk → embed → ChromaDB
        ↓
   Test Retrieval  →  top-3 chunks + score
        ↓
   Verdict + explanation  →  append results.json
```

### 1. Ingest

- Paste a **current** `http://` or `https://` URL and click **Scrape**.
- HTML: nav/header/footer/script/style are stripped; a suggested title is returned.
- PDF URLs (path ends in `.pdf`): text is extracted with pdfplumber.
- If the site blocks the scraper (**404** moved page, **403** Cloudflare/bot check), open the page/PDF in a browser, save it, keep the URL in the box for source metadata, then **Load file**.

### 2. Correct

Edit title, category, source (defaults to the URL), version (default `1.0`), the cleaned textarea, and the PII checkbox. Fix wording yourself; nothing is auto-redacted.

### 3. Approve & Add to KB

Text is split by headings when possible, otherwise ~400-token windows with 50-token overlap. Each chunk is embedded and stored with:

`record_id`, `title`, `category`, `source`, `version`, `pii`, `chunk_index`

### 4. Test retrieval

Enter a query → **Search** returns the top 3 chunks with source and similarity. For each hit, set verdict (`correct` / `partially correct` / `incorrect`), write an explanation, **Save Result**. Rows append to `results.json`:

`{query, chunk, source, score, verdict, explanation}`

## Reset the knowledge base

Stop uvicorn, then:

```powershell
Remove-Item -Recurse -Force .\chroma_db
# optional evaluation log:
Remove-Item -Force .\results.json
```

Restart the app. `.venv` is the Python environment, not the KB.

## Common scrape errors

| What you see | Meaning | What to do |
| --- | --- | --- |
| 404 Not Found | That path is gone | Use the live URL from the site |
| HTTP 403 / Cloudflare | Site blocked the scraper | Save in the browser → **Load file** |

Uvicorn `CancelledError` / `KeyboardInterrupt` during `--reload` after a file save is normal. Wait for `Application startup complete`.
