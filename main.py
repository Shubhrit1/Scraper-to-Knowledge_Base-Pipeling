"""Local RAG knowledge-base pipeline for public health-insurance policy content.

Processes only pages/PDFs you scrape and approve. Do not paste real personal data.
No external LLM/API keys are used; embeddings run locally via sentence-transformers.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import pdfplumber
import requests
from bs4 import BeautifulSoup, Comment, NavigableString
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

import chromadb

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "chroma_db"
RESULTS_PATH = ROOT / "results.json"
COLLECTION_NAME = "health_insurance_kb"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_TOKENS = 400
CHUNK_OVERLAP = 50
REQUEST_TIMEOUT = 30
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

STRIP_TAGS = {
    "nav",
    "header",
    "footer",
    "script",
    "style",
    "noscript",
    "iframe",
    "svg",
    "form",
    "aside",
    "button",
}

PII_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b\d{3}\s\d{2}\s\d{4}\b"),
    re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"
    ),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    re.compile(
        r"\b(?:date of birth|d\.?o\.?b\.?|birth(?:date| day)?)\b"
        r".{0,40}?\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b.{0,24}?"
        r"\b(?:date of birth|d\.?o\.?b\.?|birth(?:date| day)?)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(?:born(?:\s+on)?)\s+\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b",
        re.IGNORECASE,
    ),
]

HEADING_LINE = re.compile(
    r"^(?:#{1,6}\s+\S.*|"
    r"[A-Z][A-Z0-9 ,/&()'’\-:]{7,}|"
    r"(?:Section|Article|Chapter|Part|Exhibit|Appendix)\s+[\w.]+.*|"
    r"\d+(?:\.\d+){0,4}\s+[A-Z].+)$"
)

_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None


class ScrapeRequest(BaseModel):
    url: str = Field(..., min_length=8)


class ApproveRequest(BaseModel):
    title: str = Field(..., min_length=1)
    category: str = ""
    source: str = ""
    version: str = "1.0"
    pii: bool = False
    text: str = Field(..., min_length=1)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)


class SaveResultRequest(BaseModel):
    query: str
    chunk: str
    source: str
    score: float
    verdict: str
    explanation: str = ""


app = FastAPI(title="Health Insurance KB Pipeline", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def token_count(text: str) -> int:
    return len(text.split())


def detect_pii(text: str) -> bool:
    return any(p.search(text) for p in PII_PATTERNS)


def clean_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    collapsed: list[str] = []
    blank = 0
    for line in lines:
        if not line:
            blank += 1
            if blank <= 1:
                collapsed.append("")
            continue
        blank = 0
        collapsed.append(line)
    return "\n".join(collapsed).strip()


def looks_like_pdf(url: str, content_type: str) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return True
    return "application/pdf" in (content_type or "").lower()


def extract_pdf_text(data: bytes) -> str:
    pages: list[str] = []
    with pdfplumber.open(BytesIO(data)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
    raw = "\n\n".join(pages)
    lines: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and HEADING_LINE.match(stripped) and not stripped.startswith("#"):
            lines.append(f"## {stripped}")
        else:
            lines.append(stripped)
    return clean_whitespace("\n".join(lines))


def html_title(soup: BeautifulSoup, url: str) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return str(og["content"]).strip()
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(" ", strip=True)
        if t:
            return t
    path = urlparse(url).path.rstrip("/").split("/")[-1]
    return path.replace("-", " ").replace("_", " ") or url


def _attr_blob(tag: Any, key: str) -> str:
    attrs = getattr(tag, "attrs", None) or {}
    value = attrs.get(key, "")
    if isinstance(value, list):
        return " ".join(str(v) for v in value).lower()
    return str(value or "").lower()


def extract_html_text(html: str, url: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = html_title(soup, url)
    for comment in soup.find_all(string=lambda n: isinstance(n, Comment)):
        comment.extract()
    for tag in list(soup.find_all(STRIP_TAGS)):
        tag.decompose()
    chrome_keys = ("nav", "footer", "header", "cookie", "banner", "sidebar", "menu")
    to_drop = []
    for tag in soup.find_all(True):
        if tag.name in {"html", "body", "main", "article"}:
            continue
        blob = " ".join((_attr_blob(tag, "class"), _attr_blob(tag, "id"), _attr_blob(tag, "role")))
        if any(key in blob for key in chrome_keys):
            to_drop.append(tag)
    for tag in to_drop:
        tag.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    for level in range(1, 7):
        for heading in list(body.find_all(f"h{level}")):
            label = heading.get_text(" ", strip=True)
            if label:
                heading.replace_with(NavigableString(f"\n{'#' * level} {label}\n"))
    text = body.get_text("\n", strip=False)
    return title, clean_whitespace(text)


def _origin(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}/"


def _request_headers(url: str) -> dict[str, str]:
    pdf = urlparse(url).path.lower().endswith(".pdf")
    return {
        "User-Agent": USER_AGENT,
        "Accept": (
            "application/pdf,*/*;q=0.8"
            if pdf
            else "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": _origin(url),
    }


def scrape_url(url: str) -> tuple[str, str]:
    session = requests.Session()
    try:
        session.get(
            _origin(url),
            timeout=min(REQUEST_TIMEOUT, 15),
            headers=_request_headers(_origin(url)),
            allow_redirects=True,
        )
    except requests.RequestException:
        pass
    try:
        response = session.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers=_request_headers(url),
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {exc}") from exc

    if response.status_code == 404:
        raise HTTPException(
            status_code=400,
            detail=(
                f"The site returned 404 Not Found for {response.url}. "
                "That page was moved or removed — paste the current URL from the live site."
            ),
        )
    if response.status_code == 403:
        raise HTTPException(
            status_code=400,
            detail=(
                "Cloudflare blocked this URL from the scraper (HTTP 403). "
                "Open the PDF in your browser, save it, then use Load file below. "
                "Keep the original URL in the URL box so source metadata is preserved."
            ),
        )
    if response.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to fetch URL: HTTP {response.status_code} for {response.url}",
        )

    content_type = response.headers.get("Content-Type", "")
    if looks_like_pdf(url, content_type) or response.content[:5] == b"%PDF-":
        text = extract_pdf_text(response.content)
        name = urlparse(url).path.rstrip("/").split("/")[-1]
        title = name.rsplit(".", 1)[0].replace("-", " ").replace("_", " ") or name
        if not text:
            raise HTTPException(status_code=422, detail="No text extracted from PDF.")
        return title, text

    title, text = extract_html_text(response.text, url)
    if not text:
        raise HTTPException(status_code=422, detail="No readable text found on the page.")
    return title, text


def extract_from_bytes(data: bytes, filename: str, source: str) -> tuple[str, str]:
    name = filename or "upload"
    lower = name.lower()
    if lower.endswith(".pdf") or data[:5] == b"%PDF-":
        text = extract_pdf_text(data)
        title = Path(name).stem.replace("-", " ").replace("_", " ") or name
        if not text:
            raise HTTPException(status_code=422, detail="No text extracted from PDF.")
        return title, text
    title, text = extract_html_text(data.decode("utf-8", errors="replace"), source or name)
    if not text:
        raise HTTPException(status_code=422, detail="No readable text found in the file.")
    return title, text


def document_payload(title: str, text: str, source: str) -> dict[str, Any]:
    return {
        "title": title,
        "text": text,
        "source": source,
        "version": "1.0",
        "pii_suspected": detect_pii(text),
    }


def split_by_headings(text: str) -> list[str]:
    lines = text.splitlines()
    sections: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if HEADING_LINE.match(line.strip()) and current:
            sections.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append(current)
    merged = [clean_whitespace("\n".join(s)) for s in sections]
    return [s for s in merged if s]


def window_chunks(text: str, size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= size:
        return [text]
    step = max(size - overlap, 1)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += step
    return chunks


def chunk_text(text: str) -> list[str]:
    sections = split_by_headings(text)
    if len(sections) <= 1 and token_count(text) <= CHUNK_TOKENS:
        return [text.strip()] if text.strip() else []
    chunks: list[str] = []
    for section in sections:
        if token_count(section) <= CHUNK_TOKENS:
            chunks.append(section)
        else:
            chunks.extend(window_chunks(section))
    return [c for c in chunks if c.strip()]


@app.post("/api/scrape")
def api_scrape(body: ScrapeRequest) -> dict[str, Any]:
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    try:
        title, text = scrape_url(url)
    except HTTPException as exc:
        print(f"SCRAPE FAILED [{url}]: {exc.detail}")
        raise
    return document_payload(title, text, url)


@app.post("/api/ingest-file")
async def api_ingest_file(
    file: UploadFile = File(...),
    source: str = Form(""),
) -> dict[str, Any]:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    name = file.filename or "upload"
    src = (source or "").strip() or name
    title, text = extract_from_bytes(data, name, src)
    return document_payload(title, text, src)


@app.post("/api/kb/add")
def api_kb_add(body: ApproveRequest) -> dict[str, Any]:
    text = body.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Corrected text is empty.")
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from the text.")

    record_id = str(uuid.uuid4())
    model = get_model()
    embeddings = model.encode(chunks, normalize_embeddings=True).tolist()
    collection = get_collection()
    ids = [f"{record_id}:{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "record_id": record_id,
            "title": body.title.strip(),
            "category": body.category.strip(),
            "source": body.source.strip(),
            "version": body.version.strip() or "1.0",
            "pii": bool(body.pii),
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return {"record_id": record_id, "chunks_added": len(chunks)}


@app.post("/api/search")
def api_search(body: SearchRequest) -> dict[str, Any]:
    collection = get_collection()
    if collection.count() == 0:
        return {"results": []}
    model = get_model()
    query_emb = model.encode([body.query.strip()], normalize_embeddings=True).tolist()
    n = min(3, collection.count())
    raw = collection.query(
        query_embeddings=query_emb,
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        distance = float(dist)
        similarity = max(0.0, min(1.0, 1.0 - distance))
        results.append(
            {
                "chunk": doc,
                "source": meta.get("source", ""),
                "title": meta.get("title", ""),
                "score": round(similarity, 4),
                "chunk_index": meta.get("chunk_index"),
                "record_id": meta.get("record_id"),
            }
        )
    return {"results": results}


@app.post("/api/results")
def api_save_result(body: SaveResultRequest) -> dict[str, Any]:
    verdict = body.verdict.strip().lower()
    allowed = {"correct", "partially correct", "incorrect"}
    if verdict not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Verdict must be correct, partially correct, or incorrect.",
        )
    payload = {
        "query": body.query,
        "chunk": body.chunk,
        "source": body.source,
        "score": body.score,
        "verdict": verdict,
        "explanation": body.explanation,
    }
    existing: list[dict[str, Any]] = []
    if RESULTS_PATH.exists():
        try:
            existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    existing.append(payload)
    RESULTS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    return {"saved": True, "count": len(existing)}
