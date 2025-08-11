# mcp_starter.py
"""
Production-ready FastMCP server (single-file) for Puch AI with A4F LLM integration.

Tools:
 - validate
 - search_research_papers
 - fetch_latest_research_news
 - summarize_paper_url (can call A4F LLM via llm_summarize_hook)
 - write_cold_email
 - generate_citation
 - a4f_llm  <-- new: direct LLM tool (messages / prompt)

Environment variables:
 - AUTH_TOKEN (required)
 - MY_NUMBER  (required)
 - HOST (default 0.0.0.0)
 - PORT (default 8086)
 - REDIS_URL (optional)
 - A4F_API_KEY (your 3rd-party provider key)
 - A4F_BASE_URL (e.g. https://api.a4f.co/v1)
 - A4F_MODEL (e.g. provider-6/gpt-4.1-mini)
"""
import os
import asyncio
import logging
import re
from datetime import datetime
from typing import Annotated, List, Dict, Any
import uuid
from typing import Annotated
from pydantic import Field
from datetime import datetime, timezone, timedelta
import json
import httpx
import feedparser
from dotenv import load_dotenv
from pydantic import Field, AnyUrl

# fastmcp / MCP imports
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import TextContent, ImageContent, INVALID_PARAMS, INTERNAL_ERROR

# Optional libs
try:
    import readabilipy
    import markdownify
except Exception:
    readabilipy = None
    markdownify = None

try:
    import fitz  # type: ignore # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    try:
        import PyMuPDF as fitz  # type: ignore
        HAS_PYMUPDF = True
    except ImportError:
        try:
            from PyMuPDF import fitz  # type: ignore
            HAS_PYMUPDF = True
        except ImportError:
            HAS_PYMUPDF = False

# Optional Redis cache
REDIS_URL = os.getenv("REDIS_URL")
redis_client = None
if REDIS_URL:
    try:
        import aioredis  # type: ignore
        redis_client = aioredis.from_url(REDIS_URL)
    except Exception:
        redis_client = None

# Load env
load_dotenv()

AUTH_TOKEN = os.getenv("AUTH_TOKEN")
MY_NUMBER = os.getenv("MY_NUMBER")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8086))
USER_AGENT = "PuchResearchCopilot/1.0 (+https://example.com)"

# A4F (OpenAI-compatible) settings
A4F_API_KEY = os.getenv("A4F_API_KEY")
A4F_BASE_URL = os.getenv("A4F_BASE_URL")
A4F_MODEL = os.getenv("A4F_MODEL")

if not AUTH_TOKEN or not MY_NUMBER:
    raise RuntimeError("Please set AUTH_TOKEN and MY_NUMBER in environment or .env file")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("research_mcp")

# httpx client (singleton)
_httpx_client: httpx.AsyncClient | None = None


def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=60)
    return _httpx_client


# --- Auth provider compatible with FastMCP and Puch ---
class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self._token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self._token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None


# Create FastMCP instance
mcp = FastMCP("Research MCP (Puch-ready)", auth=SimpleBearerAuthProvider(AUTH_TOKEN))


# -----------------------
# Utilities
# -----------------------
async def fetch_text_from_url(url: str, force_raw: bool = False) -> tuple[str, str]:
    client = get_httpx_client()
    try:
        resp = await client.get(url, follow_redirects=True, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        logger.exception("http fetch failed")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

    if resp.status_code >= 400:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - HTTP {resp.status_code}"))

    content_type = resp.headers.get("content-type", "")
    text = resp.text

    if ("text/html" in content_type or "application/xhtml+xml" in content_type) and not force_raw:
        if readabilipy and markdownify:
            try:
                parsed = readabilipy.simple_json.simple_json_from_html_string(text, use_readability=True)
                html_content = parsed.get("content") or parsed.get("excerpt") or ""
                if not html_content:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(text, "html.parser")
                    article = soup.find("article")
                    html_content = article.decode_contents() if article else (soup.body.decode_contents() if soup.body else text)
                md = markdownify.markdownify(html_content, heading_style=markdownify.ATX)
                return md, ""
            except Exception:
                logger.exception("readability/markdownify failed")
                return text, "<error>Failed to simplify HTML; returning raw text</error>\n\n"
        else:
            return text, "<error>readabilipy/markdownify not installed; returning raw HTML</error>\n\n"

    return text, f"Content type {content_type} returned as raw text.\n\n"


def extract_text_from_pdf_bytes(data: bytes, max_pages: int = 6) -> str:
    if not HAS_PYMUPDF:
        raise RuntimeError("PyMuPDF (fitz) is not installed. Install with `pip install pymupdf` to enable PDF extraction.")
    doc = fitz.open(stream=data, filetype="pdf")
    pages = []
    for i, p in enumerate(doc):
        if i >= max_pages:
            break
        pages.append(p.get_text())
    return "\n".join(pages)


# -----------------------
# A4F LLM integration
# -----------------------
async def call_a4f_chat_completions(messages: List[Dict[str, str]], model: str | None = None, temperature: float = 0.0) -> str:
    """
    Call A4F /chat/completions endpoint. Expects OpenAI-compatible response.
    Returns the assistant text.
    """
    if not (A4F_API_KEY and A4F_BASE_URL and (model or A4F_MODEL)):
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="A4F configuration missing. Set A4F_API_KEY, A4F_BASE_URL and A4F_MODEL."))

    payload = {
        "model": model or A4F_MODEL,
        "messages": messages,
        "temperature": temperature,
        # you can add max_tokens, top_p, etc. here as needed
    }

    client = get_httpx_client()
    try:
        r = await client.post(
            f"{A4F_BASE_URL.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {A4F_API_KEY}",
                "Content-Type": "application/json",
                "User-Agent": USER_AGENT,
            },
            json=payload,
            timeout=60,
        )
    except httpx.RequestError as e:
        logger.exception("A4F request error")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"A4F request failed: {e!r}"))

    if r.status_code >= 400:
        logger.error("A4F error: %s", r.text)
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"A4F API error: {r.status_code} - {r.text}"))

    try:
        data = r.json()
    except Exception:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message="A4F returned non-json response"))

    # Try to get the assistant content from common response shapes
    # 1) choices[0].message.content (chat-completions)
    # 2) choices[0].text (older completion style)
    try:
        choice = data.get("choices", [])[0]
        if isinstance(choice.get("message"), dict) and choice["message"].get("content"):
            return choice["message"]["content"]
        if choice.get("text"):
            return choice["text"]
        # fallback to whole choice
        return str(choice)
    except Exception:
        # if response shape unexpected, return the raw JSON string
        return str(data)


async def llm_summarize_hook(text: str, max_tokens: int = 500) -> str:
    """
    Use A4F LLM to summarize text. Builds a short prompt and calls the chat completions endpoint.
    """
    if not (A4F_API_KEY and A4F_BASE_URL and A4F_MODEL):
        # fallback to simple truncation if A4F not configured
        return text[: max_tokens * 2] + ("..." if len(text) > max_tokens * 2 else "")

    system = {"role": "system", "content": "You are a concise summarization assistant for research papers."}
    user = {"role": "user", "content": f"Summarize the following text in {max_tokens} tokens or fewer, producing clear bullet points or short paragraphs:\n\n{text}"}
    return await call_a4f_chat_completions([system, user], model=A4F_MODEL, temperature=0.0)


# -----------------------
# Tools
# -----------------------

# validate: required by Puch
@mcp.tool
async def validate() -> str:
    return MY_NUMBER
 
@mcp.tool
async def about() -> dict[str, str]:
    server_name = "Research Buddy MCP"
    server_description = dedent("""
    A modular Python-based system for managing research workflows. 
    Provides tools for setting reminders, generating reading lists, searching academic databases, and formatting citations. 
    Built on the MCP framework with integrations to OpenAlex, CrossRef, arXiv, and AI-powered utilities for summarization and keyword extraction.
    """)

    return {
        "name": server_name,
        "description": server_description
    }

# ---- search_research_papers ----
@mcp.tool
async def search_research_papers(
    keyword: Annotated[str, Field(description="Keyword to search arXiv (e.g., 'graph neural networks')")],
    max_results: Annotated[int, Field(description="Max results (1-20)", ge=1, le=20)] = 5,
    start: Annotated[int, Field(description="Offset for pagination", ge=0)] = 0,
) -> str:
    if not keyword or not keyword.strip():
        raise McpError(ErrorData(code=INVALID_PARAMS, message="keyword must be provided"))

    arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{keyword}&start={start}&max_results={max_results}"
    client = get_httpx_client()
    try:
        resp = await client.get(arxiv_url, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        logger.exception("arXiv fetch failed")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"arXiv fetch failed: {e!r}"))

    if resp.status_code >= 400:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"arXiv responded with {resp.status_code}"))

    parsed = feedparser.parse(resp.text)
    entries = parsed.entries or []
    if not entries:
        return f"No papers found for '{keyword}'."

    lines = [f"🧠 Top {min(len(entries), max_results)} papers for **{keyword}**:\n"]
    for e in entries[:max_results]:
        title = (e.get("title") or "").strip().replace("\n", " ")
        summary = (e.get("summary") or "").strip().replace("\n", " ")
        authors = ", ".join(a.name for a in e.get("authors", [])) if e.get("authors") else "Unknown"
        published = e.get("published", "")[:10] if e.get("published") else ""
        pdf_link = None
        for link in e.get("links", []):
            if link.get("type") == "application/pdf" or (link.get("href") and link.get("href").endswith(".pdf")):
                pdf_link = link.get("href")
                break
        pdf_link = pdf_link or e.get("link") or ""
        lines.append(f"**{title}**  \nAuthors: {authors}  \nPublished: {published}  \nPDF: {pdf_link or 'N/A'}\n{summary[:500]}...\n")
    return "\n".join(lines)


# ---- fetch_latest_research_news ----
@mcp.tool
async def fetch_latest_research_news(
    source: Annotated[str, Field(description="arxiv_cs | arxiv_ai | nature | science | mit | custom")],
    max_items: Annotated[int, Field(description="Max items (1-20)", ge=1, le=20)] = 5,
    custom_rss: Annotated[str | None, Field(description="If source == custom, provide RSS/Atom URL")] = None,
) -> str:
    source_map = {
        "arxiv_cs": "http://export.arxiv.org/rss/cs",
        "arxiv_ai": "http://export.arxiv.org/rss/cs.AI",
        "nature": "https://www.nature.com/nature.rss",
        "science": "https://www.sciencemag.org/rss/news_current.xml",
        "mit": "https://www.technologyreview.com/feed/",
    }

    if source == "custom":
        if not custom_rss:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="custom source requires custom_rss"))
        feed_url = custom_rss
    else:
        feed_url = source_map.get(source)
        if not feed_url:
            valid = ", ".join(list(source_map.keys()) + ["custom"])
            raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Unknown source '{source}'. Valid: {valid}"))

    client = get_httpx_client()
    try:
        resp = await client.get(feed_url, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        logger.exception("feed fetch failed")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Feed fetch failed: {e!r}"))

    if resp.status_code >= 400:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Feed responded with {resp.status_code}"))

    parsed = feedparser.parse(resp.text)
    entries = parsed.entries or []
    if not entries:
        return f"No items found in feed: {feed_url}"

    items = entries[:max_items]
    lines = [f"📰 Latest from {source} (showing {len(items)} items):\n"]
    for e in items:
        title = (e.get("title") or "<no title>").strip()
        link = e.get("link") or ""
        pd = e.get("published_parsed") or e.get("updated_parsed")
        published = ""
        try:
            if pd:
                published = datetime(*pd[:6]).strftime("%Y-%m-%d")
            else:
                published = (e.get("published") or e.get("updated") or "")[:10]
        except Exception:
            published = (e.get("published") or e.get("updated") or "")[:10]

        summary_raw = e.get("summary") or e.get("description") or ""
        summary = re.sub(r"\s+", " ", summary_raw).strip()
        if len(summary) > 300:
            summary = summary[:300].rstrip() + "..."
        lines.append(f"**{title}**  \nSource: {source}  \nDate: {published}  \nLink: {link}  \n{summary}\n")
    return "\n".join(lines)


# ---- summarize_paper_url ----
@mcp.tool
async def summarize_paper_url(
    url: Annotated[AnyUrl, Field(description="URL to a PDF or HTML article")],
    max_sentences: Annotated[int, Field(description="Max sentences in short summary", ge=1, le=20)] = 5,
    use_llm: Annotated[bool, Field(description="If True, call LLM summarizer when available")] = False,
) -> str:
    client = get_httpx_client()
    cache_key = f"summarize:{str(url)}:{max_sentences}:{use_llm}"
    # try cache
    if redis_client:
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                return cached.decode()
        except Exception:
            pass

    try:
        resp = await client.get(str(url), headers={"User-Agent": USER_AGENT}, follow_redirects=True)
    except Exception as e:
        logger.exception("fetch for summarize failed")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

    if resp.status_code >= 400:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"URL returned {resp.status_code}"))

    content_type = resp.headers.get("content-type", "")
    text = ""
    try:
        if "pdf" in content_type or str(url).lower().endswith(".pdf"):
            if not HAS_PYMUPDF:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message="PDF extraction requires PyMuPDF (pip install pymupdf)"))
            text = extract_text_from_pdf_bytes(resp.content, max_pages=8)
        else:
            md, _err = await fetch_text_from_url(str(url))
            text = md
    except McpError:
        raise
    except Exception as e:
        logger.exception("extraction failed")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Extraction failed: {e!r}"))

    # extractive summary (fallback)
    def extractive_summary(t: str, max_sent: int = 5) -> str:
        sents = re.split(r"(?<=[.!?])\s+", t.strip())
        if len(sents) <= max_sent:
            return " ".join(sents)
        words = re.findall(r"\w+", t.lower())
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        scores = []
        for s in sents:
            ws = re.findall(r"\w+", s.lower())
            score = sum(freq.get(w, 0) for w in ws) / (len(ws) ** 0.5 if ws else 1)
            scores.append((score, s))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = [s for _, s in scores[:max_sent]]
        top_sorted = [s for s in sents if s in top]
        return " ".join(top_sorted)

    summary = ""
    # use A4F LLM if requested and configured
    if use_llm and A4F_API_KEY and A4F_BASE_URL and A4F_MODEL:
        try:
            summary = await llm_summarize_hook(text, max_tokens=400)
        except Exception:
            summary = extractive_summary(text, max_sent=max_sentences)
    else:
        summary = extractive_summary(text, max_sent=max_sentences)

    result = f"🔎 **Summary for {url}**\n\n{summary}\n\n---\n\n(Use `use_llm=True` and set A4F_API_KEY/A4F_BASE_URL/A4F_MODEL to call LLM summarizer.)"

    if redis_client:
        try:
            await redis_client.set(cache_key, result, ex=60 * 60)
        except Exception:
            pass

    return result


# ---- write_cold_email ----
@mcp.tool
async def write_cold_email(
    recipient_name: Annotated[str, Field(description="Recipient full name")],
    recipient_affiliation: Annotated[str | None, Field(description="Recipient affiliation")] = None,
    your_name: Annotated[str, Field(description="Your full name")] = None,
    your_affiliation: Annotated[str | None, Field(description="Your affiliation")] = None,
    role: Annotated[str | None, Field(description="Your role/title")] = None,
    context: Annotated[str | None, Field(description="One line context for outreach")] = None,
    tone: Annotated[str, Field(description="professional|concise|casual")] = "professional",
) -> str:
    if not recipient_name or not your_name:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="recipient_name and your_name are required"))

    salutation = f"Dear {recipient_name},"
    if tone == "casual":
        salutation = f"Hi {recipient_name},"

    role_part = f", {role}" if role else ""
    aff_part = f" from {your_affiliation}" if your_affiliation else ""

    opening = f"My name is {your_name}{role_part}{aff_part}."
    if context:
        opening += f" {context.strip()}"

    body = (
        "I'm reaching out because I believe our research interests align and I'd be excited to discuss a potential collaboration. "
        "If you're open, I'd love to schedule a 20-minute call to explore ideas."
    )

    closing = f"Thank you for your time.\n\nBest regards,\n{your_name}"

    concise = f"{salutation}\n\n{opening} {body}\n\n{closing}"
    detailed = f"{salutation}\n\n{opening}\n\n{body}\n\nIf helpful, I can send a one-page summary of our recent results and potential directions.\n\n{closing}"

    return concise if tone == "concise" else detailed


# ---- generate_citation ----
CROSSREF = "https://api.crossref.org/works/"


def _format_authors_cr(auths):
    parts = []
    for a in auths:
        given = a.get("given", "")
        family = a.get("family", "")
        if given and family:
            parts.append(f"{given} {family}")
        elif family:
            parts.append(family)
    return parts


@mcp.tool
async def generate_citation(
    doi: Annotated[str, Field(description="DOI of publication (e.g., 10.1000/xyz123)")],
    style: Annotated[str, Field(description="APA|MLA|IEEE")] = "APA",
) -> str:
    if not doi or not doi.strip():
        raise McpError(ErrorData(code=INVALID_PARAMS, message="doi required"))

    url = CROSSREF + doi.strip()
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        logger.exception("CrossRef fetch failed")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"CrossRef fetch failed: {e!r}"))

    if resp.status_code >= 400:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"CrossRef responded {resp.status_code}"))

    data = resp.json().get("message", {})
    title = (data.get("title") or [""])[0]
    authors = _format_authors_cr(data.get("author", []))
    year = (data.get("issued", {}).get("date-parts", [[None]])[0][0])
    journal = (data.get("container-title") or [""])[0]

    style = style.upper()
    if style == "APA":
        authors_form = ", ".join([a.split()[-1] for a in authors]) or "Unknown"
        return f"{authors_form} ({year}). {title}. {journal}."
    elif style == "MLA":
        authors_form = ", ".join(authors) or "Unknown"
        return f"{authors_form}. \"{title}.\" {journal}, {year}."
    elif style == "IEEE":
        authors_form = ", ".join(authors) or "Unknown"
        return f"{authors_form}, \"{title},\" {journal}, {year}."
    else:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="Unsupported style"))


# ---- a4f_llm: direct LLM tool ----
@mcp.tool
async def a4f_llm(
    prompt: Annotated[str | None, Field(description="Single-user prompt (mutually exclusive with messages)")]=None,
    messages: Annotated[List[Dict[str, str]] | None, Field(description="Chat-style messages (list of {role,content})")]=None,
    model: Annotated[str | None, Field(description="Optional override model name")] = None,
    temperature: Annotated[float, Field(description="temperature")] = 0.0,
) -> str:
    """
    Call your A4F OpenAI-compatible model.
    Provide either 'prompt' (string) or 'messages' (chat format). If both provided, messages is used.
    """
    if messages is None:
        if not prompt:
            raise McpError(ErrorData(code=INVALID_PARAMS, message="Provide either prompt or messages"))
        messages = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt},
        ]

    # validate messages shape
    if not isinstance(messages, list) or not all(isinstance(m, dict) and "role" in m and "content" in m for m in messages):
        raise McpError(ErrorData(code=INVALID_PARAMS, message="messages must be a list of {role,content} dicts"))

    return await call_a4f_chat_completions(messages, model=model or A4F_MODEL, temperature=temperature)

    # In-memory fallback store if Redis isn't available
_local_reminders: dict[str, dict] = {}

def _reminder_key(user_id: str) -> str:
    return f"reminders:{user_id}"

# ---------- Reminder tools ----------
@mcp.tool
async def add_reminder(
    user_id: Annotated[str, Field(description="Unique user id (use puch_user_id if available)")],
    title: Annotated[str, Field(description="Short title for the reminder")],
    due_iso: Annotated[str, Field(description="Due date/time in ISO 8601 (e.g. 2025-08-10T15:30:00+05:30)")],
    note: Annotated[str | None, Field(description="Optional note/body for reminder")] = None,
) -> str:
    """
    Adds a reminder for a user. Returns reminder id and stored info.
    """
    # validate
    try:
        due = datetime.fromisoformat(due_iso)
    except Exception:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="due_iso must be a valid ISO 8601 datetime string"))

    reminder = {
        "id": str(uuid.uuid4()),
        "title": title.strip(),
        "due_iso": due.isoformat(),
        "note": (note or "").strip(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if redis_client:
        try:
            key = _reminder_key(user_id)
            # store as a simple json list in Redis
            existing = await redis_client.get(key)
            if existing:
                arr = json.loads(existing)
            else:
                arr = []
            arr.append(reminder)
            await redis_client.set(key, json.dumps(arr))
        except Exception:
            # fallback to in-memory store on error
            user_map = _local_reminders.setdefault(user_id, {})
            user_map[reminder["id"]] = reminder
    else:
        user_map = _local_reminders.setdefault(user_id, {})
        user_map[reminder["id"]] = reminder

    return f"Reminder added: {reminder['id']} (due {reminder['due_iso']})"


@mcp.tool
async def list_reminders(
    user_id: Annotated[str, Field(description="Unique user id (use puch_user_id if available)")],
) -> str:
    """
    List reminders for a user, sorted by soonest due date.
    """
    reminders = []
    if redis_client:
        try:
            key = _reminder_key(user_id)
            existing = await redis_client.get(key)
            if existing:
                reminders = json.loads(existing)
        except Exception:
            reminders = list(_local_reminders.get(user_id, {}).values())
    else:
        reminders = list(_local_reminders.get(user_id, {}).values())

    if not reminders:
        return "No reminders found."

    # sort by due date
    def _due_dt(r):
        try:
            return datetime.fromisoformat(r["due_iso"])
        except Exception:
            return datetime.max

    reminders.sort(key=_due_dt)
    lines = []
    for r in reminders:
        lines.append(f"- ID: {r['id']}\n  Title: {r['title']}\n  Due: {r['due_iso']}\n  Note: {r.get('note','')}\n")
    return "\n".join(lines)


@mcp.tool
async def remove_reminder(
    user_id: Annotated[str, Field(description="Unique user id (use puch_user_id if available)")],
    reminder_id: Annotated[str, Field(description="ID returned when reminder was created")],
) -> str:
    """
    Remove a reminder by id.
    """
    removed = False
    if redis_client:
        try:
            key = _reminder_key(user_id)
            existing = await redis_client.get(key)
            if existing:
                arr = json.loads(existing)
                new_arr = [r for r in arr if r.get("id") != reminder_id]
                if len(new_arr) != len(arr):
                    await redis_client.set(key, json.dumps(new_arr))
                    removed = True
        except Exception:
            user_map = _local_reminders.get(user_id, {})
            if user_map.pop(reminder_id, None):
                removed = True
    else:
        user_map = _local_reminders.get(user_id, {})
        if user_map and user_map.pop(reminder_id, None):
            removed = True

    if removed:
        return f"Removed reminder {reminder_id}"
    raise McpError(ErrorData(code=INVALID_PARAMS, message=f"Reminder {reminder_id} not found"))


# ---------- Reading-list generator using OpenAlex ----------
OPENALEX_BASE = "https://api.openalex.org/works"

@mcp.tool
async def generate_reading_list(
    topic: Annotated[str, Field(description="Research topic or query (e.g., 'convolutional neural networks')")],
    max_results: Annotated[int, Field(description="Max results (1-20)", ge=1, le=20)] = 10,
    years_back: Annotated[int | None, Field(description="Optional: only include papers published in the last N years")] = None,
) -> str:
    """
    Returns a top-cited reading list for a topic using OpenAlex (sorted by cited_by_count).
    """
    if not topic or not topic.strip():
        raise McpError(ErrorData(code=INVALID_PARAMS, message="topic required"))

    # build filter
    filters = []
    # text search
    query = topic.strip()
    params = {
        "search": query,
        "per_page": max_results,
        "sort": "cited_by_count:desc"
    }
    if years_back is not None:
        try:
            years_back = int(years_back)
            cutoff_year = datetime.now().year - years_back
            params["filter"] = f"from_publication_date:{cutoff_year}-01-01"
        except Exception:
            pass

    client = get_httpx_client()
    try:
        resp = await client.get(OPENALEX_BASE, params=params, headers={"User-Agent": USER_AGENT})
    except Exception as e:
        logger.exception("OpenAlex fetch failed")
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"OpenAlex fetch failed: {e!r}"))

    if resp.status_code >= 400:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"OpenAlex responded with {resp.status_code}"))

    data = resp.json()
    results = data.get("results", [])[:max_results]
    if not results:
        return f"No works found for topic '{topic}'"

    lines = [f"📚 Top {len(results)} most-cited works for **{topic}** (from OpenAlex):\n"]
    for w in results:
        title = w.get("display_name")
        doi = w.get("doi") or ""
        authors_list = []
        for au in w.get("authorships", [])[:6]:
            name = au.get("author", {}).get("display_name")
            if name:
                authors_list.append(name)
        authors = ", ".join(authors_list) if authors_list else "Unknown"
        year = w.get("publication_year") or ""
        cited = w.get("cited_by_count", 0)
        host_venue = w.get("host_venue", {}).get("display_name", "")
        open_url = w.get("id")  # OpenAlex URL
        # Try to extract an article link (e.g., via 'primary_location')
        primary = w.get("primary_location", {}) or {}
        source_url = primary.get("source_url") or primary.get("landing_page_url") or ""
        line = (
            f"**{title}**  \nAuthors: {authors}  \nVenue: {host_venue}  \nYear: {year}  \nCitations: {cited}  \nDOI: {doi or 'N/A'}  \nURL: {source_url or open_url}\n"
        )
        lines.append(line)

    return "\n".join(lines)



# -----------------------
# Run server
# -----------------------
async def main():
    logger.info(f"Starting Research MCP (Puch-ready) on http://{HOST}:{PORT}/mcp/")
    await mcp.run_async("streamable-http", host=HOST, port=PORT)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down Research MCP")
