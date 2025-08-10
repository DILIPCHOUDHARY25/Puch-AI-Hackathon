# tools/common.py
import os
import logging
import httpx
from dotenv import load_dotenv
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
from mcp.server.auth.provider import AccessToken

load_dotenv()

USER_AGENT = "PuchResearchCopilot/1.0 (+https://example.com)"
logger = logging.getLogger("research_mcp.common")

REDIS_URL = os.environ.get("REDIS_URL")
redis_client = None
try:
    if REDIS_URL:
        import aioredis  # type: ignore
        redis_client = aioredis.from_url(REDIS_URL)
except ImportError:
    redis_client = None

_httpx_client = None
def get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=30)
    return _httpx_client

class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self._token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self._token:
            return AccessToken(token=token, client_id="puch-client", scopes=["*"], expires_at=None)
        return None

# Lightweight fetcher using readabilipy + markdownify for HTML
from bs4 import BeautifulSoup
import markdownify
import readabilipy
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR

class Fetch:
    @staticmethod
    async def fetch_text_from_url(url: str, force_raw: bool = False) -> tuple[str, str]:
        client = get_httpx_client()
        try:
            resp = await client.get(url, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))

        if resp.status_code >= 400:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - HTTP {resp.status_code}"))

        content_type = resp.headers.get("content-type", "")
        text = resp.text

        if ("text/html" in content_type or "application/xhtml+xml" in content_type) and not force_raw:
            try:
                parsed = readabilipy.simple_json.simple_json_from_html_string(text, use_readability=True)
                html_content = parsed.get("content") or parsed.get("excerpt") or ""
                if not html_content:
                    soup = BeautifulSoup(text, "html.parser")
                    article = soup.find("article")
                    html_content = article.decode_contents() if article else soup.body.decode_contents()
                md = markdownify.markdownify(html_content, heading_style=markdownify.ATX)
                return md, ""
            except Exception as e:
                logger.exception("readability failed")
                return text, "<error>Failed to simplify HTML; returning raw text</error>\n\n"

        return text, f"Content type {content_type} returned as raw text.\n\n"
