# tools/search_papers.py
from typing import Annotated
from pydantic import Field
import xml.etree.ElementTree as ET
from urllib.parse import quote_plus
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR
from tools.common import get_httpx_client, USER_AGENT

def _parse_arxiv_feed(xml_text: str, max_items: int = 5):
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_text)
    entries = root.findall("atom:entry", ns)
    results = []
    for e in entries[:max_items]:
        title = (e.find("atom:title", ns).text or "").strip().replace("\n", " ")
        summary = (e.find("atom:summary", ns).text or "").strip().replace("\n", " ")
        published = e.find("atom:published", ns).text if e.find("atom:published", ns) is not None else ""
        authors = [a.find("atom:name", ns).text for a in e.findall("atom:author", ns)]
        links = {}
        for l in e.findall("atom:link", ns):
            href = l.attrib.get("href")
            lt = l.attrib.get("type") or l.attrib.get("rel") or "link"
            links[lt] = href
        results.append({
            "title": title,
            "summary": summary,
            "published": published,
            "authors": authors,
            "links": links
        })
    return results

def register(mcp):
    @mcp.tool
    async def search_research_papers(
        keyword: Annotated[str, Field(description="Keyword to search arXiv")],
        max_results: Annotated[int, Field(description="Max results", ge=1, le=20)] = 5,
        start: Annotated[int, Field(description="Result offset for pagination", ge=0)] = 0,
    ) -> str:
        if not keyword or not keyword.strip():
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="keyword must be provided"))

        query = quote_plus(keyword.strip())
        url = (
            f"http://export.arxiv.org/api/query?"
            f"search_query=all:{query}&start={start}&max_results={max_results}"
        )

        client = get_httpx_client()
        try:
            resp = await client.get(url, headers={"User-Agent": USER_AGENT})
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"arXiv fetch failed: {e!r}"))

        if resp.status_code >= 400:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"arXiv responded with {resp.status_code}"))

        items = _parse_arxiv_feed(resp.text, max_results)
        if not items:
            return f"No papers found for '{keyword}'"

        lines = [f"🧠 Top {len(items)} papers for **{keyword}**:\n"]
        for it in items:
            authors = ", ".join(it["authors"][:5]) or "Unknown"
            pdf_link = (
                it["links"].get("application/pdf")
                or it["links"].get("pdf")
                or next((v for k, v in it["links"].items() if "pdf" in (v or "")), None)
            )
            lines.append(
                f"**{it['title']}**  \n"
                f"Authors: {authors}  \n"
                f"Published: {it['published'][:10]}  \n"
                f"PDF: {pdf_link or 'N/A'}\n"
                f"{it['summary'][:400]}...\n"
            )

        return "\n".join(lines)
