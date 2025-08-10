# tools/citations.py
from typing import Annotated
from pydantic import Field
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR
import httpx

CROSSREF = "https://api.crossref.org/works/"

def _format_authors_cr(auths):
    parts = []
    for a in auths:
        given = a.get("given","")
        family = a.get("family","")
        if given and family:
            parts.append(f"{given} {family}")
        elif family:
            parts.append(family)
    return parts

def register(mcp):
    @mcp.tool
    async def generate_citation(
        doi: Annotated[str, Field(description="DOI of publication")],
        style: Annotated[str, Field(description="APA|MLA|IEEE")] = "APA",
    ) -> str:
        if not doi or not doi.strip():
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="doi required"))

        url = CROSSREF + doi
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, headers={"User-Agent": "PuchResearch/1.0"})
        except Exception as e:
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
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Unsupported style"))
