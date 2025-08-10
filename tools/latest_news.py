# tools/latest_news.py
from typing import Annotated
from pydantic import Field
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR
from tools.common import get_httpx_client, USER_AGENT
import feedparser
import re
from datetime import datetime

def register(mcp):
    @mcp.tool
    async def fetch_latest_research_news(
        source: Annotated[str, Field(description="arxiv_cs | arxiv_ai | nature | science | mit | custom")],
        max_items: Annotated[int, Field(description="Max items to return", ge=1, le=20)] = 5,
        custom_rss: Annotated[str | None, Field(description="If source == 'custom', provide RSS/Atom URL")] = None,
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
                raise McpError(ErrorData(code=INTERNAL_ERROR, message="custom source requires custom_rss"))
            feed_url = custom_rss
        else:
            feed_url = source_map.get(source)
            if not feed_url:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Unknown source {source}"))

        client = get_httpx_client()
        try:
            resp = await client.get(feed_url, headers={"User-Agent": USER_AGENT})
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Feed fetch error: {e!r}"))

        if resp.status_code >= 400:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Feed returned {resp.status_code}"))

        parsed = feedparser.parse(resp.text)
        if not parsed.entries:
            return f"No items in feed {feed_url}"

        items = parsed.entries[:max_items]
        lines = [f"📰 Latest from {source} (showing {len(items)} items):\n"]
        for e in items:
            title = e.get("title", "<no title>").strip()
            link = e.get("link", "")
            published = e.get("published", "") or e.get("updated", "")
            pd = e.get("published_parsed") or e.get("updated_parsed")
            try:
                if pd:
                    published = datetime(*pd[:6]).strftime("%Y-%m-%d")
            except Exception:
                pass
            summary = e.get("summary", "") or e.get("description", "")
            summary = re.sub(r"\s+", " ", summary).strip()
            if len(summary) > 300:
                summary = summary[:300].rstrip() + "..."
            lines.append(f"**{title}**  \nSource: {source}  \nDate: {published}  \nLink: {link}  \n{summary}\n")

        return "\n".join(lines)
