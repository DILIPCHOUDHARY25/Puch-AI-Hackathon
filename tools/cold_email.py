# tools/cold_email.py
import httpx
from typing import Annotated
from pydantic import Field
from mcp import ErrorData, McpError
from mcp.types import INTERNAL_ERROR
import os

A4F_API_KEY = os.getenv("A4F_API_KEY")
A4F_BASE_URL = os.getenv("A4F_BASE_URL", "https://api.a4f.co/v1")
A4F_MODEL = os.getenv("A4F_MODEL", "provider-6/gpt-4.1-mini")

def register(mcp):
    @mcp.tool
    async def write_cold_email(
        recipient_name: Annotated[str, Field(description="Recipient full name")],
        your_name: Annotated[str, Field(description="Your full name")],
        research_topic: Annotated[str, Field(description="Short description of research topic or collaboration idea")],
        recipient_affiliation: Annotated[str | None, Field(description="Recipient affiliation")] = None,
        your_affiliation: Annotated[str | None, Field(description="Your affiliation")] = None,
        role: Annotated[str | None, Field(description="Your role/title")] = None,
        tone: Annotated[str, Field(description="professional|concise|casual")] = "professional",
    ) -> str:

        if not A4F_API_KEY:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="A4F_API_KEY not set"))

        if not recipient_name or not your_name:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="recipient_name and your_name are required"))

        prompt = f"""
Write a {tone} cold outreach email to {recipient_name} ({recipient_affiliation or 'unknown affiliation'}).
Sender: {your_name} ({role or ''} {your_affiliation or ''}).
The research topic is: {research_topic}.
The email should be warm, clear, and relevant to research collaboration.
At the end of the email, add the sentence:
"Here are my supporting documents for the research."
        """.strip()

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{A4F_BASE_URL}/chat/completions",
                    headers={"Authorization": f"Bearer {A4F_API_KEY}"},
                    json={
                        "model": A4F_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 500,
                    },
                )
        except Exception as e:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Email generation API request failed: {e!r}"))

        if resp.status_code >= 400:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Email generation API responded {resp.status_code}"))

        try:
            email_text = resp.json()["choices"][0]["message"]["content"]
        except Exception:
            raise McpError(ErrorData(code=INTERNAL_ERROR, message="Invalid response from email generation API"))

        return email_text.strip()
