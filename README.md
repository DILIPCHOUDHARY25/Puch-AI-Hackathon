# Research Assistant MCP Tools

A modular Python-based system of tools for managing research-related workflows including reminders, reading list generation, research paper searching, and citation formatting. Built on MCP framework with integrations to OpenAlex, CrossRef, arXiv, and AI-powered utilities.

---

## Features

### Reminder System
- Add, list, and remove personal reminders with deadlines.
- Supports ISO 8601 timestamps.
- Stores data either in Redis or an in-memory fallback.
- Ideal for research deadlines, meeting reminders, or study schedules.

### Reading List Generation
- Fetches top-cited research papers on a specified topic using the OpenAlex API.
- Supports filtering papers by publication year.
- Returns detailed metadata including title, authors, publication venue, citation count, DOI, and links.

### Research Paper Search
- Searches arXiv for papers using keywords.
- Returns recent papers with summaries, authors, published date, and PDF links.

### Citation Generator
- Generates formatted citations (APA, MLA, IEEE) from DOIs using CrossRef API.
- Automatically formats author names, titles, journals, and publication years.

### Cold Email Writer
- Generates professional, concise, or casual cold emails for outreach.
- Customizable with recipient and sender details, context, and tone.

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/research-assistant-mcp.git
cd research-assistant-mcp

2. **Set up a Python virtual environment**
python -m venv .venv
source .venv/bin/activate   # On Windows use `.venv\Scripts\activate`

3. **Install dependencies**
pip install -r requirements.txt

4. **Set up Redis (optional but recommended)**
Install Redis on your machine or use a cloud Redis provider.
Configure Redis connection in your environment or config file.
