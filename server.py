"""Nellow's tools: web search, news search and page reading, plus prompts.

Every tool takes a `sources` list that belongs to one request, so replies
for different users never mix their source links.
"""
import base64
import io
import json
import re
import time
import zipfile
from datetime import date

import trafilatura
from ddgs import DDGS

MODEL = "llama3.1:8b"
VISION_MODEL = "gemma3:4b"
NAME = "Nellow"


def _search(kind, query, **kw):
    """Ask the search engines, with one retry: they sometimes rate-limit or come back empty."""
    last = None
    for attempt in range(2):
        try:
            d = DDGS()
            results = (d.text if kind == "text" else d.news)(query, **kw)
            if results:
                return results
        except Exception as e:
            last = e
        if attempt == 0:
            time.sleep(0.8)
    if last and "no results" not in str(last).lower():
        raise last
    return []


def web_search(query, sources):
    try:
        results = _search("text", query, max_results=5)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found. Try again with a shorter or different query."
    out = []
    for i, r in enumerate(results):
        sources.append(r["href"])
        entry = f"{r['title']}\n{r['href']}\n{r['body']}"
        if i < 1:
            try:
                html = trafilatura.fetch_url(r["href"])
                text = trafilatura.extract(html) if html else None
                if text:
                    entry += "\nFULL TEXT: " + text[:1500]
            except Exception:
                pass
        out.append(entry)
    return "\n\n".join(out)


def news_search(query, sources):
    try:
        results = _search("news", query, max_results=5, timelimit="w")
    except Exception:
        results = []
    if not results:  # nothing in the news this week: general web results are better than nothing
        return "No news from the past week. General web results instead:\n\n" + web_search(query, sources)
    out = []
    for r in results:
        sources.append(r["url"])
        out.append(f"{r['title']} ({r.get('date', '')}, {r.get('source', '')})\n{r['url']}\n{r['body']}")
    return "\n\n".join(out)


def read_page(url, sources):
    try:
        html = trafilatura.fetch_url(url)
        text = trafilatura.extract(html) if html else None
    except Exception as e:
        return f"Could not read page: {e}"
    if text:
        sources.append(url)
    return text[:6000] if text else "Could not extract text from that page."


TOOLS = {
    "web_search": (web_search, "query", "Search the web. Use for facts, including things that change over time: "
                   "latest versions, prices, schedules, scores, and current details about people, products or companies."),
    "news_search": (news_search, "query", "Search news articles from the past 7 days. Use only for current events and breaking news stories."),
    "read_page": (read_page, "url", "Read the full text of a web page. Use after a search when a result needs more detail."),
}

# One schema for both Groq and Ollama
TOOL_SPECS = [
    {"type": "function", "function": {"name": n, "description": d, "parameters": {
        "type": "object", "properties": {a: {"type": "string"}}, "required": [a]}}}
    for n, (_, a, d) in TOOLS.items()
]


def run_tool(name, args, sources):
    if name not in TOOLS:
        return "Unknown tool."
    fn, arg, _ = TOOLS[name]
    if not isinstance(args, dict):
        args = {}
    value = args.get(arg) or next((v for v in args.values() if isinstance(v, str)), "")
    if not value:
        return f"Tool error: missing '{arg}'."
    try:
        return fn(str(value), sources)
    except Exception as e:
        return f"Tool error: {e}"


def parse_text_tool_call(text):
    """Small local models sometimes print a tool call as JSON instead of calling it."""
    try:
        data = json.loads(text.strip())
        name = data.get("name")
        args = data.get("parameters") or data.get("arguments") or {}
        if name in TOOLS and isinstance(args, dict):
            return name, args
    except Exception:
        pass
    return None


ROUTER_PROMPT = (
    "Decide if the user's message needs a LIVE WEB SEARCH. "
    "Reply with exactly one word: SEARCH or CHAT.\n"
    "SEARCH = current events, news, latest/recent/today/this week, prices, "
    "scores, weather, or facts about specific people, products or companies "
    "that may have changed recently.\n"
    "CHAT = greetings, questions about the assistant or the user (names, "
    "what was said earlier), opinions, advice, explanations, math, coding, "
    "writing, translations, questions about an attached document or image, "
    "and general knowledge that does not change."
    "\nShort follow-ups like 'what about X?' or 'any news on X?' after a "
    "SEARCH topic are SEARCH."
)


def system_prompt(user_name=None):
    who = f" You are talking with {user_name}." if user_name else ""
    return (
        f"You are {NAME}, a helpful AI assistant. Today's date is {date.today()}.{who} "
        "For greetings, casual chat, or things you already know, answer "
        "directly WITHOUT using any tools. "
        "Use web_search for facts that may have changed since your training: latest versions, prices, "
        "scores, who holds a role, product details. Use news_search only for news events from the past week. "
        "Write short, specific search queries. If the results look unrelated to the question, search again "
        "with a clearer query (or the other search tool) before you answer. Check the dates on results and "
        "ignore old articles when the question is about recent events. "
        "Only state what the sources actually say. If results are unclear or "
        "conflicting, say so. When you use search results, credit them with normal Markdown links, "
        "like [Reuters](https://...). Never write citation markers such as 【1†L2-L5】. "
        "When the user attaches documents, their text is included in the message "
        "between [Document: name] markers; answer from that text. "
        "Format answers with Markdown: short paragraphs, lists where they help, "
        "and fenced code blocks with a language tag for code. "
        "Answer clearly and concisely."
    )


# ---------- documents ----------
TEXT_EXT = {"txt", "md", "markdown", "csv", "tsv", "json", "xml", "html", "htm", "yaml", "yml", "log", "ini", "toml",
            "py", "js", "ts", "jsx", "tsx", "java", "c", "h", "cpp", "hpp", "cs", "go", "rs", "rb", "php", "swift",
            "kt", "sql", "sh", "bat", "ps1", "css", "scss", "r", "m", "dart", "lua", "pl", "tex", "rtf"}


def _docx_text(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "ignore")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab/>", "\t", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")):
        text = text.replace(a, b)
    return text


def _pptx_text(raw):
    out = []
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        slides = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                        key=lambda n: int(re.search(r"\d+", n.rsplit("/", 1)[1]).group()))
        for i, n in enumerate(slides, 1):
            xml = z.read(n).decode("utf-8", "ignore")
            parts = re.findall(r"<a:t>([^<]*)</a:t>", xml)
            out.append(f"--- Slide {i} ---\n" + " ".join(parts))
    return "\n".join(out)


def extract_text(name, data_b64):
    """Turn an uploaded file into plain text. Raises ValueError with a friendly message."""
    raw = base64.b64decode(data_b64)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext == "pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ValueError("PDF support needs the pypdf package. Run: pip install pypdf")
        reader = PdfReader(io.BytesIO(raw))
        text = "\n\n".join((p.extract_text() or "") for p in reader.pages[:200])
        if not text.strip():
            raise ValueError("This PDF has no selectable text (it may be a scan).")
    elif ext == "docx":
        text = _docx_text(raw)
    elif ext == "pptx":
        text = _pptx_text(raw)
    elif ext in TEXT_EXT or not ext:
        text = raw.decode("utf-8-sig", "replace")
        if ext in ("html", "htm"):
            text = trafilatura.extract(text) or re.sub(r"<[^>]+>", " ", text)
    else:
        raise ValueError(f"Nellow can't read .{ext} files yet. Try PDF, Word, PowerPoint or a text file.")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("No text found in this file.")
    return text
