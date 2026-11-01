import inspect
import json
import threading
from datetime import date

import trafilatura
from ddgs import DDGS

MODEL = "llama3.1:8b"
VISION_MODEL = "gemma3:4b"
NAME = "Nellow"

SOURCES = []
LOCK = threading.Lock()  # one reply at a time


def web_search(query: str) -> str:
    """Search the web for general facts and information.

    Args:
        query: What to search for.
    """
    try:
        results = DDGS().text(query, max_results=4)
    except Exception as e:
        return f"Search failed: {e}"
    if not results:
        return "No results found."
    out = []
    for i, r in enumerate(results):
        SOURCES.append(r["href"])
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


def news_search(query: str) -> str:
    """Search NEWS from the past week. Use for recent, latest, trending or breaking topics.

    Args:
        query: What to search news for.
    """
    try:
        results = DDGS().news(query, max_results=4, timelimit="w")
    except Exception as e:
        return f"News search failed: {e}"
    if not results:
        return "No recent news found."
    out = []
    for r in results:
        SOURCES.append(r["url"])
        out.append(f"{r['title']} ({r.get('date', '')}, {r.get('source', '')})\n{r['url']}\n{r['body']}")
    return "\n\n".join(out)


def read_page(url: str) -> str:
    """Read the full text of a web page. Use after a search to get details.

    Args:
        url: The full page URL to read.
    """
    try:
        html = trafilatura.fetch_url(url)
        text = trafilatura.extract(html) if html else None
    except Exception as e:
        return f"Could not read page: {e}"
    return text[:6000] if text else "Could not extract text from that page."


TOOLS = {"web_search": web_search, "news_search": news_search, "read_page": read_page}


def run_tool(name, args):
    if name not in TOOLS:
        return "Unknown tool."
    fn = TOOLS[name]
    allowed = inspect.signature(fn).parameters
    clean = {k: v for k, v in args.items() if k in allowed}  # drop made-up arguments
    try:
        return fn(**clean)
    except Exception as e:
        return f"Tool error: {e}"


ROUTER_PROMPT = (
    "Decide if the user's message needs a LIVE WEB SEARCH. "
    "Reply with exactly one word: SEARCH or CHAT.\n"
    "SEARCH = current events, news, latest/recent/today/this week, prices, "
    "scores, weather, or facts about specific people, products or companies "
    "that may have changed recently.\n"
    "CHAT = greetings, questions about the assistant or the user (names, "
    "what was said earlier), opinions, advice, explanations, math, coding, "
    "writing, translations, and general knowledge that does not change."
    "\nShort follow-ups like 'what about X?' or 'any news on X?' after a "
    "SEARCH topic are SEARCH."
)


def parse_text_tool_call(text):
    try:
        data = json.loads(text.strip())
        name = data.get("name")
        args = data.get("parameters") or data.get("arguments") or {}
        if name in TOOLS and isinstance(args, dict):
            return name, args
    except Exception:
        pass
    return None


def new_history():
    return [{
        "role": "system",
        "content": (
            f"You are {NAME}, a helpful AI assistant. Today's date is {date.today()}. "
            "For greetings, casual chat, or things you already know, answer "
            "directly WITHOUT using any tools. "
            "Use news_search for anything recent, latest, trending or breaking. "
            "Use web_search for general facts. Check the dates on results and "
            "ignore old articles when the question is about recent events. "
            "Only state what the sources actually say. If results are unclear or "
            "conflicting, say so. Answer clearly and concisely."
        ),
    }]
