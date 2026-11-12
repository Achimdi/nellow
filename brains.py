"""Nellow's AI brains.

Cloud brains are tried in CHAIN order ("Auto"). Each brain has a few models, and each model has its own
free limit, so a brain only counts as "down" when all of its models are. A brain that hits a limit is rested
(skipped) until the limit should have reset. When no cloud brain can answer, Ollama on this computer takes over,
so people never get stuck.

API keys are read from (first match wins):
  1. an environment variable, e.g. GEMINI_API_KEY
  2. a line in the .env file in this folder, e.g. GEMINI_API_KEY=your-key
  3. a file named <brain>_key.txt in this folder, e.g. gemini_key.txt
A brain without a key is simply skipped.
"""
import json
import os
import re
import secrets
import time

import httpx
import ollama

import server as S

EFFORTS = ("turbo", "balanced", "thorough")

BRAINS = {
    "groq": {
        "name": "Groq", "free": True, "api": "openai", "url": "https://api.groq.com/openai/v1",
        "env": ["GROQ_API_KEY"], "signup": "https://console.groq.com/keys", "about": "gpt-oss 120B, very fast",
        "models": {"turbo": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
                   "balanced": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
                   "thorough": ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]},
        "vision": ["qwen/qwen3.8-27b"],
        "effort": {"turbo": "low", "balanced": "low", "thorough": "medium"}, "effort_models": ("openai/gpt-oss",),
        "empty": None,
    },
    "gemini": {
        "name": "Gemini", "free": True, "api": "openai", "url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "signup": "https://aistudio.google.com/apikey", "about": "Google Gemini Flash",
        "models": {"turbo": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"],
                   "balanced": ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
                   "thorough": ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash-lite"]},
        "vision": ["gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],
        "effort": {"turbo": "low", "balanced": "medium", "thorough": "high"}, "effort_models": ("gemini-",),
        "empty": "",
    },
    "openrouter": {
        "name": "OpenRouter", "free": True, "api": "openai", "url": "https://openrouter.ai/api/v1",
        "env": ["OPENROUTER_API_KEY"], "signup": "https://openrouter.ai/keys", "about": "the best free models of the day",
        "models": {e: ["openrouter/free", "google/gemma-4-31b-it:free", "nvidia/nemotron-3.5-super-vl:free"] for e in EFFORTS},
        "vision": ["openrouter/free", "google/gemma-4-31b-it:free", "nvidia/nemotron-3.5-super-vl:free"],
        "headers": {"X-Title": "Nellow"}, "empty": "",
    },
    "mistral": {
        "name": "Mistral", "free": True, "api": "openai", "url": "https://api.mistral.ai/v1",
        "env": ["MISTRAL_API_KEY"], "signup": "https://console.mistral.ai/api-keys", "about": "Mistral Medium",
        "models": {"turbo": ["mistral-small-latest", "mistral-medium-latest"],
                   "balanced": ["mistral-medium-latest", "mistral-small-latest"],
                   "thorough": ["mistral-large-latest", "mistral-medium-latest"]},
        "vision": ["mistral-medium-latest", "mistral-small-latest"],
        "empty": "",
    },
    "xai": {
        "name": "Grok", "free": False, "api": "openai", "url": "https://api.x.ai/v1",
        "env": ["XAI_API_KEY"], "signup": "https://console.x.ai", "about": "xAI Grok 4.3 (paid)",
        "models": {"turbo": ["grok-4.20-0309-non-reasoning", "grok-4.3"],
                   "balanced": ["grok-4.3", "grok-4.20-0309-non-reasoning"],
                   "thorough": ["grok-4.20-0309-reasoning", "grok-4.3"]},
        "vision": ["grok-4.3"],
        "empty": "",
    },
    "claude": {
        "name": "Claude", "free": False, "api": "anthropic", "url": "https://api.anthropic.com/v1",
        "env": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"], "signup": "https://console.anthropic.com", "about": "Anthropic Claude (paid)",
        "models": {"turbo": ["claude-haiku-5-5", "claude-sonnet-5-5"],
                   "balanced": ["claude-sonnet-5-5", "claude-haiku-5-5"],
                   "thorough": ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]},
        "vision": ["claude-haiku-5-5", "claude-sonnet-5-5"],
    },
}
CHAIN = ["groq", "gemini", "openrouter", "mistral", "xai", "claude"]  # "Auto" order: free brains first, paid last

LOCAL_MODELS = {"turbo": "llama3.2:3b", "balanced": "llama3.2:3b", "thorough": "qwen3:4b"}
LOCAL_FALLBACK = "llama3.2:3b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
if "://" not in OLLAMA_URL:
    OLLAMA_URL = "http://" + OLLAMA_URL


# ---------- keys ----------
def _dotenv():
    try:
        with open(".env", encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except OSError:
        return {}
    out = {}
    for ln in lines:
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip().strip("\"'")
    return out


def brain_key(bid):
    env = _dotenv()
    for name in BRAINS[bid]["env"]:
        k = os.environ.get(name) or env.get(name)
        if k and k.strip():
            return k.strip().strip("\"'").strip()
    path = f"{bid}_key.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            return f.read().strip().strip("\"'").strip()
    return ""


def any_cloud():
    return any(brain_key(b) for b in CHAIN)


def name_of(bid):
    return "Ollama" if bid == "local" else BRAINS[bid]["name"]


def short_model(m):
    return m.split("/")[-1].replace(":free", "") if m != "openrouter/free" else "free model"


# ---------- limits ----------
_REST = {}  # brain id, or "brain/model" -> (until, why)


def rest(key, seconds, why):
    if seconds and seconds > 0:
        _REST[key] = (time.time() + min(seconds, 6 * 3600), why)


def resting(key):
    r = _REST.get(key)
    if r and r[0] > time.time():
        return r
    _REST.pop(key, None)
    return None


class BrainDown(Exception):
    """A brain (or one of its models) can't answer right now. scope "brain" skips its other models too."""

    def __init__(self, why, wait=60, scope="model"):
        super().__init__(why)
        self.why, self.wait, self.scope = why, wait, scope


DAILY = ("per day", "rpd", "tpd", "daily", "quota", "resource_exhausted", "resource exhausted")


def down(status, text, headers):
    low = (text or "").lower()
    try:
        wait = float(headers.get("retry-after", "") or 0)
    except ValueError:
        wait = 0
    if status == 401:
        return BrainDown("didn't accept its API key", 900, "brain")
    if status == 402 or "credit balance" in low or "insufficient credit" in low or "insufficient_quota" in low:
        return BrainDown("has run out of credit", 3600, "brain")
    if status == 429:
        if any(w in low for w in DAILY):
            return BrainDown("has used up its free allowance for now", max(wait, 1800))
        return BrainDown("reached its free per-minute limit", wait or 60)
    if status == 413:
        return BrainDown("can't take a request this large on the free plan", 0)
    if status == 404:
        return BrainDown("couldn't find the model", 1800)
    if status == 403:
        return BrainDown("refused the request", 900)
    if status >= 500:
        return BrainDown("is overloaded right now", 30)
    return BrainDown(f"returned an error ({status})", 30)


def brain_order(choice):
    """Cloud brains to try, in order. The chosen brain goes first; the others follow so nobody gets stuck."""
    if choice == "local":
        return []
    ready = [b for b in CHAIN if brain_key(b) and not resting(b)]
    if choice in ready:
        ready.remove(choice)
        ready.insert(0, choice)
    return ready


def model_chain(bid, models, make):
    """Try each model of one brain in turn. Each model has its own free limit."""
    last, said = None, False
    for m in models:
        if resting(f"{bid}/{m}"):
            continue
        try:
            if said:
                yield "reset", None
                said = False
            yield "model", m
            for kind, val in make(m):
                if kind in ("token", "think"):
                    said = True
                yield kind, val
            return
        except BrainDown as e:
            if e.scope == "brain":
                raise
            rest(f"{bid}/{m}", e.wait, e.why)
            last = e
    raise last or BrainDown("is resting after hitting its limit", 60)


# ---------- talking to the APIs ----------
def flatten(msgs):
    """Turn tool calls and results into plain text, for a final answer without tools."""
    out, notes = [], []
    for m in msgs:
        if m.get("role") == "tool":
            notes.append(m["content"])
        elif not m.get("tool_calls"):
            out.append({k: v for k, v in m.items() if not k.startswith("_")})
    if notes:
        out.append({"role": "user", "content": "Search results gathered so far:\n\n" + "\n\n".join(notes) +
                    "\n\nUsing only these results, write your final answer now. Do not search again."})
    return out


def parse_failed_call(text):
    """Some models write a tool call in the wrong format. Recover it so web search still works."""
    m = re.search(r"<function=([\w.-]+)>(.*?)</function>", text or "", re.S)
    if not m or m.group(1) not in S.TOOLS:
        return None
    args = dict(re.findall(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", m.group(2), re.S))
    return {"id": "call_" + secrets.token_hex(6), "name": m.group(1), "args": json.dumps(args)}


def _clean(msgs, empty):
    out = []
    for m in msgs:
        m = {k: v for k, v in m.items() if not k.startswith("_")}
        if m.get("tool_calls") and m.get("content") is None and empty is not None:
            m["content"] = empty
        out.append(m)
    return out


def oai_stream(bid, model, msgs, use_tools, effort, show_think, _effort=True):
    """Stream from any OpenAI-compatible API (Groq, Gemini, OpenRouter, Mistral, xAI)."""
    b = BRAINS[bid]
    body = {"model": model, "stream": True,
            "messages": _clean(msgs, b.get("empty")) if use_tools else flatten(msgs)}
    if use_tools:
        body["tools"] = S.TOOL_SPECS
    level = b.get("effort", {}).get(effort) if _effort and model.startswith(b.get("effort_models", ("\0",))) else None
    if level:
        body["reasoning_effort"] = level
    if bid == "openrouter" and show_think and _effort:
        body["reasoning"] = {"effort": "high"}
    headers = {"Authorization": "Bearer " + brain_key(bid), **b.get("headers", {})}
    calls, started, first = {}, False, True
    with httpx.stream("POST", b["url"] + "/chat/completions", headers=headers, json=body, timeout=120) as r:
        if r.status_code != 200:
            r.read()
            text = r.text
            if r.status_code == 400 and _effort and "reason" in text.lower() and ("reasoning_effort" in body or "reasoning" in body):
                yield from oai_stream(bid, model, msgs, use_tools, effort, show_think, _effort=False)  # model can't take it
                return
            if use_tools and ("tool_use_failed" in text or (r.status_code == 400 and "tool" in text.lower())):
                try:
                    failed = (r.json().get("error") or {}).get("failed_generation", "")
                except ValueError:
                    failed = ""
                call = parse_failed_call(failed)
                if call:
                    yield "calls", [call]
                else:
                    yield from oai_stream(bid, model, msgs, False, effort, show_think, _effort)  # once more, no tools
                return
            raise down(r.status_code, text, r.headers)
        for ln in r.iter_lines():
            if not ln.startswith("data:"):
                continue
            data = ln[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get("error"):
                err = ev["error"]
                msg = str(err.get("message", err) if isinstance(err, dict) else err)[:300]
                if not started:
                    raise BrainDown("hit an error", 30)
                raise RuntimeError(f"{b['name']} stopped with an error: {msg}")
            if first and ev.get("model") and bid == "openrouter":
                yield "model", ev["model"]  # the free model OpenRouter picked
            first = False
            ch = ev.get("choices") or []
            if not ch:
                continue
            d = ch[0].get("delta") or {}
            thought = d.get("reasoning") or d.get("reasoning_content")
            if thought and show_think and isinstance(thought, str):
                yield "think", thought
            if d.get("content"):
                started = True
                yield "token", d["content"]
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "args": ""})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["name"] = fn.get("name") or c["name"]
                c["args"] += fn.get("arguments") or ""
    if calls:
        out = list(calls.values())
        for c in out:
            c["id"] = c["id"] or "call_" + secrets.token_hex(6)
        yield "calls", out


def to_anthropic(msgs):
    """OpenAI-style messages -> Claude's format (system text + alternating user/assistant blocks)."""
    system, out = [], []

    def add(role, blocks):
        blocks = [x for x in blocks if not (x.get("type") == "text" and not (x.get("text") or "").strip())]
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})

    for m in msgs:
        role, c = m["role"], m.get("content")
        if role == "system":
            system.append(c or "")
        elif role == "tool":
            add("user", [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": c or "(no result)"}])
        elif role == "assistant":
            add("assistant", list(m["_blocks"]) if m.get("_blocks") else [{"type": "text", "text": c or ""}])
        elif isinstance(c, list):
            blocks = []
            for p in c:
                if p.get("type") == "text":
                    blocks.append({"type": "text", "text": p["text"]})
                elif p.get("type") == "image_url":
                    head, data = p["image_url"]["url"].split(";base64,", 1)
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": head[5:], "data": data}})
            add("user", blocks)
        else:
            add("user", [{"type": "text", "text": c or ""}])
    while out and out[0]["role"] != "user":
        out.pop(0)
    return "\n\n".join(system), out


def claude_stream(bid, model, msgs, use_tools, effort, show_think, _think=True):
    """Stream from Anthropic's Messages API, with web search tools and (in Thorough) extended thinking."""
    b = BRAINS[bid]
    system, conv = to_anthropic(msgs if use_tools else flatten(msgs))
    body = {"model": model, "max_tokens": 4096, "messages": conv, "stream": True}
    if system:
        body["system"] = system
    if use_tools:
        body["tools"] = [{"name": n, "description": d, "input_schema": {
            "type": "object", "properties": {a: {"type": "string"}}, "required": [a]}} for n, (_, a, d) in S.TOOLS.items()]
    thinking = _think and effort == "thorough"
    if thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 4000}
        body["max_tokens"] = 12000
    headers = {"x-api-key": brain_key(bid), "anthropic-version": "2023-06-01", "content-type": "application/json"}
    blocks, started = {}, False
    with httpx.stream("POST", b["url"] + "/messages", headers=headers, json=body, timeout=180) as r:
        if r.status_code != 200:
            r.read()
            if thinking and r.status_code == 400 and "thinking" in r.text.lower():
                yield from claude_stream(bid, model, msgs, use_tools, effort, show_think, _think=False)
                return
            raise down(r.status_code, r.text, r.headers)
        for ln in r.iter_lines():
            if not ln.startswith("data:"):
                continue
            try:
                ev = json.loads(ln[5:].strip())
            except ValueError:
                continue
            t = ev.get("type")
            if t == "content_block_start":
                cb = dict(ev.get("content_block") or {})
                if cb.get("type") == "tool_use":
                    cb["_json"] = ""
                blocks[ev.get("index", len(blocks))] = cb
            elif t == "content_block_delta":
                cb = blocks.setdefault(ev.get("index", 0), {"type": "text", "text": ""})
                d = ev.get("delta") or {}
                kind = d.get("type")
                if kind == "text_delta":
                    started = True
                    cb["text"] = cb.get("text", "") + d.get("text", "")
                    yield "token", d.get("text", "")
                elif kind == "thinking_delta":
                    cb["thinking"] = cb.get("thinking", "") + d.get("thinking", "")
                    if show_think:
                        yield "think", d.get("thinking", "")
                elif kind == "signature_delta":
                    cb["signature"] = cb.get("signature", "") + d.get("signature", "")
                elif kind == "input_json_delta":
                    cb["_json"] = cb.get("_json", "") + d.get("partial_json", "")
            elif t == "error":
                err = ev.get("error") or {}
                if not started:
                    raise BrainDown("is overloaded right now" if err.get("type") == "overloaded_error" else "hit an error", 30)
                raise RuntimeError("Claude stopped with an error: " + str(err.get("message", ""))[:200])
    raw, calls = [], []
    for i in sorted(blocks):
        cb, kind = blocks[i], blocks[i].get("type")
        if kind == "text" and cb.get("text"):
            raw.append({"type": "text", "text": cb["text"]})
        elif kind == "thinking":
            raw.append({"type": "thinking", "thinking": cb.get("thinking", ""), "signature": cb.get("signature", "")})
        elif kind == "redacted_thinking":
            raw.append({"type": "redacted_thinking", "data": cb.get("data", "")})
        elif kind == "tool_use":
            try:
                args = json.loads(cb.get("_json") or "{}")
            except ValueError:
                args = {}
            raw.append({"type": "tool_use", "id": cb["id"], "name": cb["name"], "input": args})
            calls.append({"id": cb["id"], "name": cb["name"], "args": json.dumps(args)})
    if calls:
        yield "raw", raw  # Claude needs its own blocks (including thinking) sent back with the tool results
        yield "calls", calls


# ---------- Ollama on this computer ----------
_LOCAL = {"checked": 0.0, "ok": False, "models": set()}


def local_ok():
    """Is Ollama running? Checked at most every 20 seconds. Also remembers which models are installed."""
    now = time.time()
    if now - _LOCAL["checked"] > 20:
        try:
            r = httpx.get(OLLAMA_URL + "/api/tags", timeout=1.5)
            _LOCAL["ok"] = r.status_code == 200
            if _LOCAL["ok"]:
                _LOCAL["models"] = {m["name"] for m in r.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            _LOCAL["ok"] = False
        _LOCAL["checked"] = now
    return _LOCAL["ok"]


def local_down():
    _LOCAL.update(ok=False, checked=time.time())


def local_model(preferred):
    """Use the preferred local model if it is installed, otherwise an installed one. Returns (name, exact)."""
    local_ok()
    models = _LOCAL["models"]
    if not models or preferred in models:
        return preferred, True
    for m in (LOCAL_FALLBACK, *sorted(models)):
        if m in models and "embed" not in m:
            return m, False
    return preferred, True


def route(prev, text):
    """Ask the small local model whether this message needs a web search."""
    ctx = "\n".join(f"- {p[:300]}" for p in prev) or "- (none)"
    try:
        r = ollama.chat(
            model=LOCAL_FALLBACK,
            messages=[{"role": "system", "content": S.ROUTER_PROMPT},
                      {"role": "user", "content": f"Previous user messages:\n{ctx}\n\nCurrent message: {text[:1500]}"}],
            options={"temperature": 0, "num_predict": 4}, keep_alive="30m",
        )
        return "SEARCH" in r.message.content.upper()
    except Exception:
        return False
