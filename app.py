import hashlib
import json
import os
import re
import secrets
import sqlite3
import time

import httpx
import ollama
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import server as S  # reuses your search tools, router and models from server.py

app = FastAPI()
Auth = Header(default="", alias="Authorization")

HITS = {}


def too_many(key, limit, window=3600):
    now = time.time()
    q = [t for t in HITS.get(key, []) if now - t < window]
    if len(q) >= limit:
        HITS[key] = q
        return True
    q.append(now)
    HITS[key] = q
    return False


def client_ip(request):
    fwd = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return fwd or (request.client.host if request.client else "unknown")

MODES = {
    "balanced": {"model": "llama3.2:3b", "think": False,
                 "style": " Keep answers clear and concise."},
    "thorough": {"model": "qwen3:4b", "think": True, "cloud_if_key": True, "effort": "medium"},
    "turbo": {"model": None, "think": False, "cloud": True, "effort": "low"},
}


def route(prev, text):
    ctx = "\n".join(f"- {p}" for p in prev) or "- (none)"
    try:
        r = ollama.chat(
            model=MODES["balanced"]["model"],
            messages=[{"role": "system", "content": S.ROUTER_PROMPT},
                      {"role": "user", "content": f"Previous user messages:\n{ctx}\n\nCurrent message: {text}"}],
            options={"temperature": 0, "num_predict": 4},
        )
        return "SEARCH" in r.message.content.upper()
    except Exception:
        return False


GROQ_MODEL = "openai/gpt-oss-120b"  # if Groq retires it, pick another from console.groq.com/docs/models
GROQ_FALLBACK = "openai/gpt-oss-20b"  # used automatically when the main model hits its per-minute limit


def groq_key():
    k = os.environ.get("GROQ_API_KEY", "")
    if not k and os.path.exists("groq_key.txt"):
        with open("groq_key.txt", encoding="utf-8-sig") as f:
            k = f.read()
    return k.strip().strip("\"'").strip()


def _spec(name, desc, arg):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {arg: {"type": "string"}}, "required": [arg]}}}


TOOL_SPECS = [
    _spec("web_search", "Search the web for general facts and information.", "query"),
    _spec("news_search", "Search news from the past week. Use for recent, latest, trending or breaking topics.", "query"),
    _spec("read_page", "Read the full text of a web page. Use after a search to get details.", "url"),
]


def flatten(msgs):
    out, notes = [], []
    for m in msgs:
        if m.get("role") == "tool":
            notes.append(m["content"])
        elif not m.get("tool_calls"):
            out.append(m)
    if notes:
        out.append({"role": "user", "content": "Search results gathered so far:\n\n" + "\n\n".join(notes) +
                    "\n\nUsing only these results, write your final answer now. Do not search again."})
    return out


def groq_stream(msgs, use_tools, effort="low", model=None):
    model = model or GROQ_MODEL
    body = {"model": model, "messages": msgs, "stream": True, "reasoning_effort": effort}
    if use_tools:
        body["tools"] = TOOL_SPECS
    else:
        body["messages"] = flatten(msgs)
    calls = {}
    with httpx.stream("POST", "https://api.groq.com/openai/v1/chat/completions",
                      headers={"Authorization": "Bearer " + groq_key()}, json=body, timeout=60) as r:
        if r.status_code != 200:
            r.read()
            if r.status_code in (413, 429) and model == GROQ_MODEL:
                yield from groq_stream(msgs, use_tools, effort, GROQ_FALLBACK)  # backup model has its own limit
                return
            if use_tools and "tool_use_failed" in r.text:
                yield from groq_stream(msgs, False, effort, model)  # retry once without tools
                return
            raise RuntimeError(f"Groq error {r.status_code}: {r.text[:200]}")
        started = False
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.strip() == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("error"):
                if model == GROQ_MODEL and not started:
                    yield from groq_stream(msgs, use_tools, effort, GROQ_FALLBACK)  # try the backup model
                    return
                err = ev["error"]
                raise RuntimeError("Groq error: " + str(err.get("message", err) if isinstance(err, dict) else err)[:300])
            if not ev.get("choices"):
                continue
            started = True
            d = ev["choices"][0].get("delta") or {}
            if d.get("reasoning"):
                yield "think", d["reasoning"]
            if d.get("content"):
                yield "token", d["content"]
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"id": "", "name": "", "args": ""})
                c["id"] = tc.get("id") or c["id"]
                fn = tc.get("function") or {}
                c["name"] = fn.get("name") or c["name"]
                c["args"] += fn.get("arguments") or ""
    if calls:
        yield "calls", list(calls.values())


# ---------- database ----------
def db():
    c = sqlite3.connect("mazzi.db")
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT UNIQUE, pw TEXT, guest INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS tokens(token TEXT PRIMARY KEY, user_id INTEGER);
    CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT DEFAULT 'New chat', updated REAL);
    CREATE TABLE IF NOT EXISTS msgs(id INTEGER PRIMARY KEY, chat_id INTEGER, role TEXT, content TEXT, image TEXT, sources TEXT);
    CREATE TABLE IF NOT EXISTS projects(id INTEGER PRIMARY KEY, user_id INTEGER, name TEXT, instructions TEXT DEFAULT '');
    """)
    try:
        c.execute("ALTER TABLE chats ADD COLUMN project_id INTEGER")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN email TEXT")
    except sqlite3.OperationalError:
        pass


def hash_pw(pw, salt=None):
    salt = salt or secrets.token_hex(16)
    h = hashlib.scrypt(pw.encode(), salt=salt.encode(), n=2**14, r=8, p=1).hex()
    return f"{salt}${h}"


def check_pw(pw, stored):
    salt, h = stored.split("$")
    return secrets.compare_digest(hash_pw(pw, salt).split("$")[1], h)


def new_token(uid):
    t = secrets.token_urlsafe(32)
    with db() as c:
        c.execute("INSERT INTO tokens VALUES(?,?)", (t, uid))
    return t


def user_of(auth):
    t = (auth or "").replace("Bearer ", "")
    with db() as c:
        u = c.execute("SELECT u.* FROM tokens t JOIN users u ON u.id=t.user_id WHERE t.token=?", (t,)).fetchone()
    if not u:
        raise HTTPException(401, "Please sign in")
    return u


def own(c, u, cid):
    r = c.execute("SELECT * FROM chats WHERE id=? AND user_id=?", (cid, u["id"])).fetchone()
    if not r:
        raise HTTPException(404, "Chat not found")
    return r


def sse(o):
    return json.dumps(o) + "\n"


# ---------- accounts ----------
class Cred(BaseModel):
    name: str
    password: str


@app.post("/api/register")
def register(b: Cred, request: Request):
    if too_many("reg:" + client_ip(request), 5):
        raise HTTPException(429, "Too many sign-ups from this network. Please try again later.")
    name = b.name.strip().lower()
    if not re.fullmatch(r"[a-z0-9_.-]{3,30}", name):
        raise HTTPException(400, "Username: 3 to 30 letters, numbers, dots, dashes or underscores")
    if len(b.password) < 6:
        raise HTTPException(400, "Password needs at least 6 characters")
    try:
        with db() as c:
            uid = c.execute("INSERT INTO users(name, pw) VALUES(?,?)", (name, hash_pw(b.password))).lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(400, "That username is taken")
    return {"token": new_token(uid)}


@app.post("/api/login")
def login(b: Cred, request: Request):
    if too_many("login:" + client_ip(request), 20):
        raise HTTPException(429, "Too many login attempts. Please wait a while and try again.")
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE name=? AND guest=0", (b.name.strip().lower(),)).fetchone()
    if not u or not check_pw(b.password, u["pw"]):
        raise HTTPException(401, "Wrong username or password")
    return {"token": new_token(u["id"])}


@app.post("/api/guest")
def guest(request: Request):
    if too_many("guest:" + client_ip(request), 10):
        raise HTTPException(429, "Too many guest sessions from this network. Please try again later.")
    with db() as c:
        uid = c.execute("INSERT INTO users(name, pw, guest) VALUES(?,?,1)", ("guest-" + secrets.token_hex(4), "")).lastrowid
    return {"token": new_token(uid)}


@app.get("/api/me")
def me(auth: str = Auth):
    u = user_of(auth)
    return {"name": u["name"], "guest": bool(u["guest"])}


# ---------- chats ----------
@app.get("/api/chats")
def chats(project: int | None = None, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        if project:
            rows = c.execute("SELECT id,title FROM chats WHERE user_id=? AND project_id=? ORDER BY updated DESC",
                             (u["id"], project)).fetchall()
        else:
            rows = c.execute("SELECT id,title FROM chats WHERE user_id=? AND project_id IS NULL ORDER BY updated DESC",
                             (u["id"],)).fetchall()
    return [dict(r) for r in rows]


class NewChat(BaseModel):
    project_id: int | None = None


@app.post("/api/chats")
def new_chat(b: NewChat = NewChat(), auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        if b.project_id and not c.execute("SELECT 1 FROM projects WHERE id=? AND user_id=?",
                                          (b.project_id, u["id"])).fetchone():
            raise HTTPException(404, "Project not found")
        cid = c.execute("INSERT INTO chats(user_id, project_id, updated) VALUES(?,?,?)",
                        (u["id"], b.project_id, time.time())).lastrowid
    return {"id": cid}


@app.get("/api/chats/{cid}")
def get_chat(cid: int, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        own(c, u, cid)
        rows = c.execute("SELECT id,role,content,image,sources FROM msgs WHERE chat_id=? ORDER BY id", (cid,)).fetchall()
    return [{**dict(r), "sources": json.loads(r["sources"] or "[]")} for r in rows]


class Title(BaseModel):
    title: str


@app.patch("/api/chats/{cid}")
def rename(cid: int, b: Title, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        own(c, u, cid)
        c.execute("UPDATE chats SET title=? WHERE id=?", (b.title.strip()[:80] or "New chat", cid))
    return {"ok": True}


@app.delete("/api/chats/{cid}")
def delete(cid: int, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        own(c, u, cid)
        c.execute("DELETE FROM msgs WHERE chat_id=?", (cid,))
        c.execute("DELETE FROM chats WHERE id=?", (cid,))
    return {"ok": True}


# ---------- projects ----------
class Proj(BaseModel):
    name: str
    instructions: str = ""


@app.get("/api/projects")
def projects(auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        rows = c.execute("SELECT id,name,instructions FROM projects WHERE user_id=? ORDER BY id", (u["id"],)).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/projects")
def add_project(b: Proj, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        pid = c.execute("INSERT INTO projects(user_id,name,instructions) VALUES(?,?,?)",
                        (u["id"], b.name.strip()[:60] or "Project", b.instructions[:4000])).lastrowid
    return {"id": pid}


@app.patch("/api/projects/{pid}")
def edit_project(pid: int, b: Proj, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        c.execute("UPDATE projects SET name=?, instructions=? WHERE id=? AND user_id=?",
                  (b.name.strip()[:60] or "Project", b.instructions[:4000], pid, u["id"]))
    return {"ok": True}


@app.delete("/api/projects/{pid}")
def del_project(pid: int, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        c.execute("UPDATE chats SET project_id=NULL WHERE project_id=? AND user_id=?", (pid, u["id"]))
        c.execute("DELETE FROM projects WHERE id=? AND user_id=?", (pid, u["id"]))
    return {"ok": True}


# ---------- sending a message ----------
class Send(BaseModel):
    message: str | None = None
    image: str | None = None
    from_id: int | None = None  # delete this message and all after it (used by edit and retry)
    effort: str = "balanced"


@app.post("/api/chats/{cid}/send")
def send(cid: int, b: Send, auth: str = Auth):
    u = user_of(auth)
    if too_many("msg:%s" % u["id"], 20 if u["guest"] else 60):
        raise HTTPException(429, "You have reached the hourly message limit. Please try again later.")
    user_msg_id = None
    with db() as c:
        chat = own(c, u, cid)
        p = c.execute("SELECT instructions FROM projects WHERE id=?", (chat["project_id"],)).fetchone() if chat["project_id"] else None
        ins = p["instructions"] if p else ""
        if b.from_id:
            c.execute("DELETE FROM msgs WHERE chat_id=? AND id>=?", (cid, b.from_id))
        if b.message or b.image:
            user_msg_id = c.execute("INSERT INTO msgs(chat_id,role,content,image) VALUES(?,?,?,?)",
                                    (cid, "user", b.message or "Describe this image.", b.image)).lastrowid
            if chat["title"] == "New chat" and b.message:
                c.execute("UPDATE chats SET title=? WHERE id=?", (b.message.strip()[:40], cid))
        rows = c.execute("SELECT role,content,image FROM msgs WHERE chat_id=? ORDER BY id", (cid,)).fetchall()
    if not rows or rows[-1]["role"] != "user":
        raise HTTPException(400, "Nothing to answer")

    hist = [S.new_history()[0]]
    if ins:
        hist[0] = {"role": "system", "content": hist[0]["content"] + "\n\nThe user set these instructions for this project. Follow them:\n" + ins}
    for r in rows[:-1]:
        tag = "[The user attached an image] " if r["image"] else ""
        hist.append({"role": r["role"], "content": tag + r["content"]})
    last = rows[-1]
    prev = [r["content"] for r in rows[:-1] if r["role"] == "user"][-2:]
    mode = MODES.get(b.effort, MODES["balanced"])
    hist[0] = {"role": "system", "content": hist[0]["content"] + mode.get("style", "")}

    def gen():
        with S.LOCK:
            try:
                yield sse({"type": "meta", "user_id": user_msg_id})
                S.SOURCES.clear()
                text = ""
                if last["image"]:
                    yield sse({"type": "status", "text": "Looking at the image…"})
                    msgs = hist + [{"role": "user", "content": last["content"], "images": [last["image"]]}]
                    for ch in ollama.chat(model=S.VISION_MODEL, messages=msgs, stream=True,
                                          options={"num_ctx": 4096}, keep_alive="10m"):
                        p = ch.message.content or ""
                        if p:
                            text += p
                            yield sse({"type": "token", "text": p})
                elif mode.get("cloud") or (mode.get("cloud_if_key") and groq_key()):
                    if not groq_key():
                        raise RuntimeError("No Groq key found. Save your key in a file named groq_key.txt in the MAZZI folder.")
                    msgs = hist + [{"role": "user", "content": last["content"]}]
                    for rnd in range(6):
                        text, calls = "", []
                        for kind, val in groq_stream(msgs, rnd < 3, mode.get("effort", "low")):
                            if kind == "token":
                                text += val
                                yield sse({"type": "token", "text": val})
                            elif kind == "think":
                                if mode.get("think"):
                                    yield sse({"type": "think", "text": val})
                            else:
                                calls = val
                        if not calls:
                            break
                        msgs.append({"role": "assistant", "content": text or None, "tool_calls": [
                            {"id": c["id"], "type": "function",
                             "function": {"name": c["name"], "arguments": c["args"] or "{}"}} for c in calls]})
                        for c in calls:
                            yield sse({"type": "status", "text": f"Using {c['name']}…"})
                            try:
                                args = json.loads(c["args"] or "{}")
                            except Exception:
                                args = {}
                            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": S.run_tool(c["name"], args)[:2500]})
                    else:
                        text = "I got stuck. Try rephrasing your question."
                        yield sse({"type": "token", "text": text})
                else:
                    msgs = hist + [{"role": "user", "content": last["content"]}]
                    tools = list(S.TOOLS.values()) if route(prev, last["content"]) else None
                    if tools:
                        yield sse({"type": "status", "text": "Searching the web…"})
                    for _ in range(6):
                        text, calls = "", []
                        extra = {"think": True} if mode["think"] else {}
                        for ch in ollama.chat(model=mode["model"], messages=msgs, tools=tools, stream=True,
                                              options={"num_ctx": 8192 if mode["think"] else 4096},
                                              keep_alive="30m", **extra):
                            m = ch.message
                            if getattr(m, "thinking", None):
                                yield sse({"type": "think", "text": m.thinking})
                            if m.tool_calls:
                                calls.extend(m.tool_calls)
                            if m.content:
                                text += m.content
                                if not text.lstrip().startswith("{"):
                                    yield sse({"type": "token", "text": m.content})
                        if calls:
                            msgs.append({"role": "assistant", "content": text, "tool_calls": calls})
                            for call in calls:
                                n = call.function.name
                                yield sse({"type": "status", "text": f"Using {n}…"})
                                msgs.append({"role": "tool", "tool_name": n, "content": S.run_tool(n, call.function.arguments)})
                            tools = list(S.TOOLS.values())
                            continue
                        held = text.lstrip().startswith("{")
                        leaked = S.parse_text_tool_call(text) if held else None
                        if leaked:
                            n, args = leaked
                            yield sse({"type": "status", "text": f"Using {n}…"})
                            msgs.append({"role": "assistant", "content": text})
                            msgs.append({"role": "tool", "tool_name": n, "content": S.run_tool(n, args)})
                            tools = list(S.TOOLS.values())
                            continue
                        if held:
                            yield sse({"type": "token", "text": text})
                        break
                    else:
                        text = "I got stuck. Try rephrasing your question."
                        yield sse({"type": "token", "text": text})
                urls = list(dict.fromkeys(S.SOURCES))
                with db() as c:
                    mid = c.execute("INSERT INTO msgs(chat_id,role,content,sources) VALUES(?,?,?,?)",
                                    (cid, "assistant", text, json.dumps(urls))).lastrowid
                    c.execute("UPDATE chats SET updated=? WHERE id=?", (time.time(), cid))
                yield sse({"type": "sources", "urls": urls})
                yield sse({"type": "done", "id": mid})
            except Exception as e:
                yield sse({"type": "token", "text": f"Something went wrong: {e}"})
                yield sse({"type": "done", "id": None})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/theme.css")
def theme():
    return FileResponse("theme.css", media_type="text/css")


@app.get("/")
def home():
    return FileResponse("index.html")