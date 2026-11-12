import base64
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
from pydantic import BaseModel, Field

import brains as B  # the AI providers, their limits and the Ollama fallback
import server as S  # tools, prompts and document reading

app = FastAPI()
Auth = Header(default="", alias="Authorization")

# ---------- rate limits ----------
HITS = {}


def too_many(key, limit, window=3600):
    now = time.time()
    if len(HITS) > 20000:  # forget old keys so memory stays small
        for k in [k for k, v in HITS.items() if not v or now - v[-1] > window]:
            HITS.pop(k, None)
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


# ---------- AI brains (the providers live in brains.py) ----------
# How much chat history (in characters) each kind of model gets. Roughly 4 characters per token.
CLOUD_BUDGET, CLOUD_DOC_CAP = 28000, 20000
LOCAL_BUDGET, LOCAL_DOC_CAP = 14000, 8000
LOCAL_CTX = 8192
STYLE = {"turbo": " Keep answers short and to the point.", "balanced": " Keep answers clear and concise.",
         "thorough": " Reason carefully and give a thorough, well-organised answer."}
GROQ_WHISPER = "whisper-large-v3-turbo"  # speech to text (voice typing on phones, Firefox, and the high-accuracy option)


def groq_key():
    return B.brain_key("groq")


# ---------- database ----------
def db():
    c = sqlite3.connect("mazzi.db", timeout=15)
    c.row_factory = sqlite3.Row
    return c


with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT UNIQUE, pw TEXT, guest INTEGER DEFAULT 0);
    CREATE TABLE IF NOT EXISTS tokens(token TEXT PRIMARY KEY, user_id INTEGER);
    CREATE TABLE IF NOT EXISTS chats(id INTEGER PRIMARY KEY, user_id INTEGER, title TEXT DEFAULT 'New chat', updated REAL);
    CREATE TABLE IF NOT EXISTS msgs(id INTEGER PRIMARY KEY, chat_id INTEGER, role TEXT, content TEXT, image TEXT, sources TEXT);
    CREATE TABLE IF NOT EXISTS projects(id INTEGER PRIMARY KEY, user_id INTEGER, name TEXT, instructions TEXT DEFAULT '');
    CREATE INDEX IF NOT EXISTS msgs_chat ON msgs(chat_id);
    CREATE INDEX IF NOT EXISTS chats_user ON chats(user_id);
    """)
    for table, col in (("chats", "project_id INTEGER"), ("users", "email TEXT"),
                       ("msgs", "images TEXT"), ("msgs", "files TEXT"), ("msgs", "model TEXT")):
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
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


def bearer(auth):
    return (auth or "").replace("Bearer ", "").strip()


def user_of(auth):
    with db() as c:
        u = c.execute("SELECT u.* FROM tokens t JOIN users u ON u.id=t.user_id WHERE t.token=?", (bearer(auth),)).fetchone()
    if not u:
        raise HTTPException(401, "Please sign in")
    return u


def own(c, u, cid):
    r = c.execute("SELECT * FROM chats WHERE id=? AND user_id=?", (cid, u["id"])).fetchone()
    if not r:
        raise HTTPException(404, "Chat not found")
    return r


def line(o):
    return json.dumps(o) + "\n"


def row_images(r):
    if r["images"]:
        return json.loads(r["images"])
    return [r["image"]] if r["image"] else []


def row_files(r):
    return json.loads(r["files"]) if r["files"] else []


# ---------- accounts ----------
class Cred(BaseModel):
    name: str = Field(max_length=60)
    password: str = Field(max_length=200)


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
    if not u or not u["pw"] or not check_pw(b.password, u["pw"]):
        raise HTTPException(401, "Wrong username or password")
    return {"token": new_token(u["id"])}


@app.post("/api/guest")
def guest(request: Request):
    if too_many("guest:" + client_ip(request), 10):
        raise HTTPException(429, "Too many guest sessions from this network. Please try again later.")
    with db() as c:
        uid = c.execute("INSERT INTO users(name, pw, guest) VALUES(?,?,1)", ("guest-" + secrets.token_hex(4), "")).lastrowid
    return {"token": new_token(uid)}


@app.post("/api/logout")
def logout(auth: str = Auth):
    with db() as c:
        c.execute("DELETE FROM tokens WHERE token=?", (bearer(auth),))
    return {"ok": True}


@app.get("/api/me")
def me(auth: str = Auth):
    u = user_of(auth)
    return {"name": u["name"], "guest": bool(u["guest"]), "cloud": B.any_cloud(), "local": B.local_ok()}


@app.get("/api/brains")
def brains(auth: str = Auth):
    """Every AI brain and whether it can answer right now (shown in the brain picker and in Settings)."""
    user_of(auth)
    out = [{"id": "auto", "name": "Auto", "about": "Picks the best brain available and switches when one hits its limit.",
            "ready": True, "free": True}]
    for bid in B.CHAIN:
        info, has_key = B.BRAINS[bid], bool(B.brain_key(bid))
        r = B.resting(bid) if has_key else None
        out.append({"id": bid, "name": info["name"], "about": info["about"], "free": info["free"], "ready": has_key,
                    "resting": int(r[0] - time.time()) if r else 0, "why": r[1] if r else "",
                    "signup": info["signup"], "keyfile": f"{bid}_key.txt", "env": info["env"][0]})
    out.append({"id": "local", "name": "On this computer", "about": "Ollama. Private, works offline, slower.",
                "free": True, "ready": B.local_ok()})
    return out


# ---------- chats ----------
@app.get("/api/chats")
def chats(project: int | None = None, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        if project:
            rows = c.execute("SELECT id,title,updated FROM chats WHERE user_id=? AND project_id=? ORDER BY updated DESC",
                             (u["id"], project)).fetchall()
        else:
            rows = c.execute("SELECT id,title,updated FROM chats WHERE user_id=? AND project_id IS NULL ORDER BY updated DESC",
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
        rows = c.execute("SELECT id,role,content,image,images,files,sources,model FROM msgs WHERE chat_id=? ORDER BY id",
                         (cid,)).fetchall()
    return [{"id": r["id"], "role": r["role"], "content": r["content"], "images": row_images(r),
             "files": [{"name": f["name"], "chars": len(f.get("text", ""))} for f in row_files(r)],
             "sources": json.loads(r["sources"] or "[]"), "model": r["model"] or ""} for r in rows]


class Title(BaseModel):
    title: str = Field(max_length=200)


@app.patch("/api/chats/{cid}")
def rename(cid: int, b: Title, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        own(c, u, cid)
        c.execute("UPDATE chats SET title=? WHERE id=?", (b.title.strip()[:80] or "New chat", cid))
    return {"ok": True}


@app.delete("/api/chats/{cid}")
def delete_chat(cid: int, auth: str = Auth):
    u = user_of(auth)
    with db() as c:
        own(c, u, cid)
        c.execute("DELETE FROM msgs WHERE chat_id=?", (cid,))
        c.execute("DELETE FROM chats WHERE id=?", (cid,))
    return {"ok": True}


# ---------- projects ----------
class Proj(BaseModel):
    name: str = Field(max_length=200)
    instructions: str = Field("", max_length=8000)


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


# ---------- files and voice ----------
class Upload(BaseModel):
    name: str = Field(max_length=300)
    data: str  # base64


MAX_DOC_CHARS = 200_000


@app.post("/api/extract")
def extract(b: Upload, auth: str = Auth):
    u = user_of(auth)
    if too_many("file:%s" % u["id"], 30 if u["guest"] else 120):
        raise HTTPException(429, "Too many uploads this hour. Please try again later.")
    if len(b.data) > 14_000_000:
        raise HTTPException(413, "Files can be up to 10 MB.")
    try:
        text = S.extract_text(b.name, b.data)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(400, "Nellow could not read this file.")
    return {"name": b.name[:120], "text": text[:MAX_DOC_CHARS], "chars": len(text), "truncated": len(text) > MAX_DOC_CHARS}


class Audio(BaseModel):
    data: str  # base64
    mime: str = Field("audio/webm", max_length=100)
    lang: str = Field("", max_length=20)  # e.g. "en-US"; empty lets Whisper detect the language


@app.post("/api/transcribe")
def transcribe(b: Audio, auth: str = Auth):
    """Speech to text through Groq Whisper, for browsers without built-in dictation (for example Firefox)."""
    u = user_of(auth)
    if too_many("voice:%s" % u["id"], 40 if u["guest"] else 150):
        raise HTTPException(429, "Too many voice messages this hour. Please try again later.")
    if not groq_key():
        raise HTTPException(400, "Voice typing in this browser needs the cloud key. Try Chrome or Edge.")
    if len(b.data) > 26_000_000:
        raise HTTPException(413, "That recording is too long.")
    mime = b.mime.split(";")[0]
    ext = {"audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "mp4", "audio/mpeg": "mp3", "audio/wav": "wav"}.get(mime, "webm")
    form = {"model": GROQ_WHISPER, "response_format": "verbose_json", "temperature": "0"}
    lang = b.lang.split("-")[0].lower()
    if re.fullmatch(r"[a-z]{2}", lang):
        form["language"] = lang
    try:
        r = httpx.post(B.BRAINS["groq"]["url"] + "/audio/transcriptions", headers={"Authorization": "Bearer " + groq_key()},
                       data=form, files={"file": ("speech." + ext, base64.b64decode(b.data), mime)}, timeout=60)
    except httpx.HTTPError:
        raise HTTPException(502, "Could not reach the speech service.")
    if r.status_code == 429:
        raise HTTPException(429, "Voice typing has hit its limit for now. Try again in a minute.")
    if r.status_code != 200:
        raise HTTPException(502, "The speech service could not understand that recording. Please try again.")
    return {"text": clean_transcript(r.json())}


FILLER = {"thank you.", "thank you", "thanks for watching!", "thank you for watching.", "thank you for watching!",
          "thanks for watching.", "you", "bye.", "bye", "bye-bye."}


def clean_transcript(j):
    """Whisper invents words over silence (usually "Thank you."). Drop segments that run past the end of the
    audio, and stock phrases stretched over a long quiet stretch."""
    segs, dur = j.get("segments"), float(j.get("duration") or 0)
    if not segs:
        return (j.get("text") or "").strip()
    keep = []
    for s in segs:
        t = (s.get("text") or "").strip()
        start, end = float(s.get("start", 0)), float(s.get("end", 0))
        if not t or (dur and (start >= dur - 0.2 or end > dur + 1.5)):
            continue
        if t.lower() in FILLER and (end - start) / max(1, len(t.split())) > 4:
            continue
        keep.append(t)
    return " ".join(keep).strip()


# ---------- sending a message ----------
class Doc(BaseModel):
    name: str = Field(max_length=300)
    text: str = Field(max_length=MAX_DOC_CHARS)


class Send(BaseModel):
    message: str | None = Field(None, max_length=30000)
    images: list[str] = Field(default_factory=list, max_length=4)
    image: str | None = None  # older clients send a single image
    docs: list[Doc] = Field(default_factory=list, max_length=5)
    from_id: int | None = None  # delete this message and all after it (used by edit and retry)
    keep: bool = False  # when editing: keep the images and documents of the message being replaced
    effort: str = Field("turbo", max_length=20)  # turbo, balanced or thorough
    brain: str = Field("auto", max_length=20)  # auto, a brain id from brains.py, or local


def with_docs(content, files, cap):
    if not files:
        return content
    each = max(1500, cap // len(files))
    parts = [content]
    for f in files:
        t = f.get("text", "")
        if len(t) > each:
            t = t[:each] + "\n[…the rest of this document was cut to fit]"
        parts.append(f"[Document: {f['name']}]\n{t}\n[End of document: {f['name']}]")
    return "\n\n".join(parts)


def build_history(rows, system, budget, doc_cap):
    """System prompt + as much recent history as fits the budget. Returns (history, latest user text)."""
    last = rows[-1]
    latest = with_docs(last["content"], row_files(last), doc_cap)
    used = len(system) + len(latest)
    older = []
    for r in reversed(rows[:-1]):
        text = r["content"]
        if r["role"] == "user":
            text = with_docs(text, row_files(r), 3000)
            if row_images(r):
                text = "[The user attached an image] " + text
        if used + len(text) > budget:
            break
        older.append({"role": r["role"], "content": text})
        used += len(text)
    older.reverse()
    return [{"role": "system", "content": system}] + older, latest


CITE_MARK = re.compile(r"【[^】\n]{0,40}】")  # gpt-oss sometimes leaks its internal citation markers


def short_title(text):
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= 48:
        return t
    cut = t[:48].rsplit(" ", 1)[0]
    return (cut or t[:48]) + "…"


@app.post("/api/chats/{cid}/send")
def send(cid: int, b: Send, auth: str = Auth):
    u = user_of(auth)
    if too_many("msg:%s" % u["id"], 20 if u["guest"] else 60):
        raise HTTPException(429, "You have reached the hourly message limit. Please try again later.")
    images = [i for i in (b.images or ([b.image] if b.image else [])) if i][:4]
    if any(len(i) > 4_000_000 for i in images):
        raise HTTPException(413, "Images can be up to 3 MB each.")
    docs = [{"name": d.name[:120], "text": d.text} for d in b.docs]
    user_msg_id = None
    with db() as c:
        chat = own(c, u, cid)
        p = c.execute("SELECT instructions FROM projects WHERE id=?", (chat["project_id"],)).fetchone() if chat["project_id"] else None
        ins = p["instructions"] if p else ""
        if b.from_id:
            if b.keep and not images and not docs:
                old = c.execute("SELECT image,images,files FROM msgs WHERE chat_id=? AND id=? AND role='user'",
                                (cid, b.from_id)).fetchone()
                if old:
                    images, docs = row_images(old), row_files(old)
            c.execute("DELETE FROM msgs WHERE chat_id=? AND id>=?", (cid, b.from_id))
        if b.message or images or docs:
            content = (b.message or "").strip() or ("Summarize this document." if docs else "Describe this image.")
            user_msg_id = c.execute(
                "INSERT INTO msgs(chat_id,role,content,images,files) VALUES(?,?,?,?,?)",
                (cid, "user", content, json.dumps(images) if images else None, json.dumps(docs) if docs else None)).lastrowid
            if chat["title"] == "New chat":
                title = b.message or (docs[0]["name"] if docs else "Image")
                c.execute("UPDATE chats SET title=? WHERE id=?", (short_title(title), cid))
        rows = c.execute("SELECT role,content,image,images,files FROM msgs WHERE chat_id=? ORDER BY id", (cid,)).fetchall()
    if not rows or rows[-1]["role"] != "user":
        raise HTTPException(400, "Nothing to answer")

    effort = b.effort if b.effort in B.EFFORTS else "turbo"
    show_think = effort == "thorough"
    system = S.system_prompt(None if u["guest"] else u["name"]) + STYLE[effort]
    if ins:
        system += "\n\nThe user set these instructions for this project. Follow them:\n" + ins
    last = rows[-1]
    last_images = row_images(last)
    has_docs = bool(row_files(last))
    prev = [r["content"] for r in rows[:-1] if r["role"] == "user"][-2:]

    def cloud(bid, sources):
        brain = B.BRAINS[bid]
        stream = B.claude_stream if brain["api"] == "anthropic" else B.oai_stream
        hist, latest = build_history(rows, system, CLOUD_BUDGET, CLOUD_DOC_CAP)
        if last_images:
            parts = [{"type": "text", "text": latest}] + [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + i}} for i in last_images]
            msgs = hist + [{"role": "user", "content": parts}]
            yield "status", "Looking at the image…"
            yield from B.model_chain(bid, brain.get("vision") or brain["models"][effort],
                                     lambda m: stream(bid, m, msgs, False, effort, False))
            return
        msgs = hist + [{"role": "user", "content": latest}]
        for rnd in range(6):
            text, calls, raw = "", [], None
            for kind, val in B.model_chain(bid, brain["models"][effort],
                                           lambda m: stream(bid, m, msgs, rnd < 3, effort, show_think)):
                if kind == "calls":
                    calls = val
                elif kind == "raw":
                    raw = val
                else:
                    if kind == "token":
                        text += val
                    elif kind == "reset":
                        text = ""
                    yield kind, val
            if not calls:
                return
            asked = {"role": "assistant", "content": text or None, "tool_calls": [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                for c in calls]}
            if raw:
                asked["_blocks"] = raw
            msgs.append(asked)
            for c in calls:
                try:
                    args = json.loads(c["args"] or "{}")
                except ValueError:
                    args = {}
                yield "status", tool_status(c["name"], args)
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": S.run_tool(c["name"], args, sources)[:2500]})
        yield "token", "I got stuck. Try rephrasing your question."

    def local(sources):
        hist, latest = build_history(rows, system, LOCAL_BUDGET, LOCAL_DOC_CAP)
        opts = {"num_ctx": LOCAL_CTX}
        if last_images:
            yield "model", S.VISION_MODEL
            yield "status", "Looking at the image…"
            msgs = hist + [{"role": "user", "content": latest, "images": last_images}]
            for ch in ollama.chat(model=S.VISION_MODEL, messages=msgs, stream=True, options=opts, keep_alive="10m"):
                if ch.message.content:
                    yield "token", ch.message.content
            return
        want = B.LOCAL_MODELS[effort]
        model, exact = B.local_model(want)
        think = show_think and exact and want.startswith("qwen3")
        yield "model", model
        msgs = hist + [{"role": "user", "content": latest}]
        tools = None if has_docs or not B.route(prev, last["content"]) else S.TOOL_SPECS
        if tools:
            yield "status", "Searching the web…"
        for _ in range(6):
            text, calls = "", []
            extra = {"think": True} if think else {}
            for ch in ollama.chat(model=model, messages=msgs, tools=tools, stream=True,
                                  options=opts, keep_alive="30m", **extra):
                m = ch.message
                if getattr(m, "thinking", None):
                    yield "think", m.thinking
                if m.tool_calls:
                    calls.extend(m.tool_calls)
                if m.content:
                    text += m.content
                    if not text.lstrip().startswith("{"):
                        yield "token", m.content
            if calls:
                msgs.append({"role": "assistant", "content": text, "tool_calls": calls})
                for call in calls:
                    n, args = call.function.name, dict(call.function.arguments or {})
                    yield "status", tool_status(n, args)
                    msgs.append({"role": "tool", "tool_name": n, "content": S.run_tool(n, args, sources)})
                tools = S.TOOL_SPECS
                continue
            held = text.lstrip().startswith("{")
            leaked = S.parse_text_tool_call(text) if held else None
            if leaked:
                n, args = leaked
                yield "status", tool_status(n, args)
                msgs.append({"role": "assistant", "content": text})
                msgs.append({"role": "tool", "tool_name": n, "content": S.run_tool(n, args, sources)})
                tools = S.TOOL_SPECS
                continue
            if held:
                yield "token", text
            return
        yield "token", "I got stuck. Try rephrasing your question."

    def gen():
        sources, answer, saved, current, used = [], "", False, "", ""
        try:
            yield line({"type": "meta", "user_id": user_msg_id})
            order = B.brain_order(b.brain)
            attempts = [(bid, (lambda bid=bid: cloud(bid, sources))) for bid in order]
            attempts.append(("local", lambda: local(sources)))
            if b.brain in B.BRAINS and b.brain not in order:
                yield line({"type": "status", "text": f"{B.name_of(b.brain)} isn't available right now, "
                                                      f"so {B.name_of(attempts[0][0])} is answering…"})
            for i, (bid, run) in enumerate(attempts):
                current = bid
                try:
                    for ev, val in run():
                        if ev == "model":
                            used = f"{B.name_of(bid)} · {B.short_model(val)}"
                            yield line({"type": "model", "text": used})
                            continue
                        if ev == "reset":
                            answer = ""
                            yield line({"type": "reset"})
                            continue
                        if ev == "token":
                            answer += val
                        yield line({"type": ev, "text": val})
                    break
                except (B.BrainDown, httpx.HTTPError) as e:
                    if bid == "local":
                        raise
                    why = e.why if isinstance(e, B.BrainDown) else "couldn't be reached"
                    B.rest(bid, e.wait if isinstance(e, B.BrainDown) else 30, why)
                    nxt = attempts[i + 1][0]
                    if nxt == "local" and not B.local_ok():
                        raise CloudBusy()
                    answer = ""
                    sources.clear()
                    yield line({"type": "reset"})
                    yield line({"type": "status", "text": f"{B.name_of(bid)} {why}, switching to {B.name_of(nxt)}…"})
            urls = list(dict.fromkeys(sources))[:8]
            answer = CITE_MARK.sub("", answer)
            with db() as c:
                mid = c.execute("INSERT INTO msgs(chat_id,role,content,sources,model) VALUES(?,?,?,?,?)",
                                (cid, "assistant", answer, json.dumps(urls), used)).lastrowid
                c.execute("UPDATE chats SET updated=? WHERE id=?", (time.time(), cid))
            saved = True
            yield line({"type": "sources", "urls": urls})
            yield line({"type": "done", "id": mid})
        except Exception as e:
            local_down = current == "local" and is_local_down(e)
            if local_down:
                B.local_down()
            err = {"type": "error", "text": friendly_error(e, local_down)}
            if local_down and b.brain == "local" and B.any_cloud():
                err["action"] = "cloud"  # the page offers a one-click switch to the cloud brains
            yield line(err)
            yield line({"type": "done", "id": None})
        finally:
            if not saved and answer.strip():  # the user pressed stop: keep what was written so far
                try:
                    with db() as c:
                        c.execute("INSERT INTO msgs(chat_id,role,content,sources,model) VALUES(?,?,?,?,?)",
                                  (cid, "assistant", answer, json.dumps(list(dict.fromkeys(sources))[:8]), used))
                        c.execute("UPDATE chats SET updated=? WHERE id=?", (time.time(), cid))
                except Exception:
                    pass

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def tool_status(name, args):
    q = next((v for v in (args or {}).values() if isinstance(v, str)), "")
    if name == "read_page":
        host = re.sub(r"^https?://(www\.)?", "", q).split("/")[0]
        return f"Reading {host}…" if host else "Reading a page…"
    if name == "news_search":
        return f"Searching news for “{q[:60]}”…" if q else "Searching the news…"
    return f"Searching the web for “{q[:60]}”…" if q else "Searching the web…"


class CloudBusy(Exception):
    """Every cloud brain is at its limit and Ollama is not running, so nothing can answer right now."""


def is_local_down(e):
    s = str(e)
    return isinstance(e, (ConnectionError, httpx.ConnectError)) or "Failed to connect to Ollama" in s


def friendly_error(e, local_down=False):
    s = str(e)
    if isinstance(e, (CloudBusy, B.BrainDown)):
        return ("Every AI brain is at its free limit right now and the local model (Ollama) isn't running to take over. "
                "Please try again in a minute.")
    if local_down:
        if B.any_cloud():
            return "The local model isn't running (Ollama is off). Switch to a cloud brain to get an answer now."
        return "No AI brain is available. Start Ollama, or add an API key (see Settings, AI brains)."
    if "not found" in s and "model" in s:
        return "A local model is missing. Run `ollama pull` for it, then try again. (" + s[:120] + ")"
    return "Something went wrong: " + s[:300]


@app.get("/")
def home():
    # no-cache: the browser must check for a newer page every time, so updates show up immediately
    return FileResponse("index.html", headers={"Cache-Control": "no-cache"})
