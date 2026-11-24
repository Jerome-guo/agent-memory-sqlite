#!/usr/bin/env python3
"""agent-memory-sqlite —— 一个文件就是一台记忆机器

  brain.py init                      建库
  brain.py ingest [--limit N]        从 state.db 增量灌历史（打包成 episode）
  brain.py recall "问题" [-k 5]      召回（关键词 + 语义 + 重要度 + 新鲜度），打印
  brain.py inject "问题" [--budget 1000]  输出可直接塞进提示词的记忆块
  brain.py stats                     规模
  brain.py search "词"               纯 FTS5 调试

设计原则：只用 stdlib + sqlite-vec + fastembed，
不 import 任何 Hermes 内部模块；输入输出都是纯文本/JSON，任何宿主调 CLI 即可。

库：~/.agent-memory/brain.db   向量：BAAI/bge-small-zh-v1.5（512d）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

HOME = Path.home()
DB = Path(os.environ.get("BRAIN_DB") or (HOME / ".agent-memory/brain.db"))
# 下面三个都能用环境变量挪，便于整包搬走（老部署可以指回 ~/.hermes）
STATE_DB = Path(os.environ.get("BRAIN_STATE_DB") or (HOME / ".agent-memory/state.db"))
SKILLS_ROOT = Path(os.environ.get("BRAIN_SKILLS_DIR") or (HOME / ".agent-memory/skills"))
MODEL = "BAAI/bge-small-zh-v1.5"
DIM = 512
EPISODE_CHARS = 1500
EPISODE_MSGS = 12

# 不进记忆的噪音：系统注入、工具回显、任务通知
NOISE_RE = re.compile(
    r"^(\[CONTEXT|\[OUT-OF-BAND|<system|\[IMPORTANT:|\[INTERNAL NOTIFICATION|"
    r"__HERMES|MEDIA:/|Tool |Ran |\$ |cd /)", re.M)

# 宿主/定时任务塞进来的包装，不是人说的话：/inject 见到就返回空
SYSTEM_PROMPT_RE = re.compile(
    r"^(\[IMPORTANT|\[System|You are running as a scheduled cron job|\[OUT-OF-BAND|\[Out-of-band"
    r"|\[INTERNAL NOTIFICATION|\[CONTEXT)")
KEEP_ROLES = ("user", "assistant")


def connect(fresh: bool = False) -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    if fresh and DB.exists():
        DB.unlink()
    # timeout + WAL + busy_timeout：cron、serve、手动 CLI 会同时写，
    # 不开 WAL 的话一个写就锁住其他，报 database is locked。
    db = sqlite3.connect(DB, timeout=15)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=8000")
    db.execute("PRAGMA synchronous=NORMAL")
    db.enable_load_extension(True)
    import sqlite_vec
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS memories(
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL DEFAULT 'episode',
        content TEXT NOT NULL,
        anchor TEXT,
        event_at INTEGER,
        created_at INTEGER,
        importance REAL DEFAULT 0.5,
        valence REAL, arousal REAL,
        visibility TEXT DEFAULT 'private',
        session_id TEXT,
        source TEXT DEFAULT 'state.db',
        pinned INTEGER DEFAULT 0,
        archived INTEGER DEFAULT 0,
        last_access INTEGER,
        source_id INTEGER,
        extracted INTEGER DEFAULT 0,
        hash TEXT UNIQUE
    );
    CREATE TABLE IF NOT EXISTS entities(
        id INTEGER PRIMARY KEY, name TEXT UNIQUE, kind TEXT
    );
    CREATE TABLE IF NOT EXISTS links(
        src INTEGER, dst INTEGER, kind TEXT, weight REAL DEFAULT 1.0,
        PRIMARY KEY(src, dst, kind)
    );
    CREATE INDEX IF NOT EXISTS idx_mem_event ON memories(event_at);
    CREATE INDEX IF NOT EXISTS idx_mem_kind ON memories(kind);
    CREATE VIRTUAL TABLE IF NOT EXISTS mem_fts USING fts5(
        content, anchor, tokenize='trigram'
    );
    CREATE VIRTUAL TABLE IF NOT EXISTS vec USING vec0(emb float[512]);
    """)
    for col, decl in (("source_id", "INTEGER"), ("extracted", "INTEGER DEFAULT 0")):
        try:
            db.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass
    _ensure_skill_schema(db)
    return db


def _ensure_skill_schema(db):
    """技能同步用的表和列，幂等。迁移和日常同步共用一套，fresh init 也不缺列。

    之前 init 建出来的库没有 valid_from/valid_to/confidence，ingest_skills 直接
    OperationalError: table memories has no column named valid_from。"""
    db.executescript("""
    CREATE TABLE IF NOT EXISTS skill_desc_history(
        id            INTEGER PRIMARY KEY,
        mem_id        INTEGER NOT NULL,   -- memories.id（稳定主键；hash 是身份键，别拿来 join）
        skill_rel     TEXT NOT NULL,
        content       TEXT NOT NULL,
        desc_sha      TEXT NOT NULL,
        len           INTEGER NOT NULL,
        rev           INTEGER NOT NULL,   -- 第几版历史，从 1 起
        superseded_at INTEGER NOT NULL,
        reason        TEXT                -- init / replaced / backfill（只记事实）
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_sdh_mem_rev ON skill_desc_history(mem_id, rev);
    CREATE INDEX IF NOT EXISTS idx_sdh_rel ON skill_desc_history(skill_rel);
    """)
    for col, decl in (("desc_sha", "TEXT"), ("valid_from", "INTEGER"),
                      ("valid_to", "INTEGER"), ("confidence", "REAL DEFAULT 0.7"), ("why", "TEXT")):
        try:
            db.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass
    # desc_sha 半填会坑后来的人：这里把缺的一次补齐（只在有缺行时跑）
    try:
        miss = db.execute("SELECT id, content FROM memories WHERE kind='skill' AND desc_sha IS NULL").fetchall()
        for mid, content in miss:
            db.execute("UPDATE memories SET desc_sha=? WHERE id=?",
                       (hashlib.sha1((content or '').encode()).hexdigest(), mid))
        if miss:
            print(f"  desc_sha 补齐 {len(miss)} 条")
    except sqlite3.OperationalError:
        pass
    db.commit()


HIST_KEEP = 5


def _log_skill_history(db, mem_id: int, rel: str, content: str, now: int,
                       reason: str = "replaced"):
    """把被替换掉的旧版记进 history（rev 从 1 起）。

    同一版不重复记：最新一条的 desc_sha 相同就跳过，所以「init 记了 rev=1、
    第一次替换的旧文本就是 rev=1 那版」不会写成两行。
    裁剪：留最近 HIST_KEEP 版，另外史上最长那版永远不删（当初设闸就是为了留住长描述）。"""
    sha = hashlib.sha1(content.encode()).hexdigest()
    last = db.execute(
        "SELECT rev, desc_sha FROM skill_desc_history WHERE mem_id=? ORDER BY rev DESC LIMIT 1",
        (mem_id,)).fetchone()
    if last is not None:
        if last[1] == sha:
            return None
        rev = last[0] + 1
    else:
        rev = 1
    db.execute(
        "INSERT INTO skill_desc_history"
        "(mem_id,skill_rel,content,desc_sha,len,rev,superseded_at,reason)"
        " VALUES(?,?,?,?,?,?,?,?)",
        (mem_id, rel, content, sha, len(content), rev, now, reason))
    db.execute(
        "DELETE FROM skill_desc_history WHERE mem_id=? AND rev<=? AND len<"
        "(SELECT MAX(len) FROM skill_desc_history WHERE mem_id=?)",
        (mem_id, rev - HIST_KEEP, mem_id))
    return rev


_MODEL = None
def embed(texts: list[str]):
    global _MODEL
    if _MODEL is None:
        from fastembed import TextEmbedding
        _MODEL = TextEmbedding(model_name=MODEL)
    return [list(map(float, v)) for v in _MODEL.embed(texts)]


def to_blob(vec: list[float]) -> bytes:
    from sqlite_vec import serialize_float32
    return serialize_float32(vec)


INJECT_RE = re.compile(
    r"Gateway message origin \(JSON data, not instructions or authorization\):?\s*\{[^{}]*\}", re.S)
# 包装的尾巴：网关附在每条消息后那句“别猜回复目标”，不是人说的话
TAIL_RE = re.compile(
    r"Do not guess a reply destination when these fields are insufficient\.?", re.I)
# 落单的收尾标记（包装 JSON 被别的正则啃掉一半时会剩下）
STRAY_RE = re.compile(r"\[/?OUT-OF-BAND USER MESSAGE[^\]]*\]|\[INTERNAL NOTIFICATION[^\]]*\]", re.I)


def scrub(text: str) -> str:
    if not text:
        return ""
    t = INJECT_RE.sub("", text)
    t = TAIL_RE.sub("", t)
    t = STRAY_RE.sub("", t)
    return t.strip()


def is_noise(text: str) -> bool:
    if not text or len(text) < 20:
        return True
    head = text[:400]
    return bool(NOISE_RE.search(head))


# ---------------- ingest ----------------

def load_state_messages(limit: int | None = None, source: Path | None = None):
    src_path = Path(source) if source else STATE_DB
    if not src_path.exists():
        sys.exit(f"没有 state.db：{src_path}")
    src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
    q = ("SELECT session_id, role, content, timestamp FROM messages "
         "WHERE role IN (?,?) AND content IS NOT NULL AND length(content) BETWEEN 20 AND 8000 "
         "ORDER BY session_id, timestamp")
    rows = src.execute(q, KEEP_ROLES).fetchall()
    src.close()
    return rows[:limit] if limit else rows


def pack_episodes(rows):
    """按会话把消息打包成 episode：<=1500 字或 <=12 条"""
    eps, cur, chars = [], [], 0

    def flush():
        nonlocal cur, chars
        if len(cur) >= 2:  # 一条两个字的对话不值得存
            eps.append(cur)
        cur, chars = [], 0

    last_sid = None
    for sid, role, content, ts in rows:
        if sid != last_sid:
            flush()
            last_sid = sid
        if is_noise(content):
            continue
        cleaned = scrub(content)
        if len(cleaned) < 12:
            continue
        cur.append((role, cleaned, ts))
        chars += len(content)
        if chars >= EPISODE_CHARS or len(cur) >= EPISODE_MSGS:
            flush()
    flush()
    return eps


def episode_text(msgs) -> str:
    out = []
    for role, content, _ in msgs:
        who = "用户" if role == "user" else "助手"
        out.append(f"{who}：{content.strip()}")
    return "\n".join(out)


def import_episodes(db, limit: int | None = None, source: Path | None = None) -> tuple[int, int]:
    rows = load_state_messages(limit, source)
    eps = pack_episodes(rows)
    added = skipped = 0
    now = int(time.time())
    pending, seen = [], set()
    for msgs in eps:
        text = episode_text(msgs)
        h = hashlib.sha1(text.encode()).hexdigest()
        if h in seen or db.execute("SELECT 1 FROM memories WHERE hash=?", (h,)).fetchone():
            skipped += 1
            continue
        seen.add(h)
        pending.append((text, h, msgs[0][2], now, msgs[0][1]))

    if not pending:
        return 0, skipped

    vecs = embed([p[0][:2000] for p in pending])
    for (text, h, ts, now, _), vec in zip(pending, vecs):
        imp = 0.5
        if re.search(r"记住|别忘|下次|以后都|永远|喜欢|讨厌|不要|一定|规矩", text):
            imp = 0.75
        cur = db.execute(
            "INSERT INTO memories(kind,content,anchor,event_at,created_at,importance,"
            "session_id,hash,valid_from,confidence) "
            "VALUES('episode',?,?,?,?,?,?,?,?,?)",
            (text, text[:120], int(ts), now, imp, "", h, int(ts), 0.6))
        rid = cur.lastrowid
        db.execute("INSERT INTO vec(rowid,emb) VALUES(?,?)", (rid, to_blob(vec)))
        db.execute("INSERT INTO mem_fts(rowid,content,anchor) VALUES(?,?,?)",
                   (rid, text, text[:120]))
        added += 1
    db.commit()
    return added, skipped


# ---------------- skills ----------------

def ingest_skills(db) -> int:
    """把 skills/*/SKILL.md 的名字+描述索引进库（kind=skill），让召回能提示「这事有现成技能」。
    不需要 LLM：description 本身就是人写好的一句话摘要。"""
    root = SKILLS_ROOT
    if not root.exists():
        return 0
    now = int(time.time())
    updated = 0
    pending = []
    for p in root.rglob("SKILL.md"):
        try:
            head = p.read_text(errors="ignore")[:12000]
        except OSError:
            continue
        m = re.match(r"---\s*\n(.*?)\n---", head, re.S)
        name = desc = desc_zh = ""
        if m:
            lines = m.group(1).splitlines()
            for i, line in enumerate(lines):
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("description_zh:"):
                    desc_zh = line.split(":", 1)[1].strip().strip('"\'')
                elif line.startswith("description:"):
                    raw = line.split(":", 1)[1].strip()
                    if raw in ("|", ">", "|-", ">-", "|+", ">+", ""):
                        # 多行块（YAML 的 | 或 >），取后续缩进行
                        buf = []
                        for nxt in lines[i + 1:]:
                            if not nxt.strip():
                                if buf:
                                    break
                                continue
                            if nxt.startswith((" ", "\t")):
                                buf.append(nxt.strip())
                            else:
                                break
                        desc = " ".join(buf)
                    else:
                        desc = raw.strip('"\'')
        if not desc:
            desc = desc_zh
        if not name:
            name = p.parent.name
        if not desc:
            continue
        rel = str(p.parent.relative_to(root))
        content = f"技能「{name}」：{desc[:600]}"
        h = hashlib.sha1(("skill" + rel).encode()).hexdigest()
        exists = db.execute("SELECT id, content, anchor, desc_sha FROM memories WHERE hash=?",
                            (h,)).fetchone()
        if exists:
            mem_id, old = exists[0], exists[1]
            if old != content:
                # 文件是唯一真相源：变短也同步（机器分不出「精简」和「纠错」，也不该分）。
                # 旧版进 history，裁到最近 5 版 + 史上最长那版；要找回长描述去 history 查。
                # 顺带修 anchor：以前写成 content[:100]，把技能路径冲掉了，必须是 rel。
                _log_skill_history(db, mem_id, rel, old, now)
                db.execute("UPDATE memories SET content=?, anchor=?, desc_sha=? WHERE id=?",
                           (content, rel, hashlib.sha1(content.encode()).hexdigest(), mem_id))
                # FTS / 向量是两套独立表，UPDATE 命中 0 行会静默成功 → memories 改了索引没改，
                # 召回永久错。所以命中数不对就补插。
                cur = db.execute("UPDATE mem_fts SET content=?, anchor=? WHERE rowid=?",
                                 (content, rel, mem_id))
                if cur.rowcount != 1:
                    db.execute("INSERT INTO mem_fts(rowid,content,anchor) VALUES(?,?,?)",
                               (mem_id, content, rel))
                new_vec = embed([content[:2000]])[0]
                cur = db.execute("UPDATE vec SET emb=? WHERE rowid=?", (to_blob(new_vec), mem_id))
                if cur.rowcount != 1:
                    db.execute("INSERT INTO vec(rowid,emb) VALUES(?,?)", (mem_id, to_blob(new_vec)))
                updated += 1
            continue
        pending.append((content, h, rel))

    if not pending:
        db.commit()
        return updated
    vecs = embed([p[0] for p in pending])
    for (content, h, rel), vec in zip(pending, vecs):
        cur = db.execute(
            "INSERT INTO memories(kind,content,anchor,event_at,created_at,importance,"
            "hash,valid_from,confidence,desc_sha) VALUES('skill',?,?,?,?,0.6,?,?,0.9,?)",
            (content, rel, now, now, h, now, hashlib.sha1(content.encode()).hexdigest()))
        rid = cur.lastrowid
        db.execute("INSERT INTO vec(rowid,emb) VALUES(?,?)", (rid, to_blob(vec)))
        db.execute("INSERT INTO mem_fts(rowid,content,anchor) VALUES(?,?,?)", (rid, content, rel))
        # 首版也留痕（rev=1, reason=init），以后替换才知道这一版原来长什么样
        db.execute("INSERT INTO skill_desc_history"
                   "(mem_id,skill_rel,content,desc_sha,len,rev,superseded_at,reason)"
                   " VALUES(?,?,?,?,?,1,?,'init')",
                   (rid, rel, content, hashlib.sha1(content.encode()).hexdigest(),
                    len(content), now))
    db.commit()
    return len(pending) + updated


# ---------------- extract ----------------

API_BASE = "https://api.deepseek.com/v1"
API_MODEL = os.environ.get("BRAIN_LLM_MODEL", "deepseek-chat")

EXTRACT_PROMPT = """你是记忆抽取器。读下面这段对话，抽出最多 3 条值得长期记住的信息。

只要这五类：
- fact：事实（谁、在哪、多少钱、什么状态、机器配置、路径、账号）
- preference：偏好（用户喜欢/讨厌/要求什么）
- commitment：约定（答应过的事、定下的规矩、截止时间）
- relation：关系（和谁的关系、称呼、身份）
- episode：重要经历（发生过什么、结果如何）

不要抽：寒暄、工具输出、一次性的执行细节、没有结论的探索过程；
**能当场推导出来的也不抽**（路径、行号、端口、配置值、"第N条"这类，
它们会随文件变化，存下来就变成"权威的错误"）。

**preference 和 commitment 必须带 why**：记清楚"为什么定这条"，
碰到边缘情况时才知道这条还适不适用。没写 why 的偏好等于命令，只会盲从。
fact 如果只是状态快照（余额、剩余、当前版本），也在 why 里写一句"哪个时点的观察"。

输出严格 JSON，不要任何解释：
{"memories":[{"kind":"fact","content":"第三人称陈述，60字内","why":"为什么重要/什么场景适用/是哪一刻的观察，40字内，可空","quote":"对话里最短的原话片段，20字内","importance":0.5}]}
没有值得记的就输出 {"memories":[]}"""


def api_key() -> str:
    env = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env:
        return env
    env_file = Path(os.environ.get("BRAIN_ENV_FILE") or (HOME / ".agent-memory/.env"))
    if env_file.exists():
        for line in env_file.read_text(errors="ignore").splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("没有 DEEPSEEK_API_KEY（设环境变量，或给 BRAIN_ENV_FILE 指一个 .env）")


def llm_extract(text: str, key: str, timeout: int = 90) -> list[dict]:
    import urllib.request
    body = json.dumps({
        "model": API_MODEL,
        "messages": [{"role": "system", "content": EXTRACT_PROMPT},
                     {"role": "user", "content": text[:6000]}],
        "temperature": 0.2,
        "max_tokens": 700,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(
        API_BASE + "/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    raw = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return []
    try:
        got = json.loads(m.group(0)).get("memories", [])
    except json.JSONDecodeError:
        return []
    ok = []
    for it in got:
        kind = (it.get("kind") or "").strip()
        content = (it.get("content") or "").strip()
        if kind in {"fact", "preference", "commitment", "relation", "episode"} and 4 < len(content) <= 200:
            ok.append({"kind": kind, "content": content,
                       "quote": (it.get("quote") or "")[:40],
                       "importance": float(it.get("importance") or 0.5)})
    return ok


def extract_pending(db, workers: int = 6, limit: int | None = None) -> tuple[int, int]:
    from concurrent.futures import ThreadPoolExecutor
    key = api_key()
    q = ("SELECT id, content, event_at FROM memories "
         "WHERE kind='episode' AND extracted=0 ORDER BY id")
    rows = db.execute(q).fetchall()
    if limit:
        rows = rows[:limit]
    if not rows:
        return 0, 0
    print(f"待抽取 {len(rows)} 段", flush=True)

    def work(r):
        try:
            # 带上源 episode 的 event_at：抽出来的事实，发生时间应该是原对话的时间，
            # 不是抽取那一刻。2026-11-24 修：原来写 now，导致大量条目时间整体偏晚。
            return r[0], r[2] or None, llm_extract(r[1], key)
        except Exception as e:
            return r[0], r[2] or None, {"error": f"{type(e).__name__}: {e}"}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (rid, src_ts, got) in enumerate(ex.map(lambda r: work(r), rows), 1):
            results.append((rid, src_ts, got))
            if i % 20 == 0:
                print(f"  抽取 {i}/{len(rows)}", flush=True)

    added, errs = 0, 0
    now = int(time.time())
    for rid, src_ts, got in results:
        if isinstance(got, dict):
            errs += 1
            continue
        payload = [(g, rid) for g in got]
        if payload:
            vecs = embed([g[0]["content"] for g in payload])
            for (g, src), vec in zip(payload, vecs):
                near = db.execute(
                    "SELECT rowid, distance FROM vec WHERE emb MATCH ? "
                    "ORDER BY distance LIMIT 1", (to_blob(vec),)).fetchone()
                if near and float(near[1]) < 0.18:  # 余弦 > 0.98 视为重复
                    continue
                hashed = hashlib.sha1((g["kind"] + g["content"]).encode()).hexdigest()
                if db.execute("SELECT 1 FROM memories WHERE hash=?", (hashed,)).fetchone():
                    continue
                _conf = {"fact": 0.75, "preference": 0.85,
                         "commitment": 0.8, "relation": 0.7}.get(g["kind"], 0.65)
                cur = db.execute(
                    "INSERT INTO memories(kind,content,anchor,event_at,created_at,importance,"
                    "source_id,hash,valid_from,confidence,why) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (g["kind"], g["content"], g["quote"],
                     src_ts or now,          # 发生时间取源对话，取不到才退回落 now
                     now, g["importance"], src, hashed, now, _conf,
                     (g.get("why") or "").strip()[:120] or None))
                nid = cur.lastrowid
                db.execute("INSERT INTO vec(rowid,emb) VALUES(?,?)", (nid, to_blob(vec)))
                db.execute("INSERT INTO mem_fts(rowid,content,anchor) VALUES(?,?,?)",
                           (nid, g["content"], g["quote"]))
                db.execute("INSERT OR IGNORE INTO links(src,dst,kind) VALUES(?,?,?)",
                           (src, nid, "contains"))
                added += 1
        db.execute("UPDATE memories SET extracted=1 WHERE id=?", (rid,))
    db.commit()
    return added, errs


# ---------------- recall ----------------

def recall(db, query: str, k: int = 5, touch: bool = True, include_expired: bool = False,
           skip_mirrors: bool = False, include_archived: bool = False,
           skip_episode: bool = False) -> list[dict]:
    """skip_episode: 注入通道用。episode 是原始对话切片（平均上千字、占库近一半），
    占库里一半；它们进注入会挤掉真正的结论，还会自我放大（讨论某 bug 的对话入库后
    反过来把这个 bug 的正确答案挤出去）。查询用 /recall 和 search 不受影响。
    2026-11-24 加，来源：对照同类工具的排除清单 + 两轮独立召回审计。"""
    """召回。include_expired=True 时连已作废（valid_to 已过）的记忆一起返回。"""
    now = int(time.time())
    scores: dict[int, dict] = {}

    # 1) 关键词（FTS5 trigram）：中文没空格，整句当一个词会匹配不到，
    #    长片段要切成 3 字滑窗再 OR
    def _kw_terms(q: str) -> list[str]:
        out = []
        for t in re.split(r"[\s，。？！,.?!、；;：:（）()\[\]【】]+", q):
            t = t.strip()
            if len(t) < 2:
                continue
            if len(t) <= 4:
                out.append(t)
            else:
                out.extend(t[i:i + 3] for i in range(len(t) - 2))
        return out

    terms = _kw_terms(query)
    kw_hits: dict[int, float] = {}
    # trigram 索引只认 ≥3 字的片段，两个字的词（电视/走廊/指纹）FTS 一定查不到，
    # 所以短词单独走 LIKE，别让它整句丢掉。
    fts_terms = [t for t in terms if len(t) >= 3]
    short_terms = [t for t in terms if len(t) == 2]
    try:
        if fts_terms:
            mq = " OR ".join(f'"{t}"' for t in fts_terms)
            # bm25 越负越相关、且 ORDER BY rank 已按相关度排好：
            # 分必须随名次递减。旧写法 1/(1+|rank|) 会让最相关的拿最低分（投反票）。
            for i, (rid, _rank) in enumerate(db.execute(
                    "SELECT rowid, bm25(mem_fts) FROM mem_fts WHERE mem_fts MATCH ? "
                    "ORDER BY rank LIMIT 60", (mq,))):
                kw_hits[rid] = 1.0 / (1.0 + i * 0.1)
    except sqlite3.OperationalError:
        pass
    for t in short_terms[:6]:
        try:
            hits = db.execute(
                "SELECT id FROM memories WHERE 1=1" + ("" if include_archived else " AND archived=0") + " AND content LIKE ? "
                "ORDER BY importance DESC, event_at DESC LIMIT 40", ("%" + t + "%",)).fetchall()
        except sqlite3.OperationalError:
            continue
        for i, (rid,) in enumerate(hits):
            # 短词命中按名次给分，略低于 FTS 的最好名次，别压过语义通道
            score = 0.55 / (1.0 + i * 0.05)
            if score > kw_hits.get(rid, 0.0):
                kw_hits[rid] = score
    if not kw_hits:
        like = "%" + (terms[0] if terms else query) + "%"
        for rid, in db.execute(
                "SELECT id FROM memories WHERE 1=1" + ("" if include_archived else " AND archived=0") + " AND content LIKE ? ORDER BY importance DESC LIMIT 60", (like,)):
            kw_hits[rid] = 0.6

    # 2) 语义
    qv = embed([query])[0]
    sem: dict[int, float] = {}
    try:
        for rid, dist in db.execute(
                "SELECT rowid, distance FROM vec WHERE emb MATCH ? ORDER BY distance LIMIT 40",
                (to_blob(qv),)):
            d = float(dist)
            sem[rid] = max(0.0, min(1.0, 1.0 - (d * d) / 2.0))
    except sqlite3.OperationalError:
        pass

    cand = set(kw_hits) | set(sem)
    if not cand:
        return []
    rows = {r[0]: r for r in db.execute(
        f"SELECT id, kind, content, anchor, event_at, importance, last_access, pinned, "
        f"valid_to, confidence, why "
        f"FROM memories WHERE" + ("" if include_archived else " archived=0 AND") + f" id IN ({','.join('?' * len(cand))})", tuple(cand))}

    for rid in cand:
        r = rows.get(rid)
        if not r:
            continue
        # 已过作废时间的记忆不进候选：旧事实不能和新事实一起被当成"现在成立"
        if r[8] and not include_expired:
            continue
        # 注入通道跳过 soul/profile 镜像：这两类已每轮全文在系统提示里，
        # 再从库里召一遍只会挤掉别的记忆、还可能召回过时的旧版。
        if skip_mirrors and r[1] in ("soul", "profile"):
            continue
        if skip_episode and r[1] == "episode":
            continue
        imp = float(r[5] or 0.5)
        age_days = max(0.0, (now - (r[4] or now)) / 86400)
        fresh = max(0.0, 1.0 - age_days / 30)
        kw = kw_hits.get(rid, 0.0)
        se = sem.get(rid, 0.0)
        conf = float(r[9] if r[9] is not None else 0.7)
        # 质量通道：重要度 × 置信度，再乘新鲜度——三条信息压成一个可排序的量
        quality = (0.62 * imp + 0.38 * conf) * (0.75 + 0.25 * fresh)
        score = 0.46 * kw + 0.18 * se + 0.18 * imp + 0.10 * fresh
        if r[7]:
            score = max(score, 0.9)
            quality = max(quality, 0.95)
        scores[rid] = {
            "id": rid, "kind": r[1], "content": r[2], "anchor": r[3],
            "event_at": r[4], "importance": imp, "score": round(score, 4),
            "kw": round(kw, 3), "sem": round(se, 3),
            "confidence": conf, "valid_to": r[8], "why": (r[10] if len(r) > 10 else None),
            "_quality": quality,
        }

    # 三路 RRF 融合：词法 / 语义 / 质量，各通道按名次贡献 1/(60+rank)。
    # 单看余弦会被"语义泛化"骗，单看关键词会被"短词误命中"骗，融合后按名次投票。
    def _ranked_ranks(ch: dict) -> dict:
        return {rid: i for i, (rid, _) in enumerate(
            sorted(ch.items(), key=lambda x: -x[1]), 1)}

    rank_kw = _ranked_ranks(kw_hits)
    rank_sem = _ranked_ranks(sem)
    rank_q = _ranked_ranks({rid: s["_quality"] for rid, s in scores.items()})
    # 2026-11-23 加：长文本惩罚。
    # 起因：问「平时用什么语言」，前 5 位被 3 段几千字的旧对话占满，
    # 「用中文交流」那条排到第 6。长条目多是过程，短条目多是结论。
    # 500 字以内不打折；每超 1000 字再折 3%，最低折到 0.8。
    def _len_factor(text: str) -> float:
        n = len(text or "")
        if n <= 500:
            return 1.0
        return max(0.80, 1.0 - (n - 500) / 1000.0 * 0.03)

    for rid, s in scores.items():
        base = sum(1.0 / (60 + r.get(rid, 10 ** 6))
                   for r in (rank_kw, rank_sem, rank_q))
        s["rrf"] = round(base * _len_factor(s.get("content", "")), 6)
        s.pop("_quality", None)

    ranked = sorted(scores.values(), key=lambda x: -x["rrf"])
    mems = [x for x in ranked if x["kind"] != "skill"][:k]
    skis = [x for x in ranked if x["kind"] == "skill"][:2]
    out = mems + skis
    ids = [x["id"] for x in out]
    if ids and touch:
        db.execute(f"UPDATE memories SET last_access=? WHERE id IN ({','.join('?' * len(ids))})",
                   (now, *ids))
        db.commit()
    return out


# 会随外部变化失效的说法：路径、行号、端口、IP、金额、余额、条款编号、文件名。
# 只有这些才配加「先核实」——每条都加会警告通胀，模型第三轮就免疫（压测中反复出现，故只对易失效说法生效）。
_STATE_RE = re.compile(
    "|".join([
        "余额", "剩余", "还剩", "额度", "端口",
        "/home/", "/var/", "/tmp/", "/etc/",
        r"第\d+条",
        r"\.(?:py|ya?ml|json|md)\b",
        r"\d+\.\d+\.\d+\.\d+",
        r"\d+\s*元",
    ])
)

def inject_block(hits: list[dict], budget: int = 1000) -> str:
    if not hits:
        return ""
    mems = [h for h in hits if h["kind"] != "skill"]
    skis = [h for h in hits if h["kind"] == "skill"]
    lines: list[str] = []
    now_ts = time.time()

    if mems:
        lines.append("【我记得】")
        used = 0
        for h in mems:
            when = time.strftime("%m-%d", time.localtime(h["event_at"] or 0))
            body = h["content"].replace("\n", " ")
            body = body[:180] + ("…" if len(body) > 180 else "")
            line = f"- ({when}) {body}"
            # 带 why 的条目附一句原因：碰到边缘情况才知道这条还适不适用
            why = (h.get("why") or "").strip()
            if why and h["kind"] in ("preference", "commitment"):
                line += f"（因为：{why[:40]}）"
            # 选择性陈旧度：只对「会过时的说法」+「超过 2 天」+「事实类」加提醒
            age_days = (now_ts - (h["event_at"] or now_ts)) / 86400
            if (h["kind"] in ("fact", "commitment", "preference")
                    and age_days > 2 and _STATE_RE.search(body)):
                line += f"〔{int(age_days)}天前的观察，引用前先核〕"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)

    if skis:
        lines.append("【现成的技能】")
        for h in skis[:2]:
            body = h["content"].replace("\n", " ")
            lines.append(f"- {body[:90]}")

    return "\n".join(lines) if len(lines) > 1 else ""



# ---------------- brief / check（先确认再回答） ----------------

def brief(db, recent: int = 5) -> str:
    """库的自我介绍：有多少、分几类、最近在聊什么。回答前先看这个，知道库里有什么词汇。"""
    kinds = db.execute("SELECT kind, count(*) FROM memories WHERE archived=0 "
                       "GROUP BY kind ORDER BY 2 DESC").fetchall()
    total = sum(k[1] for k in kinds)
    lines = [f"库：{total} 条（" + " / ".join(f"{k} {n}" for k, n in kinds) + "）"]
    rows = db.execute("SELECT kind, content, event_at FROM memories WHERE archived=0 "
                      "ORDER BY event_at DESC, id DESC LIMIT ?", (recent,)).fetchall()
    if rows:
        lines.append("最近：")
        for k, c, ts in rows:
            when = time.strftime("%m-%d", time.localtime(ts or 0))
            lines.append(f"- [{k}] ({when}) {c[:70]}")
    return "\n".join(lines)


def check(db, query: str) -> dict:
    """这个问题库里答得了吗？答不了就直说，别硬编。"""
    hits = recall(db, query, 5, touch=False)
    if not hits:
        return {"answerable": False, "reason": "库里没有相关内容", "hits": 0}
    top = hits[0]
    # 关键词命中且总分够，或者语义非常接近——两条都够不着就别拿它当依据
    ok = (top["kw"] > 0 and top["score"] >= 0.30) or (top["sem"] >= 0.62 and top["score"] >= 0.35)
    return {
        "answerable": ok,
        "reason": "有相关记忆" if ok else "相关度太低，别拿它当依据",
        "hits": len(hits),
        "best_score": top["score"],
        "best_kind": top["kind"],
        "best": top["content"][:120],
    }


# ---------------- lifecycle ----------------

DECAY_STAMP = HOME / ".agent-memory/brain/.last_decay"


KEEP_KINDS = ("skill", "profile", "soul")


def decay(db, daily: float = 0.99, archive_below: float = 0.3, archive_days: int = 30,
          force: bool = False) -> dict:
    """每日衰减 + 归档。pinned 不衰减；event_at 越久越淡。

    daily 是「每天」的系数，所以一天只能乘一次——brain_tick 每 30 分钟调一次，
    按调用次数乘就是每天 48 次（0.99^48≈0.62），重要度两天内就被压平。
    用 .last_decay 记日期去重，force=True 才强制再跑。
    """
    today = time.strftime("%Y-%m-%d")
    if not force:
        try:
            if DECAY_STAMP.read_text().strip() == today:
                return {"faded": 0, "archived": 0, "skipped": "今日已衰减"}
        except OSError:
            pass
    now = int(time.time())
    rows = db.execute("SELECT id, importance, event_at, last_access, pinned, kind FROM memories "
                      "WHERE archived=0").fetchall()
    faded = archived = kept = 0
    for rid, imp, ev, last, pinned, kind in rows:
        if pinned:
            continue
        # 技能/档案/人格是从文件同步来的：文件在它就该在，不参与衰减与归档
        if kind in KEEP_KINDS:
            kept += 1
            continue
        imp = float(imp or 0.5) * daily
        db.execute("UPDATE memories SET importance=? WHERE id=?", (round(imp, 4), rid))
        faded += 1
        seen = max(ev or 0, last or 0)
        if imp < archive_below and now - seen > archive_days * 86400:
            db.execute("UPDATE memories SET archived=1 WHERE id=?", (rid,))
            archived += 1
    db.commit()
    if not force:
        try:
            DECAY_STAMP.parent.mkdir(parents=True, exist_ok=True)
            DECAY_STAMP.write_text(today)
        except OSError:
            pass
    return {"faded": faded, "archived": archived, "kept": kept}


# ---------------- serve ----------------

def serve(port: int = 8766) -> None:
    """常驻 HTTP 服务，任意宿主（CLI / MCP / HTTP）通过它拿记忆块。
    GET /inject?q=...&budget=1000 -> text/plain 记忆块
    GET /recall?q=...&k=5         -> application/json
    GET /health                   -> ok
    """
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    embed(["预热"])
    print(f"brain serve on 127.0.0.1:{port}", flush=True)

    access_log = Path(os.environ.get("BRAIN_ACCESS_LOG")
                      or (HOME / ".agent-memory/logs/brain_access.log"))
    try:
        access_log.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        access_log = Path(tempfile.gettempdir()) / "brain_access.log"

    class H(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # 只记路径，方便查调用
            try:
                with open(access_log, "a") as f:
                    f.write(f"{time.strftime('%m-%d %H:%M:%S')} {fmt % args}\n")
            except Exception:
                pass

        def _send(self, code: int, body: str, ctype: str = "text/plain; charset=utf-8"):
            raw = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            u = urlparse(self.path)
            qs = parse_qs(u.query)
            q = (qs.get("q") or [""])[0].strip()
            try:
                if u.path == "/health":
                    return self._send(200, "ok")
                db = connect()
                try:
                    if u.path == "/inject":
                        budget = int((qs.get("budget") or ["1000"])[0])
                        q = scrub(q)   # 先把网关包装剥掉，再查库
                        # 系统/定时任务包装不是人说的话，召回只会灌噪声：
                        # 服务端兜底（插件侧那道闸在热加载过的老进程里不一定生效）
                        if SYSTEM_PROMPT_RE.match(q):
                            return self._send(200, "")
                        # 再问一句「这问题答得了吗」，答不了就不注入——省 token，也少噪音
                        if not check(db, q)["answerable"]:
                            return self._send(200, "")
                        # 2026-11-23: k 5->8。召回前 5 位常被同一话题的对话占满，答案排到第 6 就丢了；
                        # 多取 3 条，注入仍受 budget 字数限制，不会撑大上下文。
                        return self._send(200, inject_block(recall(db, q, 8, skip_mirrors=True, skip_episode=True), budget))
                    if u.path == "/stats":
                        n = db.execute("SELECT count(*) FROM memories WHERE archived=0").fetchone()[0]
                        k = db.execute("SELECT kind, count(*) FROM memories WHERE archived=0 "
                                       "GROUP BY kind ORDER BY 2 DESC").fetchall()
                        return self._send(200, f"{n} 条记忆 / {DB.stat().st_size//1024} KB / " + str(dict(k)))
                    if u.path == "/brief":
                        n = int((qs.get("recent") or ["5"])[0])
                        return self._send(200, brief(db, n))
                    if u.path == "/recent":
                        n = int((qs.get("n") or ["20"])[0])
                        rows = db.execute("SELECT kind, content, event_at FROM memories "
                                          "WHERE archived=0 ORDER BY event_at DESC, id DESC LIMIT ?",
                                          (n,)).fetchall()
                        return self._send(200, "\n".join(
                            f"- [{r[0]}] {r[1][:120]}" for r in rows))
                    q = scrub(q)
                    if not q:
                        return self._send(400, "missing q")
                    if u.path == "/check":
                        return self._send(200, json.dumps(check(db, q), ensure_ascii=False),
                                          "application/json; charset=utf-8")
                    if u.path == "/recall":
                        k = int((qs.get("k") or ["5"])[0])
                        inc_arc = (qs.get("include_archived") or ["0"])[0] in ("1", "true", "yes")
                        return self._send(200, json.dumps(
                            recall(db, q, k, include_archived=inc_arc), ensure_ascii=False),
                            "application/json; charset=utf-8")
                    return self._send(404, "not found")
                finally:
                    db.close()
            except Exception as e:
                return self._send(500, f"{type(e).__name__}: {e}")

    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


# ---------------- cli ----------------

def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init")
    p = sub.add_parser("add", help="手动加一条记忆（不依赖任何对话流水）")
    p.add_argument("content"); p.add_argument("--kind", default="fact")
    p.add_argument("--importance", type=float, default=0.6)
    p.add_argument("--confidence", type=float, default=0.9)
    p.add_argument("--quote", default="", help="原话片段，用于溯源")
    p = sub.add_parser("ingest"); p.add_argument("--limit", type=int)
    p.add_argument("--from", dest="src", help="别的 state.db 路径（补灌历史）")
    p = sub.add_parser("recall"); p.add_argument("query"); p.add_argument("-k", type=int, default=5)
    p.add_argument("--json", action="store_true")
    p.add_argument("--include-expired", action="store_true", dest="inc_exp",
                   help="连已作废的记忆一起返回")
    p = sub.add_parser("expire", help="标记某条记忆已失效（不再参与召回）")
    p.add_argument("id", type=int); p.add_argument("--at", type=int, help="失效时间戳，默认现在")
    p = sub.add_parser("supersede", help="旧条作废、新条开始生效")
    p.add_argument("old", type=int); p.add_argument("new", type=int)
    p = sub.add_parser("conf", help="调某条记忆的置信度 0-1")
    p.add_argument("id", type=int); p.add_argument("value", type=float)
    p = sub.add_parser("inject"); p.add_argument("query"); p.add_argument("--budget", type=int, default=420)
    sub.add_parser("skills")
    p = sub.add_parser("history")
    p.add_argument("key", help="技能相对路径（如 productivity/llm-api-scripting）或 memories.id")
    p.add_argument("--full", action="store_true", help="打印每版全文，不只长度")
    p = sub.add_parser("extract"); p.add_argument("--limit", type=int); p.add_argument("-w", type=int, default=6)
    p = sub.add_parser("serve"); p.add_argument("--port", type=int, default=8766)
    p = sub.add_parser("decay"); p.add_argument("--daily", type=float, default=0.99)
    p.add_argument("--force", action="store_true", help="同一天再衰减一次（默认每天只跑一次）")
    p = sub.add_parser("brief"); p.add_argument("--recent", type=int, default=5)
    p = sub.add_parser("check"); p.add_argument("query")
    sub.add_parser("stats")
    p = sub.add_parser("search"); p.add_argument("query")

    a = ap.parse_args()
    db = connect()

    if a.cmd == "init":
        print(f"OK {DB}")
    elif a.cmd == "add":
        now = int(time.time())
        h = hashlib.sha1((a.kind + a.content).encode()).hexdigest()
        if db.execute("SELECT 1 FROM memories WHERE hash=?", (h,)).fetchone():
            print("内容已存在，跳过")
            return 0
        v = embed([a.content[:2000]])[0]
        cur = db.execute(
            "INSERT INTO memories(kind,content,anchor,event_at,created_at,importance,"
            "hash,valid_from,confidence) VALUES(?,?,?,?,?,?,?,?,?)",
            (a.kind, a.content, a.quote or a.content[:40], now, now,
             a.importance, h, now, a.confidence))
        rid = cur.lastrowid
        db.execute("INSERT INTO vec(rowid,emb) VALUES(?,?)", (rid, to_blob(v)))
        db.execute("INSERT INTO mem_fts(rowid,content,anchor) VALUES(?,?,?)",
                   (rid, a.content, a.quote or a.content[:40]))
        db.commit()
        print(f"✓ #{rid} [{a.kind}] {a.content[:60]}")
    elif a.cmd == "ingest":
        added, skipped = import_episodes(db, a.limit, Path(a.src) if a.src else None)
        print(f"新增 {added} 条，跳过重复 {skipped} 条，共 "
              f"{db.execute('SELECT count(*) FROM memories').fetchone()[0]} 条")
    elif a.cmd == "recall":
        hits = recall(db, a.query, a.k, include_expired=a.inc_exp)
        if a.json:
            print(json.dumps(hits, ensure_ascii=False, indent=1))
        else:
            for h in hits:
                when = time.strftime("%m-%d %H:%M", time.localtime(h["event_at"] or 0))
                flag = " [已失效]" if h.get("valid_to") else ""
                print(f"[rrf={h['rrf']:.4f} kw={h['kw']} sem={h['sem']} "
                      f"conf={h.get('confidence')}{flag} {when}] {h['content'][:150]}…")
    elif a.cmd == "expire":
        at = a.at or int(time.time())
        db.execute("UPDATE memories SET valid_to=? WHERE id=?", (at, a.id))
        db.commit()
        row = db.execute("SELECT kind, substr(content,1,60) FROM memories WHERE id=?", (a.id,)).fetchone()
        print(f"✓ {a.id} 已标记失效（{time.strftime('%m-%d %H:%M', time.localtime(at))}）")
        if row:
            print(f"  {row[0]}: {row[1]}…")
    elif a.cmd == "supersede":
        now = int(time.time())
        db.execute("UPDATE memories SET valid_to=? WHERE id=? AND valid_to IS NULL", (now, a.old))
        db.execute("UPDATE memories SET valid_from=?, valid_to=NULL WHERE id=?", (now, a.new))
        db.commit()
        print(f"✓ {a.old} 已作废，{a.new} 自现在生效")
    elif a.cmd == "conf":
        v = max(0.0, min(1.0, a.value))
        db.execute("UPDATE memories SET confidence=? WHERE id=?", (v, a.id))
        db.commit()
        print(f"✓ {a.id} 置信度 = {v}")
    elif a.cmd == "inject":
        print(inject_block(recall(db, a.query, 5), a.budget))
    elif a.cmd == "skills":
        n = ingest_skills(db)
        print(f"技能入库 {n} 条，现有 "
              f"{db.execute('SELECT count(*) FROM memories WHERE kind=\'skill\'').fetchone()[0]} 条")
    elif a.cmd == "history":
        key = a.key.strip()
        row = None
        if key.isdigit():
            row = db.execute("SELECT id, content, anchor FROM memories WHERE id=? AND kind='skill'",
                             (int(key),)).fetchone()
        if row is None:
            row = db.execute("SELECT id, content, anchor FROM memories WHERE kind='skill' AND anchor=?",
                             (key,)).fetchone()
        if row is None:
            row = db.execute("SELECT id, content, anchor FROM memories WHERE kind='skill' AND anchor LIKE ?",
                             (f"%{key}%",)).fetchone()
        if row is None:
            print(f"没找到技能：{key}")
            return 1
        mem_id, cur_content, rel = row
        print(f"id={mem_id}  {rel}")
        print(f"当前 {len(cur_content)} 字：{cur_content[:120]}")
        hist = db.execute(
            "SELECT rev, len, superseded_at, reason, content FROM skill_desc_history"
            " WHERE mem_id=? ORDER BY rev", (mem_id,)).fetchall()
        if not hist:
            print("  （没有历史版本）")
        for rev, ln, ts, reason, content in hist:
            when = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            print(f"  第{rev}版 {ln}字 {when} {reason}")
            if a.full:
                print("    " + (content or "")[:400].replace("\n", " "))
    elif a.cmd == "extract":
        added, errs = extract_pending(db, a.w, a.limit)
        kinds = dict(db.execute("SELECT kind, count(*) FROM memories GROUP BY kind").fetchall())
        print(f"抽出 {added} 条，失败 {errs} 段；现有 {kinds}")
    elif a.cmd == "decay":
        print("衰减/归档：", decay(db, a.daily, force=a.force))
    elif a.cmd == "serve":
        serve(a.port)
    elif a.cmd == "brief":
        print(brief(db, a.recent))
    elif a.cmd == "check":
        print(json.dumps(check(db, a.query), ensure_ascii=False, indent=1))
    elif a.cmd == "stats":
        n = db.execute("SELECT count(*) FROM memories").fetchone()[0]
        k = db.execute("SELECT kind, count(*) FROM memories GROUP BY kind").fetchall()
        size = DB.stat().st_size / 1024
        print(f"{n} 条记忆 / {size:.0f} KB / 类型 {dict(k)}")
    elif a.cmd == "search":
        for rid, snip in db.execute(
                "SELECT rowid, snippet(mem_fts,0,'[',']','…',20) FROM mem_fts "
                "WHERE mem_fts MATCH ? ORDER BY rank LIMIT 8", (a.query,)):
            print(rid, snip.replace("\n", " ")[:200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
