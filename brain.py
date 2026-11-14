#!/usr/bin/env python3
"""agent-memory-sqlite — 一个文件就是一台记忆机器

Long-term memory for AI agents: one SQLite file, no server, no cloud.
Chinese-first (FTS5 trigram + sliding-window query), with bi-temporal validity,
confidence, and three-lane RRF retrieval.

  brain.py init                          建库
  brain.py add "记得的事" [--kind fact]   手动加一条
  brain.py ingest [--limit N]            从对话流水增量灌（打包成 episode）
  brain.py recall "问题" [-k 5]          召回
  brain.py inject "问题" [--budget 420]  输出可直接塞进提示词的记忆块
  brain.py check "问题"                  这个库答得了吗（答不了就别注入）
  brain.py expire <id>                   某条不再成立
  brain.py supersede <旧id> <新id>       旧条作废、新条生效
  brain.py conf <id> 0.9                 调置信度
  brain.py decay                         每日衰减 + 归档
  brain.py stats / brief / search

设计原则：只用 stdlib + sqlite-vec + fastembed，不 import 任何宿主内部模块；
输入输出都是纯文本/JSON，任何 agent 调 CLI 或 MCP 即可。

库：~/.agent-memory/brain.db（BRAIN_HOME / BRAIN_DB 可覆盖）
向量：BAAI/bge-small-zh-v1.5（512 维，中文场景；英文可换 bge-small-en-v1.5）
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
# 一切都跟着环境变量走，整包搬走时只改 BRAIN_HOME 一个变量
ROOT = Path(os.environ.get("BRAIN_HOME") or (HOME / ".agent-memory"))
DB = Path(os.environ.get("BRAIN_DB") or (ROOT / "brain.db"))
# 下面两个是「可选原料」：接了才有 ingest（对话流水）和 skills（技能索引）
STATE_DB = Path(os.environ.get("BRAIN_STATE_DB") or (ROOT / "state.db"))
SKILLS_ROOT = Path(os.environ.get("BRAIN_SKILLS_DIR") or (ROOT / "skills"))
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
    db = sqlite3.connect(DB)
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
        -- 双时间：什么时候开始成立 / 什么时候不再成立（NULL = 至今成立）
        valid_from INTEGER,
        valid_to INTEGER,
        -- 置信度（0-1）：跟 importance 分开，一个答"多重要"，一个答"多可信"
        confidence REAL DEFAULT 0.7,
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
    db.execute("CREATE INDEX IF NOT EXISTS idx_mem_valid ON memories(valid_to)")
    for col, decl in (("source_id", "INTEGER"), ("extracted", "INTEGER DEFAULT 0"),
                      ("valid_from", "INTEGER"), ("valid_to", "INTEGER"),
                      ("confidence", "REAL DEFAULT 0.7")):
        try:
            db.execute(f"ALTER TABLE memories ADD COLUMN {col} {decl}")
        except sqlite3.OperationalError:
            pass
    return db


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


def scrub(text: str) -> str:
    return INJECT_RE.sub("", text).strip()


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
        exists = db.execute("SELECT id, content FROM memories WHERE hash=?", (h,)).fetchone()
        if exists:
            if exists[1] != content:
                # 只在「不缩短」时更新：SKILL.md 的描述可以为了省提示词 token 而精简，
                # 但库里的完整描述要留着（召回靠它）。补充/修正（变长）照常同步。
                if len(content) >= len(exists[1]):
                    db.execute("UPDATE memories SET content=?, anchor=? WHERE id=?", (content, content[:100], exists[0]))
                    db.execute("UPDATE mem_fts SET content=?, anchor=? WHERE rowid=?", (content, content[:100], exists[0]))
                    new_vec = embed([content[:2000]])[0]
                    db.execute("UPDATE vec SET emb=? WHERE rowid=?", (to_blob(new_vec), exists[0]))
            continue
        pending.append((content, h, rel))

    if not pending:
        db.commit()
        return 0
    vecs = embed([p[0] for p in pending])
    for (content, h, rel), vec in zip(pending, vecs):
        cur = db.execute(
            "INSERT INTO memories(kind,content,anchor,event_at,created_at,importance,"
            "hash,valid_from,confidence) VALUES('skill',?,?,?,?,0.6,?,?,0.9)",
            (content, rel, now, now, h, now))
        rid = cur.lastrowid
        db.execute("INSERT INTO vec(rowid,emb) VALUES(?,?)", (rid, to_blob(vec)))
        db.execute("INSERT INTO mem_fts(rowid,content,anchor) VALUES(?,?,?)", (rid, content, rel))
    db.commit()
    return len(pending)


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

不要抽：寒暄、工具输出、一次性的执行细节、没有结论的探索过程。

输出严格 JSON，不要任何解释：
{"memories":[{"kind":"fact","content":"第三人称陈述，60字内","quote":"对话里最短的原话片段，20字内","importance":0.5}]}
没有值得记的就输出 {"memories":[]}"""


def api_key() -> str:
    env = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env:
        return env
    env_file = Path(os.environ.get("BRAIN_ENV_FILE") or (ROOT / ".env"))
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
            return r[0], llm_extract(r[1], key)
        except Exception as e:
            return r[0], {"error": f"{type(e).__name__}: {e}"}

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (rid, got) in enumerate(ex.map(work, rows), 1):
            results.append((rid, got))
            if i % 20 == 0:
                print(f"  抽取 {i}/{len(rows)}", flush=True)

    added, errs = 0, 0
    now = int(time.time())
    for rid, got in results:
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
                    "source_id,hash,valid_from,confidence) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (g["kind"], g["content"], g["quote"], now, now,
                     g["importance"], src, hashed, now, _conf))
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

def recall(db, query: str, k: int = 5, touch: bool = True, include_expired: bool = False) -> list[dict]:
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
    try:
        mq = " OR ".join(f'"{t}"' for t in terms if len(t) >= 3) or query
        for rid, rank in db.execute(
                "SELECT rowid, bm25(mem_fts) FROM mem_fts WHERE mem_fts MATCH ? "
                "ORDER BY rank LIMIT 60", (mq,)):
            kw_hits[rid] = 1.0 / (1.0 + abs(float(rank)))
    except sqlite3.OperationalError:
        pass
    if not kw_hits:
        like = "%" + (terms[0] if terms else query) + "%"
        for rid, in db.execute(
                "SELECT id FROM memories WHERE archived=0 AND content LIKE ? ORDER BY importance DESC LIMIT 60", (like,)):
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
        f"valid_to, confidence "
        f"FROM memories WHERE archived=0 AND id IN ({','.join('?' * len(cand))})", tuple(cand))}

    for rid in cand:
        r = rows.get(rid)
        if not r:
            continue
        # 已过作废时间的记忆不进候选：旧事实不能和新事实一起被当成"现在成立"
        if r[8] and not include_expired:
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
            "confidence": conf, "valid_to": r[8],
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
    for rid, s in scores.items():
        s["rrf"] = round(sum(1.0 / (60 + r.get(rid, 10 ** 6))
                             for r in (rank_kw, rank_sem, rank_q)), 6)
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


def inject_block(hits: list[dict], budget: int = 420) -> str:
    if not hits:
        return ""
    mems = [h for h in hits if h["kind"] != "skill"]
    skis = [h for h in hits if h["kind"] == "skill"]
    lines: list[str] = []

    if mems:
        lines.append("【我记得】")
        used = 0
        for h in mems:
            when = time.strftime("%m-%d", time.localtime(h["event_at"] or 0))
            body = h["content"].replace("\n", " ")
            body = body[:180] + ("…" if len(body) > 180 else "")
            line = f"- ({when}) {body}"
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

DECAY_STAMP = ROOT / ".last_decay"


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
    rows = db.execute("SELECT id, importance, event_at, last_access, pinned FROM memories "
                      "WHERE archived=0").fetchall()
    faded = archived = 0
    for rid, imp, ev, last, pinned in rows:
        if pinned:
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
    return {"faded": faded, "archived": archived}


# ---------------- serve ----------------

def serve(port: int = 8766) -> None:
    """常驻 HTTP 服务，宿主（Hermes/DSH/任何东西）通过它拿记忆块。
    GET /inject?q=...&budget=420  -> text/plain 记忆块
    GET /recall?q=...&k=5         -> application/json
    GET /health                   -> ok
    """
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
    from urllib.parse import urlparse, parse_qs

    embed(["预热"])
    print(f"brain serve on 127.0.0.1:{port}", flush=True)

    access_log = Path(os.environ.get("BRAIN_ACCESS_LOG")
                      or (ROOT / "brain_access.log"))
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
                        budget = int((qs.get("budget") or ["420"])[0])
                        # 系统/定时任务包装不是人说的话，召回只会灌噪声：
                        # 服务端兜底（插件侧那道闸在热加载过的老进程里不一定生效）
                        if SYSTEM_PROMPT_RE.match(q):
                            return self._send(200, "")
                        # 再问一句「这问题答得了吗」，答不了就不注入——省 token，也少噪音
                        if not check(db, q)["answerable"]:
                            return self._send(200, "")
                        return self._send(200, inject_block(recall(db, q, 5), budget))
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
                    if not q:
                        return self._send(400, "missing q")
                    if u.path == "/check":
                        return self._send(200, json.dumps(check(db, q), ensure_ascii=False),
                                          "application/json; charset=utf-8")
                    if u.path == "/recall":
                        k = int((qs.get("k") or ["5"])[0])
                        return self._send(200, json.dumps(recall(db, q, k), ensure_ascii=False),
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
