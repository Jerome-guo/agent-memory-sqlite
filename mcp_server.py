#!/usr/bin/env python3
"""brain MCP server —— 自包含版：直接读包里的 brain.db，不依赖任何外部服务。

两种模式：
  1) 默认：本进程自己开库（需要 sqlite-vec + fastembed）
  2) 设了 BRAIN_URL=http://127.0.0.1:8766 时：转发给本机 brain 服务

路径全部相对本文件，不用 call env.bat 也能工作：
    data/brain.db    库（只读查询用）
    data/state.db    ingest 的原料
    skills/          技能原文

用法（stdio MCP）：
    python mcp_server.py
宿主配置：command=<本机 python 绝对路径>，args=[<本机 mcp_server.py 绝对路径>]

资源：
    brain://stats            库的规模
    brain://recent           最近 20 条
    brain://recall/{query}   按问题召回（资源模板，UTF-8 与 %XX 都吃）
工具：
    recall(query, k)         返回 JSON
    inject(query, budget)    返回可塞进提示词的记忆块
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from urllib.parse import unquote, urlencode
from urllib.request import urlopen

from mcp.server.mcpserver import MCPServer

HERE = Path(__file__).resolve().parent
SERVICE = os.environ.get("BRAIN_URL", "").rstrip("/")
TIMEOUT = float(os.environ.get("BRAIN_TIMEOUT", "10"))

if not SERVICE:
    # 跟着 BRAIN_HOME 走；没设就用 ~/.agent-memory（与 brain.py 的默认一致）
    root = Path(os.environ.get("BRAIN_HOME") or (Path.home() / ".agent-memory"))
    default_db = root / "brain.db"
    if not default_db.exists():
        default_db = HERE / "demo" / "brain.db"      # 仓库里自带的示例库
    os.environ.setdefault("BRAIN_DB", str(default_db))
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))

mcp = MCPServer("brain")

_brain_mod = None
_warm_error: str | None = None


def _brain():
    """懒加载 brain 模块 + 首次调用时预热 embedding 模型。"""
    global _brain_mod, _warm_error
    if _brain_mod is None:
        import brain as _b
        _brain_mod = _b
    return _brain_mod


def _warmup() -> None:
    """后台预热：模型首次加载要几十秒（首次还要下 ~100MB），
    放在 initialize 的空档里做，别让第一个查询撞超时。"""
    global _warm_error
    if SERVICE:
        return
    try:
        _brain().embed(["预热"])
    except Exception as e:  # 记录原因，第一次查询时原样报出来
        _warm_error = f"{type(e).__name__}: {e}"


def _q(name: str) -> str:
    """资源模板里的参数可能是 %XX 编码的（宿主行为不一），统一解一次。"""
    return unquote(unquote(name or "")).strip()


def _http(path: str, **params) -> str:
    url = f"{SERVICE}{path}"
    if params:
        url += "?" + urlencode(params)
    with urlopen(url, timeout=TIMEOUT) as r:
        return r.read().decode("utf-8", "ignore")


@mcp.resource("brain://stats")
def stats() -> str:
    """记忆库规模：多少条、按类型分布。"""
    if SERVICE:
        return _http("/stats")
    b = _brain()
    try:
        db = b.connect()
        n = db.execute("SELECT count(*) FROM memories WHERE archived=0").fetchone()[0]
        k = db.execute("SELECT kind, count(*) FROM memories WHERE archived=0 "
                       "GROUP BY kind ORDER BY 2 DESC").fetchall()
        return f"{n} 条记忆 / {dict(k)}"
    except Exception as e:
        raise RuntimeError(f"brain://stats 失败：{type(e).__name__}: {e}") from e


@mcp.resource("brain://brief")
def brief_res() -> str:
    """库的自我介绍：有多少条、分几类、最近在聊什么。回答前先读这个。"""
    if SERVICE:
        return _http("/brief")
    b = _brain()
    try:
        return b.brief(b.connect())
    except Exception as e:
        raise RuntimeError(f"brain://brief 失败：{type(e).__name__}: {e}") from e


@mcp.tool()
def check(query: str) -> str:
    """问库：这个问题答得了吗？（先查再答，答不了就直说）

    Args:
        query: 准备回答的问题
    """
    if SERVICE:
        return _http("/check", q=query)
    b = _brain()
    try:
        return json.dumps(b.check(b.connect(), query), ensure_ascii=False)
    except Exception as e:
        raise RuntimeError(f"check 失败：{type(e).__name__}: {e}") from e


@mcp.resource("brain://recent")
def recent() -> str:
    """最近 20 条记忆。"""
    if SERVICE:
        return _http("/recent", n=20)
    b = _brain()
    try:
        db = b.connect()
        rows = db.execute("SELECT kind, content FROM memories WHERE archived=0 "
                          "ORDER BY event_at DESC, id DESC LIMIT 20").fetchall()
        return "\n".join(f"- [{r[0]}] {r[1][:120]}" for r in rows)
    except Exception as e:
        raise RuntimeError(f"brain://recent 失败：{type(e).__name__}: {e}") from e


@mcp.resource("brain://recall/{query}")
def recall_resource(query: str) -> str:
    """按问题召回相关记忆（URI 里带查询词，支持 UTF-8 与 %XX 编码）。"""
    q = _q(query)
    if not q:
        raise ValueError("brain://recall/ 后面要带查询词")
    return inject(q)


@mcp.tool()
def recall(query: str, k: int = 5) -> str:
    """按问题召回记忆，返回 JSON（含分数、类型、原话锚点）。

    Args:
        query: 问题或关键词，例如「群收款那首歌怎么定的」
        k: 返回条数，默认 5
    """
    if SERVICE:
        return _http("/recall", q=query, k=k)
    b = _brain()
    try:
        db = b.connect()
        # touch=False：查询不该写库，避免每轮都改 last_access
        return json.dumps(b.recall(db, query, k, touch=False), ensure_ascii=False)
    except Exception as e:
        if _warm_error:
            raise RuntimeError(f"模型未就绪（{_warm_error}），请稍后重试") from e
        raise RuntimeError(f"recall 失败：{type(e).__name__}: {e}") from e


@mcp.tool()
def inject(query: str, budget: int = 420) -> str:
    """取一段可直接放进提示词的记忆块。

    Args:
        query: 当前这轮用户说的话
        budget: 字符预算，默认 420
    """
    if SERVICE:
        return _http("/inject", q=query, budget=budget)
    b = _brain()
    try:
        db = b.connect()
        return b.inject_block(b.recall(db, query, 5, touch=False), budget)
    except Exception as e:
        if _warm_error:
            raise RuntimeError(f"模型未就绪（{_warm_error}），请稍后重试") from e
        raise RuntimeError(f"inject 失败：{type(e).__name__}: {e}") from e


if __name__ == "__main__":
    threading.Thread(target=_warmup, daemon=True).start()
    mcp.run(transport="stdio")
