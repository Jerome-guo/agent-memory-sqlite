"""memory-brain — 每轮从 brain.db 召回相关记忆，附到用户消息（不碰系统提示词）。

宿主无关：只打一个 HTTP 接口，服务在 brain.py serve。
规则同「息壤」：失效静默、不阻塞唤醒、宁可少说，不硬编。

2026-11-14 加：每轮把当前时刻塞进上下文，免得模型凭感觉编时间。
时间不走缓存，每次现算。
"""
from __future__ import annotations

import logging
import os
import time
from urllib.parse import urlencode
from urllib.request import urlopen

logger = logging.getLogger(__name__)

SERVICE = os.environ.get("BRAIN_URL", "http://127.0.0.1:8766").rstrip("/")
TIMEOUT = float(os.environ.get("BRAIN_TIMEOUT", "2.5"))
MIN_CHARS = 6
TTL = 720
_MAX_CACHE = 200
_cache: dict[str, tuple[float, str]] = {}


def _trace(line: str) -> None:
    try:
        with open("/tmp/brain_hook.log", "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    except Exception:
        pass


def _now_line() -> str:
    """当前时刻，每轮现算，不进缓存。"""
    lt = time.localtime()
    wd = "一二三四五六日"[lt.tm_wday]
    return f"[现在 {time.strftime('%Y-%m-%d %H:%M', lt)} 星期{wd}]"


def _on_pre_llm_call(user_message=None, is_first_turn: bool = False, **_kw):
    _trace(f"hook called len={len(user_message or '') if isinstance(user_message, str) else -1}")
    if os.environ.get("BRAIN_INJECT", "1") == "0":
        return None

    ts = _now_line()
    q = user_message if isinstance(user_message, str) else ""
    q = (q or "").strip()

    # cron 的包装提示、系统注入都不是人说的话：不召回、也不塞时间
    if q.startswith(("[IMPORTANT", "[System", "You are running as", "[Out-of-band", "[OUT-OF-BAND")):
        return None

    # 短消息、斜杠命令：不召回，但时间还是要给
    if len(q) < MIN_CHARS or q.startswith("/"):
        return {"context": ts}

    key = q[:200]

    now = time.time()
    hit = _cache.get(key)
    if hit and now - hit[0] < TTL:
        return {"context": f"{ts}\n{hit[1]}" if hit[1] else ts}

    try:
        with urlopen(f"{SERVICE}/inject?" + urlencode({"q": key}), timeout=TIMEOUT) as r:
            block = r.read().decode("utf-8", "ignore").strip()
    except Exception:
        logger.debug("brain inject unavailable", exc_info=True)
        return {"context": ts}

    if len(_cache) > _MAX_CACHE:
        _cache.clear()
    _cache[key] = (now, block)
    return {"context": f"{ts}\n{block}" if block else ts}


def register(ctx) -> None:
    _trace("register() called")
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
