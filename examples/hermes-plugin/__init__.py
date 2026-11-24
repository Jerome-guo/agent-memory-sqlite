"""memory-brain v2 候选版 —— 未部署，放在候选目录。

与在线版的差别（详见 DIFF.md）：
  1. 注入缓存整个禁用（WAL 下 mtime 判更新不可靠，性能待测后再议）
  2. 短指令补上文：只读消息的 content 字段，绝不读 api_content（那里含上次注入）
  3. 查询截断留痕：记原始长度、实际使用长度、URL 编码后长度
  4. 日志区分 skip / empty / ok / timeout / error，只记数字与状态，不记正文
  5. 条目数叫「注入条目估算」，不等于数据库命中数

依赖：Hermes 的 pre_llm_call 钩子（参数已在 agent/turn_context.py:861 核实）。
"""
from __future__ import annotations

import logging
import os
import re
import time
from urllib.parse import urlencode
from urllib.request import urlopen

logger = logging.getLogger(__name__)

SERVICE = os.environ.get("BRAIN_URL", "http://127.0.0.1:8766").rstrip("/")
TIMEOUT = float(os.environ.get("BRAIN_TIMEOUT", "2.5"))
MIN_CHARS = 6
TRACE_PATH = os.environ.get("BRAIN_TRACE", "/tmp/brain_hook_v2.log")

# 查询长度：字符数与 URL 编码后字节数都要卡
MAX_Q_CHARS = 800
MAX_Q_ENCODED = 1800

# 缓存：候选版整段关闭
CACHE_ENABLED = False

# 注入块的行首标记，用来估算条目数
_LINE_MARK = re.compile(r"^- \(\d\d-\d\d\) ", re.MULTILINE)


def _trace(line: str) -> None:
    try:
        with open(TRACE_PATH, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    except Exception:
        pass


def _now_line() -> str:
    lt = time.localtime()
    wd = "一二三四五六日"[lt.tm_wday]
    return f"[现在 {time.strftime('%Y-%m-%d %H:%M', lt)} 星期{wd}]"


def _clean_user_text(msg) -> str:
    """只取 content；api_content 里含上一轮的注入块，读了就会重复拼接。"""
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return ""
    c = msg.get("content")
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):  # 多模态：只取文本片段
        parts = [p.get("text", "") for p in c if isinstance(p, dict)]
        return " ".join(p for p in parts if isinstance(p, str)).strip()
    return ""


def _prev_user_text(history, current: str) -> str:
    """从后往前找上一条真正的用户原话。找不到就返回空，不编。"""
    for m in reversed(list(history or [])):
        t = _clean_user_text(m)
        if t and t != current:
            return t
    return ""


def _fit_query(q: str) -> tuple[str, int, int, bool]:
    """截断并按 URL 编码长度再收一次，两次都保头保尾。

    尾部是当前指令（短指令补上文时，当前那句拼在最后），不能砍。
    返回 (查询串, 原始长度, 编码后长度, 是否截断)。
    """
    raw_len = len(q)
    truncated = False
    if len(q) > MAX_Q_CHARS:
        q = q[:600] + "\n…\n" + q[-200:]
        truncated = True
    enc = len(urlencode({"q": q}))
    # 二次收缩：对半砍中间，头尾各留一份，绝不整段从头截
    # 下限用 60 字符：1 个中文字符 URL 编码后占 9 个字符，1800 字节约等于 200 个中文字
    while enc > MAX_Q_ENCODED and len(q) > 60:
        keep_head = max(30, int(len(q) * 0.45))
        keep_tail = max(30, int(len(q) * 0.35))
        q = q[:keep_head] + "\n…\n" + q[-keep_tail:]
        enc = len(urlencode({"q": q}))
        truncated = True
    return q, raw_len, enc, truncated


def _estimate_items(block: str) -> int:
    """按返回文本里 "- (MM-DD) " 的行数估算条目数。这不是数据库命中数。"""
    return len(_LINE_MARK.findall(block or ""))


def _on_pre_llm_call(user_message=None, conversation_history=None,
                     is_first_turn: bool = False, **_kw):
    ts = _now_line()
    q = user_message if isinstance(user_message, str) else ""
    q = (q or "").strip()

    # 系统/定时任务的包装提示：不召回，也不塞时间
    if q.startswith(("[IMPORTANT", "[System", "You are running as",
                     "[Out-of-band", "[OUT-OF-BAND")):
        _trace("skip=cron_or_system")
        return None

    if os.environ.get("BRAIN_INJECT", "1") == "0":
        _trace("skip=disabled_by_env")
        return None

    # 短指令：尝试用上一条用户原话承接
    if len(q) < MIN_CHARS or q.startswith("/"):
        if q.startswith("/"):
            _trace(f"skip=slash qlen={len(q)}")
            return {"context": ts}
        prev = _prev_user_text(conversation_history, q)
        if not prev:
            _trace(f"skip=short_no_context qlen={len(q)} turn={int(bool(is_first_turn))}")
            return {"context": ts}
        q = f"{prev[-300:]}\n{q}"
        short_mode = "with_context"
        prev_len = len(prev)
    else:
        short_mode = "no"
        prev_len = 0

    query, raw_len, enc_len, truncated = _fit_query(q)
    t0 = time.time()
    try:
        with urlopen(f"{SERVICE}/inject?" + urlencode({"q": query}), timeout=TIMEOUT) as r:
            block = r.read().decode("utf-8", "ignore").strip()
    except TimeoutError:
        _trace(f"result=timeout short={short_mode} qraw={raw_len} qenc={enc_len} "
               f"trunc={'yes' if truncated else 'no'} prev={prev_len} t={int((time.time()-t0)*1000)}ms")
        return {"context": ts}
    except Exception as exc:
        _trace(f"result=error:{type(exc).__name__} short={short_mode} qraw={raw_len} "
               f"qenc={enc_len} trunc={'yes' if truncated else 'no'} t={int((time.time()-t0)*1000)}ms")
        return {"context": ts}

    ms = int((time.time() - t0) * 1000)
    if not block:
        _trace(f"result=empty short={short_mode} qraw={raw_len} qenc={enc_len} "
               f"trunc={'yes' if truncated else 'no'} t={ms}ms")
        return {"context": ts}

    est = _estimate_items(block)
    _trace(f"result=ok short={short_mode} qraw={raw_len} qenc={enc_len} "
           f"trunc={'yes' if truncated else 'no'} cache=off bytes={len(block)} "
           f"est_items={est} t={ms}ms")
    return {"context": f"{ts}\n{block}"}


def register(ctx) -> None:
    _trace("register() called")
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
