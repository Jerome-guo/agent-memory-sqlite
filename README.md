# agent-memory-sqlite

**Long-term memory for AI agents: one SQLite file, no server, no cloud.**

A single-file memory store you can hand to any agent — copy it, back it up, move it to another machine. Built for Chinese first (FTS5 trigram + sliding-window queries), with bi-temporal validity, confidence scores, and three-lane RRF retrieval.

> 一个文件就是一台记忆机器。无服务、无云、无依赖地狱；拷走就能用，任何 agent 调 CLI 或 MCP 即可。

```
~800 lines of Python · SQLite + sqlite-vec + fastembed · no framework, no API server required
```

---

## 为什么不是又一个向量库

| | 常见做法 | 这个 |
|---|---|---|
| 部署 | 起服务 / 跑容器 / 接云 | **一个 `.db` 文件**，`cp` 就是备份 |
| 检索 | 只按余弦距离排 | 词法 + 语义 + 质量**三路 RRF 融合** |
| 时间 | 只有创建时间 | **双时间**：事实何时成立、何时不再成立 |
| 真假 | 一条记忆一个权重 | `importance`（多重要）与 `confidence`（多可信）**分开** |
| 中文 | 默认分词器整句当一个词，MATCH 永远 0 命中 | FTS5 `trigram` + **查询侧 3 字滑窗** |
| 工具 | 记忆和技能两套 | 技能也入记忆（`kind=skill`），单独占召回名额 |

## 快速开始

```bash
pip install sqlite-vec fastembed

python brain.py init
python brain.py add "用户的时区是 Asia/Shanghai，默认用中文" --kind preference
python brain.py recall "他大概在哪个时区"
python brain.py inject "他大概在哪个时区"     # 直接能塞进提示词的一块
```

首次运行会下载 `BAAI/bge-small-zh-v1.5`（约 100MB）。默认库在 `~/.agent-memory/brain.db`，想放别处就设 `BRAIN_DB=/你的路径/brain.db`。国内建议先设：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 命令

| 命令 | 作用 |
|---|---|
| `init` | 建库 |
| `add "内容" [--kind] [--importance] [--confidence]` | 手动加一条（不依赖任何对话流水） |
| `ingest` | 从 `state.db` 式对话流水增量灌入（打包成 episode） |
| `recall "问题" [-k N] [--include-expired]` | 召回，打印分数明细 |
| `inject "问题" [--budget 420]` | 输出可直接拼进提示词的记忆块 |
| `check "问题"` | 这个库答得了吗？答不了别注入（省 token） |
| `expire <id>` / `supersede <旧> <新>` | 某条不再成立 / 旧条作废、新条生效 |
| `conf <id> 0.9` | 调置信度 |
| `decay [--daily 0.99]` | 每日衰减 + 归档旧的 |
| `stats` / `brief` / `search` | 规模 / 自我介绍 / 纯 FTS 调试 |
| `history <技能路径或 id> [--full]` | 看这条记忆/技能的历史版本（被替换过的旧描述都在） |
| `serve --port 8766` | 起 HTTP 服务（可选） |

## 接进你的 agent

**① CLI（任何语言都能调）**

```bash
python brain.py inject "当前这轮用户说的话"
```

**② stdio MCP（Claude Desktop / Claude Code / 任何 MCP 宿主）**

```json
{
  "mcpServers": {
    "memory": {
      "command": "/absolute/path/to/python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

暴露 `recall(query,k)` / `inject(query,budget)` 两个工具，和 `brain://stats` / `brain://recall/{query}` 资源。
不设 `BRAIN_URL` 时进程自己开库，**不需要任何后台服务**。

**③ HTTP（多进程共享一个库）**

```bash
python brain.py serve --port 8766
curl "http://127.0.0.1:8766/inject?q=..."
```

**④ 插件：每轮自动注入（推荐，`examples/hermes-plugin/`）**

前三种都是「模型想起来了才去查」。插件不一样——它在模型思考**之前**就跑，把相关记忆和当前时间直接塞进上下文：

```
用户发消息
   ↓
pre_llm_call 钩子触发
   ├─ 生成 [现在 2026-11-14 08:20 星期六]   ← 每次现算，不进缓存
   ├─ 拿这句话去召回 top-5
   └─ 返回 {"context": "时间\n记忆块"}
   ↓
模型收到：时间 + 相关记忆 + 用户这句话
```

设计要点（照抄即可）：

- **不碰系统提示词**，只附在用户消息后面 —— 系统提示一改，prompt 缓存全失效
- **失败静默** —— 服务挂了/超时就当没记忆，不能把对话卡住
- **同一句话 12 分钟不重复查**（TTL 缓存）
- **短消息只给时间不查库**（“好”“行”“做吧”不值得一次向量计算）

以 Hermes 为例：把 `examples/hermes-plugin/` 放进 `~/.hermes/plugins/memory-brain/`，`plugins` 配置里加上 `memory-brain`，重启网关即可。

## 把人设档案也存进库

常驻记忆（`MEMORY.md` / `USER.md`）和人格设定（`SOUL.md`）有字数上限，满了就得挤掉旧的——**被挤掉的原文如果不在库里，就真没了**。

做法：按分隔符拆条，`kind` 用 `profile`（记忆/用户档案）和 `soul`（人格设定），重要度和置信度都给 0.95～1.0：

```python
for it in open("MEMORY.md").read().split("§"):
    content = "[memory] " + it.strip()
    # 三写：memories + vec + mem_fts，缺 vec 就召回不到
```

这样常驻记忆可以放心瘦身，挤掉的还能召回。

## 设计要点

**双时间（bi-temporal）**：`valid_from` / `valid_to`。用户换工作、换住址、改偏好之后，旧事实不会跟新事实一起被召回。`valid_to` 非空即不进召回。

**置信度**：`importance` 回答"多重要"，`confidence` 回答"多可信"。用户亲口说的 0.95，模型抽出来的 0.75，对话原始流水 0.60。抽取错了能一条条降下去。

**三路 RRF**：词法、语义、质量三个通道各自排名，按名次投票（`1/(60+rank)`）。单看余弦会被"语义泛化"骗，单看关键词会被短词误命中骗。注意 **RRF 只用于排序，不要拿绝对值当阈值**——满分也只有 ≈0.049。

**中文特化**：FTS5 `trigram` 分词 + 查询侧三字滑窗。中文没有空格，不切窗的话整句当一个 token，`MATCH` 永远 0 命中——这是中文场景最常见的坑。

**注入前置检查**：`check()` 先问"这问题答得了吗"，答不了返回空字符串。不相关的问题 token 消耗归零，也不会把五条无关记忆塞进上下文。

## 更新日志

### 2026-11-24

**修的坑（都是会让人踩到的）**

- **`why` 列表没有建表语句**：老库里那列是手工 ALTER 加的，新 clone 的人 `init` 后一写 why 就报 `no such column`。现在写进 DDL。
- **bm25 打分方向投反**：旧写法让最相关的拿最低分，等于投反票。
- **索引漏更新静默成功**：`UPDATE ... WHERE rowid=?` 命中 0 行也算成功，改了正文而 FTS/向量没跟上，召回永久错。现在按 `rowcount` 判断并补插。
- **并发写库报 locked**：开 WAL + `busy_timeout`，cron 和 CLI 同时写不再撞。
- **技能 anchor 被冲掉**：原来写 `content[:100]`，把技能路径截没了。

**新增的**

- **`why` 字段**：偏好/承诺类记忆记「为什么定这条」，注入时附成「（因为：…）」。只存禁令，模型会在没写过的情形里判错。
- **技能描述历史**：`skill_desc_history` 表，保留最近 5 版，另加一版是「史上最长那版」，永不删。
- **两字词 LIKE 通道**：FTS5 trigram 只认 ≥3 个字，「电视」「走廊」这类必查不到，现在单走 LIKE。
- **长文本惩罚**：长对话不再挤掉短结论（RRF 分按长度打折，最低 0.8）。
- **选择性陈旧度提醒**：只对会过时的说法（路径、端口、金额、条款号）加「引用前先核」，不再每条都加，避免警告通胀。

**变更**

- 默认库路径改为 `~/.agent-memory/`；老部署用 `BRAIN_DB` 指回原位置即可。
- 同步语义反转：文件是唯一真相源，内容变短也同步（机器分不清「精简」和「纠错」，宁可跟文件走）。

## 局限

- 单机单文件。要多人并发 / 上百万条，换 Milvus、Qdrant 这类。
- 语义模型是 `bge-small`，5 亿参数以下，泛化强但精度一般。要更准就换更大的 embedding。
- `ingest` / `skills` 只对 Hermes 格式的 `state.db` / `skills/` 有效；别的宿主用 `add` 或自己写导入。
- 没做加密。私密数据请自己管好文件权限。

## License

MIT
