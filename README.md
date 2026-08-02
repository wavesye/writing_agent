# VIN Agent v6.3

最小闭环：

```text
用户 -> 记忆/摘要 -> LangGraph -> MCP车辆工具/知识库RAG -> 用户
```

## 运行

需要 Python 3.10+，可选择 DeepSeek 或公司内部模型。

```bash
cd agent-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

安装完成后使用两个终端。

终端 1，启动 VIN API：

```bash
cd agent-demo
source .venv/bin/activate
uvicorn api:app --reload
```

可先验证接口：

```bash
curl -X POST http://127.0.0.1:8000/analyze-vin \
  -H "Content-Type: application/json" \
  -d '{"vin":"VIN123"}'
```

终端 2，启动 Agent：

```bash
cd agent-demo
source .venv/bin/activate
export DEEPSEEK_API_KEY="你的 DeepSeek API Key"
python main.py
```

默认会话 ID 为 `vin-agent-cli`。可以显式指定：

```bash
export AGENT_THREAD_ID="waves-demo-1"
python main.py
```

使用相同 `thread_id` 时，即使退出程序后重新启动，也会恢复上下文；
换一个 `thread_id` 就相当于开始新会话。

然后输入：

```text
分析 VIN123
```

Agent 会在每次请求开始时通过 MCP 动态发现工具。可以尝试：

```text
查询 VIN123 的车型
分析 VIN123 的状态
分析 VIN123 并给出维修建议
查询 VIN123 的车型、状态并给出维修建议
分析 VIN789
它是什么车型？
```

模型会自动选择一个或多个工具。程序会依次打印实际执行的工具及
返回值，再打印模型组织的自然语言答案。

默认模型是 `deepseek-v4-flash`。如需覆盖：

```bash
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

## 切换到公司内部 LLM

模型层使用统一 Provider，默认是 DeepSeek。公司接口至少需要配置：

```bash
export LLM_PROVIDER="company"
export COMPANY_LLM_BASE_URL="http://10.8.3.75:9192"
export COMPANY_LLM_TOKEN="登录接口返回的token"
export COMPANY_LLM_CHAT_PATH="/实际发送消息的路径"
python main.py
```

默认初始化接口为 `GET /chat/init`。如果不同：

```bash
export COMPANY_LLM_INIT_PATH="/你的初始化路径"
```

如果初始化响应是嵌套结构，例如：

```json
{"data": {"sessionId": "abc"}}
```

可以明确指定字段：

```bash
export COMPANY_LLM_SESSION_FIELD="data.sessionId"
export COMPANY_LLM_SESSION_REQUEST_FIELD="session_id"
```

公司接口原生支持 `tools` 时使用默认模式：

```bash
export COMPANY_LLM_TOOL_MODE="native"
```

如果公司接口只支持普通文本，可以使用 JSON Prompt 模拟 Tool Calling：

```bash
export COMPANY_LLM_TOOL_MODE="prompt"
```

可选配置：

```bash
export COMPANY_LLM_MODEL="公司模型名"
export COMPANY_LLM_TIMEOUT="60"
```

真实 Token 不要写入代码、README 或 Git。

## 文件职责

- `main.py`：命令行交互
- `llm.py`：建立 MCP 会话并启动 LangGraph
- `llm_providers/`：DeepSeek 与公司 HTTP 模型适配器
- `agent_graph.py`：状态、节点、条件边与循环上限
- `data/checkpoints.sqlite`：本地会话状态（自动创建，不提交 Git）
- `mcp_server.py`：MCP Server，声明并暴露标准 MCP Tools
- `knowledge_base.py`：Markdown 分段、FTS5 索引与检索
- `knowledge/`：本地维修手册和规范
- `data/knowledge.sqlite`：自动生成的全文索引
- `tools.py`：MCP Tool 到 FastAPI 的 HTTP 适配层
- `api.py`：车辆业务 API 与模拟数据

当前工具：

- `list_vehicles`：列出车队中真实存在的 VIN
- `analyze_vin`：状态、温度与异常分析
- `get_vehicle_info`：品牌、车型与年份查询
- `get_maintenance_advice`：根据分析结果给出维修建议
- `search_knowledge`：检索维修手册、故障规范和流程

接口当前返回固定演示数据。后续可以只替换 API 内部的业务逻辑，
Agent 和 Tool 协议无需改变。

如 API 不在本机默认地址，可以设置：

```bash
export VIN_API_URL="http://你的服务地址"
```

## MCP 与 API 的区别

FastAPI 是业务接口，可以在浏览器访问
[`http://127.0.0.1:8000/docs`](http://127.0.0.1:8000/docs)。

MCP 是 Agent 与工具之间的标准协议。本项目采用 `stdio` 传输：
Agent 会自动启动 `mcp_server.py`、发现工具 Schema，并通过 MCP 调用，
因此不需要为 MCP 单独打开端口或终端。

## LangGraph 状态

v5 将原来的 Python `for` 循环改成状态图：

```text
START -> model -> [需要工具?]
                    | 是
                    v
                mcp_tools
                    |
                    +-----> model
                    |
                    +-----> limit -> END
               否 -> END
```

共享状态包含：

- `messages`：完整对话与工具结果
- `tool_rounds`：已执行的工具轮数
- `phase`：当前/最终节点阶段
- `tool_trace`：实际调用过的 MCP Tool 名称
- `final_answer`：最终返回给用户的内容
- `current_vin`：当前会话正在关注的车辆

## v6.1 多轮记忆

项目使用 `AsyncSqliteSaver` 在 LangGraph 每个步骤后保存 checkpoint，
并通过 `thread_id` 隔离会话：

```text
thread_id=demo-a: 分析 VIN789 -> “它”仍指 VIN789
thread_id=demo-b: 新会话，不继承 demo-a
```

默认数据库：

```text
data/checkpoints.sqlite
```

如需修改位置：

```bash
export AGENT_CHECKPOINT_DB="/安全目录/agent-memory.sqlite"
```

Checkpoint 会包含用户消息和工具结果，不应提交到 Git。当前已启用
`LANGGRAPH_STRICT_MSGPACK=true`。SQLite 适合本地开发，生产环境应改用
PostgreSQL Checkpointer，并按用户/组织设计不可猜测的 `thread_id`。

## v6.2 摘要记忆

完整消息仍保存在 SQLite 中用于恢复和审计，但发送给模型的上下文会
自动压缩为：

```text
系统规则 + 历史摘要 + 最近消息
```

默认在未摘要消息超过 20 条时触发摘要，并保留最近 8 条原始消息：

```bash
export SUMMARY_TRIGGER_MESSAGES="20"
export SUMMARY_KEEP_RECENT="8"
```

触发阈值必须大于保留条数。为了快速观察摘要节点，可以临时设置：

```bash
export SUMMARY_TRIGGER_MESSAGES="6"
export SUMMARY_KEEP_RECENT="3"
```

运行日志会出现：

```text
[graph:check_context] 未摘要消息=7, 触发阈值=6
[graph:summarize] 已摘要4条，保留最近3条
```

LangGraph 状态新增：

- `conversation_summary`：累计历史摘要
- `summary_cursor`：已摘要到消息列表的哪个位置
- `summary_count`：累计执行摘要的次数

摘要会额外调用一次当前 LLM，因此不要设置过低阈值。精确 VIN 仍保存
在结构化的 `current_vin` 中，不依赖自然语言摘要。

## v6.3 本地知识库 RAG

知识库采用不需要 Embedding 的本地方案：

```text
knowledge/*.md
  -> 按 Markdown 标题分段
  -> SQLite FTS5 trigram 索引
  -> search_knowledge MCP Tool
  -> LLM 根据片段回答并标注来源
```

手动建立或刷新索引：

```bash
python knowledge_base.py
```

通常不需要手动刷新；`search_knowledge` 会计算文档指纹，Markdown
内容变化后自动重建索引。默认索引文件为：

```text
data/knowledge.sqlite
```

查看 MCP Tool 及一次真实知识检索：

```bash
python inspect_mcp.py
```

建议测试：

```text
VIN789为什么需要立即停运？请给出处置依据和来源。
C级故障应走什么维修流程？
紧急维修工单需要记录哪些字段？
```

检索结果包含 `source`、`section`、`content` 和 `rank`，模型被要求以
`[文件名#章节]` 引用。当前文档全部是演示内容，不应作为真实维修依据。

只查看 MCP Server 暴露的工具：

```bash
python inspect_mcp.py
```
