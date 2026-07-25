# VIN Agent v4

最小闭环：

```text
用户 -> DeepSeek -> MCP Client -> MCP Server -> FastAPI -> 用户
```

## 运行

需要 Python 3.10+ 和 DeepSeek API Key。

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
```

模型会自动选择一个或多个工具。程序会依次打印实际执行的工具及
返回值，再打印模型组织的自然语言答案。

默认模型是 `deepseek-v4-flash`。如需覆盖：

```bash
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

## 文件职责

- `main.py`：命令行交互
- `llm.py`：MCP Client、DeepSeek 调度和多轮工具调用
- `mcp_server.py`：MCP Server，声明四个标准 MCP Tools
- `tools.py`：MCP Tool 到 FastAPI 的 HTTP 适配层
- `api.py`：车辆业务 API 与模拟数据

当前工具：

- `list_vehicles`：列出车队中真实存在的 VIN
- `analyze_vin`：状态、温度与异常分析
- `get_vehicle_info`：品牌、车型与年份查询
- `get_maintenance_advice`：根据分析结果给出维修建议

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

只查看 MCP Server 暴露的工具：

```bash
python inspect_mcp.py
```
