# VIN Agent v3

最小闭环：

```text
用户 -> LLM -> 自动选择 Tool(s) -> FastAPI -> LLM -> 用户
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

可以尝试不同意图：

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
- `llm.py`：调用模型、识别工具请求、回传工具结果
- `tools.py`：注册并通过 HTTP 调用三个 Agent 工具
- `api.py`：提供三个车辆业务接口

当前工具：

- `analyze_vin`：状态、温度与异常分析
- `get_vehicle_info`：品牌、车型与年份查询
- `get_maintenance_advice`：根据分析结果给出维修建议

接口当前返回固定演示数据。后续可以只替换 API 内部的业务逻辑，
Agent 和 Tool 协议无需改变。

如 API 不在本机默认地址，可以设置：

```bash
export VIN_API_URL="http://你的服务地址"
```
