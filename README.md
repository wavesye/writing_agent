# VIN Agent v2

最小闭环：

```text
用户 -> LLM -> Tool -> FastAPI -> Python -> LLM -> 用户
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

程序会先打印实际执行的工具及返回值，再打印模型组织的自然语言答案。

默认模型是 `deepseek-v4-flash`。如需覆盖：

```bash
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

## 文件职责

- `main.py`：命令行交互
- `llm.py`：调用模型、识别工具请求、回传工具结果
- `tools.py`：通过 HTTP 调用 FastAPI 的 Agent 工具
- `api.py`：提供 `POST /analyze-vin` 业务接口

`POST /analyze-vin` 当前返回固定演示数据。后续可以只替换 API
内部的业务逻辑，Agent 和 Tool 协议无需改变。

如 API 不在本机默认地址，可以设置：

```bash
export VIN_API_URL="http://你的服务地址"
```
