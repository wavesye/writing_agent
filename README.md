# VIN Agent MVP

最小闭环：

```text
用户 -> LLM -> analyze_vin -> Python -> LLM -> 用户
```

## 运行

需要 Python 3.10+ 和 DeepSeek API Key。

```bash
cd agent-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
- `tools.py`：本地 Python 工具及工具注册表

`analyze_vin` 当前只返回固定演示数据。后续可保持工具协议不变，
把函数内部替换成 FastAPI 或数据库调用。
