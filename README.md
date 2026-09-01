# Academic Writing Agent

一个面向论文写作与润色的本地 RAG Agent。它从参考论文中检索与当前段落相关的
表达样本，学习其句式、衔接、语气和段落组织，再在不改变事实与引文的前提下修改文本。

## 知识库

将风格相近、质量可靠且你有权使用的文件放入 `knowledge/`（可使用子目录）：

- `.pdf`：直接提取文本并保留页码；扫描版 PDF 需先 OCR。
- `.md`：按标题分段并保留章节名。
- `.txt`：按长度分段。

建议按领域或目标期刊建立子目录，例如：

```text
knowledge/
├── target-journal/
│   ├── paper-01.pdf
│   └── paper-02.pdf
└── own-best-work/
    └── accepted-paper.pdf
```

首次检索会创建关键词索引 `data/knowledge.sqlite` 和 Chroma 向量库 `data/chroma/`。
文件内容、分段参数或 embedding 模型发生变化后，相关索引会自动重建。
知识库是文风样本，不被视为事实或引用依据；Agent 被要求提炼风格特征而非复制原句。

## Hybrid RAG 与 Embedding

默认 `RAG_MODE=hybrid`：Chroma 语义检索和 SQLite FTS5 关键词检索分别召回候选片段，
再用融合排序返回结果。也可设置为 `vector` 或 `keyword`。

默认使用 Python 进程内的 ONNX `all-MiniLM-L6-v2` 生成向量，不需要 Ollama、
独立模型服务或 embedding API。第一次使用时会自动下载模型，之后完全从本地加载：

```bash
export EMBEDDING_PROVIDER="local"
export RAG_MODE="hybrid"
python prepare_local_model.py
```

也可以使用 OpenAI 或其他兼容的 embedding API：

```bash
export EMBEDDING_PROVIDER="openai-compatible"
export EMBEDDING_BASE_URL="https://api.openai.com/v1"
export EMBEDDING_API_KEY="你的 API Key"
export EMBEDDING_MODEL="你账号中可用的 embedding 模型"
```

聊天模型与 embedding 模型完全独立，例如可以用 Claude 写作、Python 在本地生成向量。
如果临时不想使用向量库，可设置 `RAG_MODE=keyword`。

## 安装与运行

需要 Python 3.10+：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export LLM_PROVIDER="openai"
export OPENAI_API_KEY="你的 API Key"
export OPENAI_MODEL="你账号中可用且支持工具调用的模型"
python app.py
```

桌面界面支持聊天模型配置、论文导入、构建知识库、来源查看和论文写作对话；API Key
只保存在当前进程内，不写入设置文件。CLI 调试入口仍为 `python main.py`。

可提前验证知识库：

```bash
python knowledge_base.py
```

示例请求：

```text
请参考知识库风格润色下面这段引言，保留引文和数字不变：……
把这段方法写得更严谨、紧凑：……
先列出知识库中的论文，再总结它们共有的文风。
```

## 模型配置

通过 `LLM_PROVIDER` 切换模型。项目不绑定某一家厂商，但所选模型必须支持工具调用。

| Provider | `LLM_PROVIDER` | 必填变量 |
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY`, `OPENAI_MODEL` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |
| 本地 Ollama | `ollama` | `OLLAMA_MODEL` |
| 通义千问 | `qwen` | `DASHSCOPE_API_KEY`, `QWEN_MODEL` |
| 月之暗面/Kimi | `moonshot` | `MOONSHOT_API_KEY`, `MOONSHOT_MODEL` |
| 智谱 | `zhipu` | `ZHIPU_API_KEY`, `ZHIPU_MODEL` |
| 硅基流动 | `siliconflow` | `SILICONFLOW_API_KEY`, `SILICONFLOW_MODEL` |
| Anthropic Claude | `anthropic` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` |
| Google Gemini | `gemini` | `GEMINI_API_KEY`, `GEMINI_MODEL` |

其他 OpenAI-compatible 服务也能直接使用：

```bash
export LLM_PROVIDER="custom"
export LLM_BASE_URL="https://服务地址/v1"
export LLM_API_KEY="你的 API Key"
export LLM_MODEL="模型 ID"
```

本地 Ollama 示例：

```bash
export LLM_PROVIDER="ollama"
export OLLAMA_MODEL="支持 tool calling 的本地模型名"
python main.py
```

通用覆盖变量 `LLM_BASE_URL`、`LLM_API_KEY` 和 `LLM_MODEL` 优先级最高，便于代理、
私有网关或兼容服务复用预设。API Key 只放环境变量，不要提交到 Git。

## 记忆配置

常用变量：

```bash
export AGENT_THREAD_ID="paper-a"
export AGENT_CHECKPOINT_DB="data/checkpoints.sqlite"
export SUMMARY_TRIGGER_MESSAGES="20"
export SUMMARY_KEEP_RECENT="8"
```

相同 `AGENT_THREAD_ID` 会恢复论文主题、术语和修改偏好。

## 打包 macOS App / DMG

先把本地 embedding 模型下载到项目数据目录，再打包：

```bash
source .venv/bin/activate
python prepare_local_model.py
pip install -r requirements-build.txt
pyinstaller --clean --noconfirm WritingAgent.spec
```

产物位于 `dist/Academic Writing Agent.app`。可用 macOS 自带工具制作 DMG：

```bash
hdiutil create -volname "Academic Writing Agent" \
  -srcfolder "dist/Academic Writing Agent.app" \
  -ov -format UDZO "dist/Academic-Writing-Agent.dmg"
```

打包时，已下载的 ONNX 模型会包含进 `.app`。打包版的论文、索引、会话和非敏感设置
保存在 `~/Library/Application Support/Writing Agent/`，不会写入只读的应用包。

## 核心文件

- `agent_graph.py`：论文写作规则、对话摘要和工具调用流程。
- `knowledge_base.py`：PDF/Markdown/TXT 提取、分段和混合检索。
- `embeddings.py`：进程内 ONNX 与 OpenAI-compatible embedding 配置。
- `vector_store.py`：Chroma 持久化向量库。
- `mcp_server.py`：暴露 `search_style_corpus` 与 `list_corpus_sources`。
- `llm.py`：模型、MCP 与持久化记忆的连接层。
- `main.py`：命令行入口。
- `app.py`：桌面可视化界面与打包入口。
- `WritingAgent.spec`：PyInstaller macOS 应用打包配置。

当前使用 Chroma + SQLite FTS5，适合本地和中小型论文语料库。
