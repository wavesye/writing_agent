"""Tk desktop interface for the Academic Writing Agent."""

import asyncio
import json
import os
import queue
import shutil
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from app_paths import KNOWLEDGE_DIR, SETTINGS_PATH, ensure_app_dirs
from knowledge_base import KnowledgeBase
from llm import run_agent
from main import _error_details

PROVIDERS = (
    "openai", "deepseek", "openrouter", "ollama", "qwen", "moonshot",
    "zhipu", "siliconflow", "anthropic", "gemini", "custom",
)


class WritingAgentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_app_dirs()
        self.title("Academic Writing Agent")
        self.geometry("1180x760")
        self.minsize(900, 620)
        self.events = queue.Queue()
        self.settings = self._load_settings()
        self._build_ui()
        self._apply_settings_to_form()
        self._refresh_sources()
        self.after(100, self._poll_events)

    def _build_ui(self):
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Helvetica", 17, "bold"))
        style.configure("Section.TLabel", font=("Helvetica", 12, "bold"))

        root = ttk.Frame(self, padding=14)
        root.pack(fill="both", expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        ttk.Label(root, text="Academic Writing Agent", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )

        sidebar = ttk.Frame(root, width=320)
        sidebar.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)
        self._build_settings(sidebar)
        self._build_corpus(sidebar)

        chat = ttk.Frame(root)
        chat.grid(row=1, column=1, sticky="nsew")
        chat.columnconfigure(0, weight=1)
        chat.rowconfigure(1, weight=1)
        ttk.Label(chat, text="论文写作对话", style="Section.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        self.transcript = scrolledtext.ScrolledText(
            chat, wrap="word", state="disabled", font=("Helvetica", 13), padx=12, pady=12
        )
        self.transcript.grid(row=1, column=0, sticky="nsew")
        self.transcript.tag_configure("user", foreground="#2457a6", spacing1=10)
        self.transcript.tag_configure("assistant", foreground="#222222", spacing1=10)
        self.transcript.tag_configure("error", foreground="#b42318", spacing1=10)

        composer = ttk.Frame(chat)
        composer.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        composer.columnconfigure(0, weight=1)
        self.prompt = tk.Text(composer, height=5, wrap="word", font=("Helvetica", 13))
        self.prompt.grid(row=0, column=0, sticky="ew")
        self.prompt.bind("<Command-Return>", lambda _event: self._send())
        buttons = ttk.Frame(composer)
        buttons.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        self.send_button = ttk.Button(buttons, text="发送 ⌘↩", command=self._send)
        self.send_button.pack(fill="x")
        ttk.Button(buttons, text="清空显示", command=self._clear_chat).pack(
            fill="x", pady=(8, 0))

        self.status = tk.StringVar(value="就绪")
        ttk.Label(root, textvariable=self.status, anchor="w").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(10, 0))

    def _build_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="模型设置", padding=10)
        frame.grid(row=0, column=0, sticky="ew")
        frame.columnconfigure(0, weight=1)
        self.provider = tk.StringVar()
        self.model = tk.StringVar()
        self.base_url = tk.StringVar()
        self.api_key = tk.StringVar()
        self.thread_id = tk.StringVar()
        fields = (
            ("聊天服务", ttk.Combobox(frame, textvariable=self.provider,
                                      values=PROVIDERS, state="readonly")),
            ("模型 ID", ttk.Entry(frame, textvariable=self.model)),
            ("Base URL（可选）", ttk.Entry(frame, textvariable=self.base_url)),
            ("API Key（仅本次运行）", ttk.Entry(frame, textvariable=self.api_key, show="•")),
            ("论文会话 ID", ttk.Entry(frame, textvariable=self.thread_id)),
        )
        for row, (label, widget) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=row * 2, column=0, sticky="w",
                                              pady=(6 if row else 0, 3))
            widget.grid(row=row * 2 + 1, column=0, sticky="ew")
        ttk.Button(frame, text="应用设置", command=self._save_settings).grid(
            row=len(fields) * 2, column=0, sticky="ew", pady=(10, 0))

    def _build_corpus(self, parent):
        frame = ttk.LabelFrame(parent, text="论文知识库", padding=10)
        frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        parent.rowconfigure(1, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        ttk.Label(frame, text="PDF / Markdown / TXT").grid(row=0, column=0, sticky="w")
        self.sources = tk.Listbox(frame, height=9)
        self.sources.grid(row=1, column=0, sticky="nsew", pady=(6, 8))
        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, sticky="ew")
        buttons.columnconfigure((0, 1), weight=1)
        ttk.Button(buttons, text="导入论文", command=self._import_documents).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(buttons, text="构建索引", command=self._build_index).grid(
            row=0, column=1, sticky="ew", padx=(4, 0))
        ttk.Label(frame, text="Embedding：本地 ONNX MiniLM（Python 内加载）",
                  wraplength=280).grid(row=3, column=0, sticky="w", pady=(9, 0))

    def _load_settings(self):
        defaults = {"provider": "openai", "model": "", "base_url": "",
                    "thread_id": "writing-agent-desktop"}
        if SETTINGS_PATH.exists():
            try:
                defaults.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                pass
        return defaults

    def _apply_settings_to_form(self):
        self.provider.set(self.settings["provider"])
        self.model.set(self.settings["model"])
        self.base_url.set(self.settings["base_url"])
        self.thread_id.set(self.settings["thread_id"])

    def _save_settings(self):
        self.settings = {"provider": self.provider.get(), "model": self.model.get().strip(),
                         "base_url": self.base_url.get().strip(),
                         "thread_id": self.thread_id.get().strip() or "writing-agent-desktop"}
        SETTINGS_PATH.write_text(json.dumps(self.settings, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        self._configure_environment()
        self.status.set("模型设置已应用；API Key 未写入磁盘")

    def _configure_environment(self):
        provider, model, base_url = (self.provider.get(), self.model.get().strip(),
                                     self.base_url.get().strip())
        key = self.api_key.get().strip()
        os.environ["LLM_PROVIDER"] = provider
        os.environ["LLM_MODEL"] = model
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url
        else:
            os.environ.pop("LLM_BASE_URL", None)
        if key:
            os.environ["LLM_API_KEY"] = key
            if provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = key
            elif provider == "gemini":
                os.environ["GEMINI_API_KEY"] = key
        if provider == "anthropic":
            os.environ["ANTHROPIC_MODEL"] = model
            if base_url:
                os.environ["ANTHROPIC_BASE_URL"] = base_url
        elif provider == "gemini":
            os.environ["GEMINI_MODEL"] = model
            if base_url:
                os.environ["GEMINI_BASE_URL"] = base_url

    def _import_documents(self):
        paths = filedialog.askopenfilenames(
            title="选择参考论文", filetypes=[("论文文件", "*.pdf *.md *.txt"),
                                           ("PDF", "*.pdf"), ("所有文件", "*")])
        imported = 0
        for raw_path in paths:
            source = Path(raw_path)
            if source.suffix.lower() not in {".pdf", ".md", ".txt"}:
                continue
            target = KNOWLEDGE_DIR / source.name
            counter = 2
            while target.exists() and target.read_bytes() != source.read_bytes():
                target = KNOWLEDGE_DIR / f"{source.stem}-{counter}{source.suffix}"
                counter += 1
            if not target.exists():
                shutil.copy2(source, target)
                imported += 1
        self._refresh_sources()
        self.status.set(f"已导入 {imported} 个新文件")

    def _refresh_sources(self):
        self.sources.delete(0, "end")
        files = sorted(path for path in KNOWLEDGE_DIR.rglob("*")
                       if path.suffix.lower() in {".pdf", ".md", ".txt"})
        for path in files:
            self.sources.insert("end", str(path.relative_to(KNOWLEDGE_DIR)))

    def _run_background(self, label, function):
        self.status.set(label)
        threading.Thread(target=self._worker, args=(function,), daemon=True).start()

    def _worker(self, function):
        try:
            self.events.put(("result", function()))
        except Exception as error:
            self.events.put(("error", _error_details(error)))

    def _build_index(self):
        self._run_background("正在加载本地 embedding 模型并构建混合索引…",
                             lambda: f"索引完成：{KnowledgeBase().ensure_index()} 个片段")

    def _send(self):
        text = self.prompt.get("1.0", "end").strip()
        if not text:
            return
        self._save_settings()
        if not self.model.get().strip():
            messagebox.showwarning("缺少模型", "请先填写聊天模型 ID。")
            return
        self.prompt.delete("1.0", "end")
        self._append("你", text, "user")
        self.send_button.configure(state="disabled")
        thread_id = self.thread_id.get().strip() or "writing-agent-desktop"
        self._run_background("Agent 正在检索论文并写作…",
                             lambda: asyncio.run(run_agent(text, thread_id)))

    def _append(self, author, content, tag):
        self.transcript.configure(state="normal")
        self.transcript.insert("end", f"{author}\n", tag)
        self.transcript.insert("end", f"{content}\n\n")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    def _poll_events(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "error":
                    self._append("错误", payload, "error")
                    self.status.set("操作失败")
                elif payload.startswith("索引完成："):
                    self.status.set(payload)
                    self._refresh_sources()
                else:
                    self._append("Agent", payload, "assistant")
                    self.status.set("完成")
                self.send_button.configure(state="normal")
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _clear_chat(self):
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")


def main():
    if "--mcp-server" in sys.argv:
        from mcp_server import mcp
        mcp.run(transport="stdio")
        return
    WritingAgentApp().mainloop()


if __name__ == "__main__":
    main()
