"""Local-only FastAPI backend for the web desktop interface."""

import asyncio
import json
import os
import secrets
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app_paths import KNOWLEDGE_DIR, RESOURCE_DIR, SETTINGS_PATH, ensure_app_dirs
from knowledge_base import KnowledgeBase
from llm import run_agent
from main import _error_details

ensure_app_dirs()
WEB_DIR = RESOURCE_DIR / "web"
APP_TOKEN = secrets.token_urlsafe(32)
SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}
app = FastAPI(title="Academic Writing Agent", docs_url=None, redoc_url=None)


class ChatRequest(BaseModel):
    message: str
    thread_id: str = "writing-agent-web"


class SettingsRequest(BaseModel):
    provider: str
    model: str
    base_url: str = ""
    api_key: str = ""


def _require_token(value: str | None) -> None:
    if not value or not secrets.compare_digest(value, APP_TOKEN):
        raise HTTPException(status_code=403, detail="Invalid local app token")


def _read_settings() -> dict:
    settings = {"provider": "openai", "model": "", "base_url": ""}
    if SETTINGS_PATH.exists():
        try:
            settings.update(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return {key: settings.get(key, "") for key in settings if key != "api_key"}


def _apply_settings(value: SettingsRequest) -> None:
    os.environ["LLM_PROVIDER"] = value.provider
    os.environ["LLM_MODEL"] = value.model
    if value.base_url:
        os.environ["LLM_BASE_URL"] = value.base_url
    else:
        os.environ.pop("LLM_BASE_URL", None)
    if value.api_key:
        os.environ["LLM_API_KEY"] = value.api_key
    if value.provider == "anthropic":
        os.environ["ANTHROPIC_MODEL"] = value.model
        if value.api_key:
            os.environ["ANTHROPIC_API_KEY"] = value.api_key
        if value.base_url:
            os.environ["ANTHROPIC_BASE_URL"] = value.base_url
    elif value.provider == "gemini":
        os.environ["GEMINI_MODEL"] = value.model
        if value.api_key:
            os.environ["GEMINI_API_KEY"] = value.api_key
        if value.base_url:
            os.environ["GEMINI_BASE_URL"] = value.base_url
    SETTINGS_PATH.write_text(json.dumps({
        "provider": value.provider, "model": value.model,
        "base_url": value.base_url,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/bootstrap")
def bootstrap() -> dict:
    return {"token": APP_TOKEN, "settings": _read_settings()}


@app.post("/api/settings")
def save_settings(value: SettingsRequest, x_app_token: str | None = Header(None)) -> dict:
    _require_token(x_app_token)
    _apply_settings(value)
    return {"ok": True, "settings": _read_settings()}


@app.get("/api/sources")
def sources() -> dict:
    try:
        return KnowledgeBase().list_sources()
    except RuntimeError as error:
        if "知识库为空" in str(error):
            return {"indexed_chunks": 0, "sources": []}
        raise


@app.post("/api/sources")
async def upload_sources(files: list[UploadFile] = File(...),
                         x_app_token: str | None = Header(None)) -> dict:
    _require_token(x_app_token)
    imported = []
    for upload in files:
        name = Path(upload.filename or "paper").name
        suffix = Path(name).suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        target = KNOWLEDGE_DIR / name
        number = 2
        while target.exists():
            target = KNOWLEDGE_DIR / f"{Path(name).stem}-{number}{suffix}"
            number += 1
        target.write_bytes(await upload.read())
        imported.append(target.name)
    return {"imported": imported}


@app.delete("/api/sources/{source_path:path}")
def delete_source(source_path: str, x_app_token: str | None = Header(None)) -> dict:
    _require_token(x_app_token)
    target = (KNOWLEDGE_DIR / source_path).resolve()
    try:
        target.relative_to(KNOWLEDGE_DIR.resolve())
    except ValueError as error:
        raise HTTPException(status_code=400, detail="Invalid source path") from error
    if target.suffix.lower() not in SUPPORTED_SUFFIXES or not target.is_file():
        raise HTTPException(status_code=404, detail="Source not found")
    target.unlink()
    return {"ok": True}


@app.post("/api/index")
async def build_index(x_app_token: str | None = Header(None)) -> dict:
    _require_token(x_app_token)
    count = await asyncio.to_thread(KnowledgeBase().ensure_index)
    return {"ok": True, "indexed_chunks": count}


@app.post("/api/chat")
async def chat(value: ChatRequest, x_app_token: str | None = Header(None)):
    _require_token(x_app_token)

    async def events():
        def event(name: str, data: dict) -> str:
            return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield event("status", {"message": "正在检索论文并组织回答…"})
        try:
            answer = await run_agent(value.message.strip(), value.thread_id)
            yield event("message", {"content": answer})
        except Exception as error:
            yield event("error", {"message": _error_details(error)})
        yield event("done", {})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/{path:path}")
def web_ui(path: str = ""):
    return FileResponse(WEB_DIR / "index.html")
