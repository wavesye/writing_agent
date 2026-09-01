"""Writable resource locations for source runs and frozen macOS apps."""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
IS_FROZEN = bool(getattr(sys, "frozen", False))
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", PROJECT_DIR))

if IS_FROZEN:
    APP_DATA_DIR = Path.home() / "Library" / "Application Support" / "Writing Agent"
else:
    APP_DATA_DIR = PROJECT_DIR

KNOWLEDGE_DIR = APP_DATA_DIR / "knowledge"
DATA_DIR = APP_DATA_DIR / "data"
MODEL_DIR = DATA_DIR / "models"
BUNDLED_MODEL_DIR = RESOURCE_DIR / "models" / "all-MiniLM-L6-v2"
SETTINGS_PATH = DATA_DIR / "settings.json"


def ensure_app_dirs() -> None:
    for path in (KNOWLEDGE_DIR, DATA_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
