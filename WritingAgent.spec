# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_all

project = Path(SPECPATH)
datas, binaries, hiddenimports = collect_all("chromadb")
for package in ("onnxruntime", "tokenizers", "mcp", "langgraph"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hidden

model_dir = project / "data" / "models" / "all-MiniLM-L6-v2"
if model_dir.exists():
    datas.append((str(model_dir), "models/all-MiniLM-L6-v2"))

a = Analysis(
    [str(project / "app.py")],
    pathex=[str(project)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="Academic Writing Agent", console=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False,
               name="Academic Writing Agent")
app = BUNDLE(
    coll,
    name="Academic Writing Agent.app",
    bundle_identifier="com.local.academic-writing-agent",
    info_plist={"NSHighResolutionCapable": True},
)
