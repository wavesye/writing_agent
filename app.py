"""Desktop launcher for the local web-based Academic Writing Agent."""

import socket
import threading
import time
import urllib.request
import webbrowser

import uvicorn

from web_app import app


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str) -> None:
    for _ in range(100):
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=0.5):
                return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError("本地界面服务启动超时")


def main() -> None:
    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    server = uvicorn.Server(uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning"
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_until_ready(url)
    try:
        import webview
    except ImportError:
        webbrowser.open(url)
        print(f"Academic Writing Agent 已打开：{url}")
        try:
            while thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            server.should_exit = True
        return
    webview.create_window(
        "Academic Writing Agent", url=url, width=1280, height=820,
        min_size=(900, 620), background_color="#f7f7f5",
    )
    webview.start(debug=False)
    server.should_exit = True
    thread.join(timeout=3)


if __name__ == "__main__":
    main()
