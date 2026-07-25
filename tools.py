"""Agent 工具：通过 HTTP 调用 VIN FastAPI 服务。"""

import os

import httpx

VIN_API_URL = os.getenv("VIN_API_URL", "http://127.0.0.1:8000")


def analyze_vin(vin: str) -> dict:
    """调用 POST /analyze-vin 并返回接口结果。"""
    try:
        response = httpx.post(
            f"{VIN_API_URL}/analyze-vin",
            json={"vin": vin},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"VIN API 调用失败，请确认 FastAPI 已启动：{error}"
        ) from error


TOOL_HANDLERS = {
    "analyze_vin": analyze_vin,
}
