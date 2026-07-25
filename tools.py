"""Agent 工具：通过 HTTP 调用 VIN FastAPI 服务。"""

import os

import httpx

VIN_API_URL = os.getenv("VIN_API_URL", "http://127.0.0.1:8000")


def _post(path: str, payload: dict) -> dict:
    """统一调用车辆 FastAPI 服务。"""
    try:
        response = httpx.post(
            f"{VIN_API_URL}{path}",
            json=payload,
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as error:
        raise RuntimeError(
            f"VIN API 调用失败，请确认 FastAPI 已启动：{error}"
        ) from error


def analyze_vin(vin: str) -> dict:
    """分析 VIN 的状态、温度和异常标记。"""
    return _post("/analyze-vin", {"vin": vin})


def get_vehicle_info(vin: str) -> dict:
    """查询 VIN 对应的车辆基本信息。"""
    return _post("/vehicle-info", {"vin": vin})


def get_maintenance_advice(temperature: float, has_anomaly: bool) -> dict:
    """根据分析结果获取维修建议。"""
    return _post(
        "/maintenance-advice",
        {"temperature": temperature, "has_anomaly": has_anomaly},
    )


TOOL_HANDLERS = {
    "analyze_vin": analyze_vin,
    "get_vehicle_info": get_vehicle_info,
    "get_maintenance_advice": get_maintenance_advice,
}
