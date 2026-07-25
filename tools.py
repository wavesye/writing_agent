"""本地工具：真正执行 Python 的地方。"""


def analyze_vin(vin: str) -> dict:
    """返回模拟的车辆分析结果。"""
    return {
        "vin": vin.upper(),
        "status": "A",
        "temperature": 42.1,
        "has_anomaly": False,
    }


TOOL_HANDLERS = {
    "analyze_vin": analyze_vin,
}
