"""把车辆 FastAPI 能力通过 MCP 标准暴露给 Agent。"""

from mcp.server.fastmcp import FastMCP

from tools import (
    analyze_vin as analyze_vin_api,
    get_maintenance_advice as get_maintenance_advice_api,
    get_vehicle_info as get_vehicle_info_api,
    list_vehicles as list_vehicles_api,
)

mcp = FastMCP("Vehicle Tools")


@mcp.tool()
def list_vehicles() -> dict:
    """列出车队中真实存在的 VIN；搜索车辆时必须先调用此工具。"""
    return list_vehicles_api()


@mcp.tool()
def analyze_vin(vin: str) -> dict:
    """分析指定 VIN，返回状态、温度（摄氏度）和异常标记。"""
    return analyze_vin_api(vin)


@mcp.tool()
def get_vehicle_info(vin: str) -> dict:
    """查询指定 VIN 的品牌、车型和年份。"""
    return get_vehicle_info_api(vin)


@mcp.tool()
def get_maintenance_advice(temperature: float, has_anomaly: bool) -> dict:
    """根据分析工具返回的温度与异常标记生成维修建议。"""
    return get_maintenance_advice_api(temperature, has_anomaly)


if __name__ == "__main__":
    mcp.run(transport="stdio")
