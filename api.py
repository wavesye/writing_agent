"""车辆工具 FastAPI 服务。"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Vehicle Tools API", version="3.0.0")


class VinRequest(BaseModel):
    vin: str = Field(min_length=3, max_length=50)


class VinResult(BaseModel):
    vin: str
    status: str
    temperature: float
    has_anomaly: bool


class VehicleInfo(BaseModel):
    vin: str
    brand: str
    model: str
    year: int


class AdviceRequest(BaseModel):
    temperature: float
    has_anomaly: bool


class MaintenanceAdvice(BaseModel):
    level: str
    advice: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "v3", "tools": 3}


@app.post("/analyze-vin", response_model=VinResult)
def analyze_vin(request: VinRequest) -> VinResult:
    """返回模拟的车辆分析结果，后续可替换为真实业务逻辑。"""
    return VinResult(
        vin=request.vin.upper(),
        status="A",
        temperature=42.1,
        has_anomaly=False,
    )


@app.post("/vehicle-info", response_model=VehicleInfo)
def get_vehicle_info(request: VinRequest) -> VehicleInfo:
    """根据 VIN 返回模拟的车辆基本信息。"""
    return VehicleInfo(
        vin=request.vin.upper(),
        brand="Tesla",
        model="Model 3",
        year=2024,
    )


@app.post("/maintenance-advice", response_model=MaintenanceAdvice)
def get_maintenance_advice(request: AdviceRequest) -> MaintenanceAdvice:
    """根据分析指标生成确定性的维修建议。"""
    if request.has_anomaly or request.temperature >= 80:
        return MaintenanceAdvice(
            level="urgent",
            advice="建议立即停止使用并安排专业检修。",
        )
    if request.temperature >= 60:
        return MaintenanceAdvice(
            level="warning",
            advice="温度偏高，建议尽快检查冷却系统。",
        )
    return MaintenanceAdvice(
        level="normal",
        advice="当前指标正常，按常规周期保养即可。",
    )
