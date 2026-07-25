"""VIN 分析 FastAPI 服务。"""

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="VIN Analysis API", version="2.0.0")


class VinRequest(BaseModel):
    vin: str = Field(min_length=3, max_length=50)


class VinResult(BaseModel):
    vin: str
    status: str
    temperature: float
    has_anomaly: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze-vin", response_model=VinResult)
def analyze_vin(request: VinRequest) -> VinResult:
    """返回模拟的车辆分析结果，后续可替换为真实业务逻辑。"""
    return VinResult(
        vin=request.vin.upper(),
        status="A",
        temperature=42.1,
        has_anomaly=False,
    )
