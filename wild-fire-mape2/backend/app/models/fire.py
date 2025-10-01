from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum

class FireStatus(str, Enum):
    ACTIVE = "active"
    CONTAINED = "contained"
    EXTINGUISHED = "extinguished"

class FireSeverity(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

class FireEvent(BaseModel):
    id: Optional[str] = None
    name: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    start_date: datetime
    end_date: Optional[datetime] = None
    status: FireStatus
    severity: FireSeverity
    area_acres: Optional[float] = None
    cause: Optional[str] = None
    containment_percentage: Optional[float] = Field(None, ge=0, le=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class FireRiskAssessment(BaseModel):
    latitude: float
    longitude: float
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: str
    factors: dict
    prediction_method: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)