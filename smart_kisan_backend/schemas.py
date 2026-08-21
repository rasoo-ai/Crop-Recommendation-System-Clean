from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

class UserCreate(BaseModel):
    name:     str
    email:    EmailStr
    password: str
    state:    Optional[str] = None
    district: Optional[str] = None

class UserOut(BaseModel):
    id:         int
    name:       str
    email:      str
    state:      Optional[str]
    district:   Optional[str]
    created_at: datetime
    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    name:     Optional[str] = None
    state:    Optional[str] = None
    district: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type:   str

class PredictionInput(BaseModel):
    state:       str
    district:    Optional[str] = None
    soil_type:   str
    agro_zone:   Optional[str] = None
    ph:          float
    nitrogen:    float
    phosphorus:  float
    potassium:   float
    ec:          Optional[float] = 0.4
    organic:     Optional[float] = 0.8
    moisture:    Optional[float] = 30.0
    zinc:        Optional[float] = 0.6
    iron:        Optional[float] = 3.2
    manganese:   Optional[float] = 1.1
    copper:      Optional[float] = 0.3
    boron:       Optional[float] = 0.4
    sulphur:     Optional[float] = 12.0
    rainfall:    float
    temperature: float
    humidity:    float

class Top3Item(BaseModel):
    crop:       str
    confidence: float

class PredictionOut(BaseModel):
    id:         Optional[int]
    top_crop:   str
    confidence: float
    top3:       List[Top3Item]
    advice:     str
    created_at: Optional[datetime]

class PredictionRecord(BaseModel):
    id:         int
    state:      Optional[str]
    district:   Optional[str]
    soil_type:  Optional[str]
    top_crop:   str
    confidence: float
    created_at: datetime
    class Config:
        from_attributes = True

class FarmCreate(BaseModel):
    farm_name:  Optional[str] = None
    farm_size:  Optional[float] = None
    state:      Optional[str] = None
    district:   Optional[str] = None
    tehsil:     Optional[str] = None
    soil_type:  Optional[str] = None
    agro_zone:  Optional[str] = None
    latitude:   Optional[float] = None
    longitude:  Optional[float] = None

class FarmOut(FarmCreate):
    id:         int
    user_id:    int
    created_at: datetime
    class Config:
        from_attributes = True
