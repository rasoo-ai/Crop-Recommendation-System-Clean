from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base

class User(Base):
    __tablename__ = "users"
    id              = Column(Integer, primary_key=True, index=True)
    name            = Column(String, nullable=False)
    email           = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    state           = Column(String, nullable=True)
    district        = Column(String, nullable=True)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    predictions     = relationship("Prediction", back_populates="user")
    farm            = relationship("Farm", back_populates="user", uselist=False)

class Prediction(Base):
    __tablename__ = "predictions"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    state       = Column(String)
    district    = Column(String)
    soil_type   = Column(String)
    ph          = Column(Float)
    nitrogen    = Column(Float)
    phosphorus  = Column(Float)
    potassium   = Column(Float)
    rainfall    = Column(Float)
    temperature = Column(Float)
    humidity    = Column(Float)
    top_crop    = Column(String)
    confidence  = Column(Float)
    top3        = Column(Text)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    user        = relationship("User", back_populates="predictions")

class Farm(Base):
    __tablename__ = "farms"
    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    farm_name   = Column(String)
    farm_size   = Column(Float)
    state       = Column(String)
    district    = Column(String)
    tehsil      = Column(String)
    soil_type   = Column(String)
    agro_zone   = Column(String)
    latitude    = Column(Float)
    longitude   = Column(Float)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())
    updated_at  = Column(DateTime(timezone=True), onupdate=func.now())
    user        = relationship("User", back_populates="farm")
