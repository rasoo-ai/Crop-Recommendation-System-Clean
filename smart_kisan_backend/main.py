from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta
from typing import List
import database, models, schemas, auth, predict

app = FastAPI(title="Smart Kisan API", description="AI-powered crop recommendation backend", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
models.Base.metadata.create_all(bind=database.engine)

@app.get("/", tags=["Health"])
def root():
    return {"status": "running", "app": "Smart Kisan API", "version": "1.0.0"}

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}

@app.post("/register", response_model=schemas.UserOut, tags=["Auth"])
def register(user: schemas.UserCreate, db: Session = Depends(database.get_db)):
    existing = db.query(models.User).filter(models.User.email == user.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    new_user = models.User(name=user.name, email=user.email, hashed_password=auth.hash_password(user.password), state=user.state, district=user.district)
    db.add(new_user); db.commit(); db.refresh(new_user)
    return new_user

@app.post("/login", response_model=schemas.Token, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.email == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect email or password")
    token = auth.create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(days=7))
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me", response_model=schemas.UserOut, tags=["User"])
def get_me(current_user: models.User = Depends(auth.get_current_user)):
    return current_user

@app.put("/me", response_model=schemas.UserOut, tags=["User"])
def update_me(updates: schemas.UserUpdate, db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_user)):
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(current_user, field, value)
    db.commit(); db.refresh(current_user)
    return current_user

@app.post("/predict", response_model=schemas.PredictionOut, tags=["Prediction"])
def crop_predict(data: schemas.PredictionInput, db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_user)):
    result = predict.run_prediction(data)
    record = models.Prediction(user_id=current_user.id, state=data.state, district=data.district, soil_type=data.soil_type, ph=data.ph, nitrogen=data.nitrogen, phosphorus=data.phosphorus, potassium=data.potassium, rainfall=data.rainfall, temperature=data.temperature, humidity=data.humidity, top_crop=result["top_crop"], confidence=result["confidence"], top3=str(result["top3"]))
    db.add(record); db.commit(); db.refresh(record)
    return {**result, "id": record.id, "created_at": record.created_at}

@app.post("/predict/guest", response_model=schemas.PredictionOut, tags=["Prediction"])
def crop_predict_guest(data: schemas.PredictionInput):
    result = predict.run_prediction(data)
    return {**result, "id": None, "created_at": None}

@app.get("/history", response_model=List[schemas.PredictionRecord], tags=["History"])
def get_history(db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_user), limit: int = 20):
    return db.query(models.Prediction).filter(models.Prediction.user_id == current_user.id).order_by(models.Prediction.created_at.desc()).limit(limit).all()

@app.delete("/history/{prediction_id}", tags=["History"])
def delete_prediction(prediction_id: int, db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_user)):
    record = db.query(models.Prediction).filter(models.Prediction.id == prediction_id, models.Prediction.user_id == current_user.id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Prediction not found")
    db.delete(record); db.commit()
    return {"message": "Deleted successfully"}

@app.post("/farm", response_model=schemas.FarmOut, tags=["Farm"])
def create_farm(farm: schemas.FarmCreate, db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_user)):
    existing = db.query(models.Farm).filter(models.Farm.user_id == current_user.id).first()
    if existing:
        raise HTTPException(status_code=400, detail="Farm profile already exists. Use PUT to update.")
    new_farm = models.Farm(user_id=current_user.id, **farm.dict())
    db.add(new_farm); db.commit(); db.refresh(new_farm)
    return new_farm

@app.get("/farm", response_model=schemas.FarmOut, tags=["Farm"])
def get_farm(db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_user)):
    farm = db.query(models.Farm).filter(models.Farm.user_id == current_user.id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="No farm profile found")
    return farm

@app.put("/farm", response_model=schemas.FarmOut, tags=["Farm"])
def update_farm(updates: schemas.FarmCreate, db: Session = Depends(database.get_db), current_user: models.User = Depends(auth.get_current_user)):
    farm = db.query(models.Farm).filter(models.Farm.user_id == current_user.id).first()
    if not farm:
        raise HTTPException(status_code=404, detail="No farm profile found. Use POST to create.")
    for field, value in updates.dict(exclude_unset=True).items():
        setattr(farm, field, value)
    db.commit(); db.refresh(farm)
    return farm

@app.get("/stats", tags=["Stats"])
def get_stats(db: Session = Depends(database.get_db)):
    return {"total_users": db.query(models.User).count(), "total_predictions": db.query(models.Prediction).count(), "model_accuracy": 0.9353, "model_version": "V1 - Random Forest"}
