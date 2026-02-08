from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel, ConfigDict

app = FastAPI()

model = joblib.load("models/insurance_xgboost_model.pkl")

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str
    
    model_config = ConfigDict(extra='allow')

@app.post("/predict")
def predict(data: InsuranceInput):
    try:
        df = pd.DataFrame([data.model_dump()])
        
        prediction = model.predict(df)
        
        return {"prediction": float(np.expm1(prediction[0]))}
    except Exception as e:
        return {"error": str(e)}