from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from pathlib import Path

ROOT = Path(".")

# --- 1 . Load the trained model ---
with open(ROOT / "models" / "model.pkl", "rb") as f:
    model = pickle.load(f)

# --- 2. Create FastAPI app --
app = FastAPI(title="Heat Equation Gaussian Process Interpolation API")

# --- 3. Define request model ---
class SimulationParams(BaseModel):
    nu: float
    I: float
    t: float

class Results(BaseModel):
    mean: list[float]
    pred_temperature_lower95: list[float]
    pred_temperature_upper95: list[float]

# --- 4. Define prediction endpoint ---
@app.post("/predict")
def predict_temp(params:SimulationParams):
    X_input = np.array([[params.nu, params.I, params.t]])
    mean, pred_temperature_lower95, pred_temperature_upper95 = model.predict(X_input)
    output = Results(mean=mean, pred_temperature_lower95=pred_temperature_lower95, pred_temperature_upper95=pred_temperature_upper95)
    return output

# --- 5. Root endpoint ---
def root():
    return {"message": "helloworld"}