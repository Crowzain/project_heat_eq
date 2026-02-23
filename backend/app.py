from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import mlflow.pyfunc


model_name = "pyfunc-gaussian-process-reg-model"
model_version = "latest"

# --- 1 . Load the trained model ---
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

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