import mlflow
import pickle
import os

# flag whether to save or not the current model
SAVE_MODEL_FLAG = False

# flag whether to write a pickle file
WRITE_PICKLE_FLAG = True

MODEL_VERSION_ID = "m-cf5a090b2c474171afd2c02d9d1b7f68"
MODEL_PATH = f"mlruns/1/models/{MODEL_VERSION_ID}/artifacts"

if SAVE_MODEL_FLAG:
    mlflow.set_experiment("Basic Model From Code")

    model_info = mlflow.pyfunc.log_model(
        python_model="model.py",  # Path to your script
        name="gpr_model"
    )

loaded_pyfunc_model = mlflow.pyfunc.load_model(MODEL_PATH)

if WRITE_PICKLE_FLAG:
    PICKLE_PATH = os.path.join("models", "model.pkl")
    with open(PICKLE_PATH, "wb") as file:
        pickle.dump(loaded_pyfunc_model, file)