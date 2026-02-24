import mlflow
import numpy as np
import DoE.simulation_data as sd
import utility
from GPR.GPR_RBF import GPR_RBF
import pickle 
from pathlib import Path
import getopt, sys

REGISTER_MODEL = False

args = sys.argv[1:]
options = "s:"
long_options = ["from_stdout"]

flag_stdout = False

try:
    arguments, values = getopt.getopt(args, options, long_options)
    for currentArg, currentVal in arguments:
        if currentArg in ("-s", "--from_stdout"):
            flag_stdout = True
except getopt.error as err:
    print(str(err)) 

WRITE_PICKLE_FLAG = True
ROOT = Path(".")

model_name = "pyfunc-gaussian-process-reg-model"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"

if WRITE_PICKLE_FLAG:
    PICKLE_PATH = ROOT / "models" /"model.pkl"
    with open(PICKLE_PATH, "wb") as file:
        loaded_pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        pickle.dump(loaded_pyfunc_model, file)

with mlflow.start_run() as run:
    # parameters
    DOUBLE_C_SIZE = 8

    # define constants
    m = 100
    n = 100
    T = 20

    N_x = n+2
    N_y = m+2
    SIZE = N_x*N_y

    # domain definition
    nu_values = [0.01, 0.03, 0.06, 0.07]
    I_values = [1, 5, 20, 50]
    t_values = list(range(0, T, 1))


    A = np.zeros((N_x*N_y, T*len(nu_values)*len(I_values)))


    #sd.generate_data(nu_values, I_values)
    sd.import_data(A, nu_values, I_values, from_stdout=True)
    Uk, Sk, Vk = utility.get_reduced_A(A, return_Uk_Sk_Vk=True)

    a = A.T@Uk
    params_array = sd.get_params_array(nu_values, I_values, t_values)

    sigma = 1e-5

    params = {"sigma": sigma}
    gpr = GPR_RBF(sigma=sigma)
    gpr.fit(params_array, a, Uk)

    mlflow.log_params(params)
    mlflow.log_metrics(gpr.score(params_array, a))

    registered_model_name = None
    if REGISTER_MODEL:
        registered_model_name = "pyfunc-gaussian-process-reg-model"

    mlflow.pyfunc.log_model(
        python_model=gpr,
        name="pyfunc-model",
        input_example=params_array,
        registered_model_name=registered_model_name
    )
    