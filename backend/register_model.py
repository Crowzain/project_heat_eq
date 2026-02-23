import mlflow
import numpy as np
import DoE.simulation_data as sd
import utility
from GPR.GPR_RBF import GPR_RBF

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
    sd.import_data(A, nu_values, I_values, from_stdout=False)
    Uk, Sk, Vk = utility.get_reduced_A(A, return_Uk_Sk_Vk=True)

    a = A.T@Uk
    params_array = sd.get_params_array(nu_values, I_values, t_values)

    sigma = 1e-5

    params = {"sigma": sigma}
    gpr = GPR_RBF(sigma=sigma)
    gpr.fit(params_array, a, Uk)

    mlflow.log_params(params)
    mlflow.log_metrics(gpr.score(params_array, a))
    mlflow.pyfunc.log_model(
        python_model=gpr,
        name="pyfunc-model",
        input_example=params_array,
        registered_model_name="pyfunc-gaussian-process-reg-model"
    )
    