import numpy as np
import subprocess
import matplotlib.pyplot as plt
import scipy.linalg as alg
import sklearn.gaussian_process as gp

# parameters
DOUBLE_C_SIZE = 8
SEED = 42
RANDOM_STATE = np.random.RandomState(SEED)
RANDOM_NUMBER_GENERATOR = np.random.default_rng(RANDOM_STATE)

# define constants
m = 100
n = 100
T = 20

N_x = n+2
N_y = m+2
SIZE = N_x*N_y

# domain definition

nu_values = {0.01, 0.03, 0.06, 0.07}
I_values = {1, 5, 20, 50} 


A = np.empty((N_x*N_y, T*len(nu_values)*len(I_values)))

def generate_data()->None:
    """
        ```generate_data``` calls the C executable to compute data and store it as binary files
    """
    for nu_iter in nu_values:
            for I_iter in I_values:
                subprocess.run(["./main", str(nu_iter), str(I_iter), f"./data/data_nu{nu_iter}_I{I_iter}.bin"])


def import_data(A:np.ndarray, from_stdout:bool=True)->None:
    """
        ```import_data``` imports data either on-the-fly from the executable stdout or from binary files
    """

    # compute data and directly imported but is not stored
    if from_stdout:
        for i, nu_iter in enumerate(nu_values):
            for j, I_iter in enumerate(I_values):
                output = subprocess.run(["./main", str(nu_iter), str(I_iter)], capture_output=True)
                stride = i*len(I_values)*T+j
                for t in range(T):
                    A[:, stride+t: stride+t+1] = np.frombuffer(output.stdout)[t*SIZE: (t+1)*SIZE].reshape(SIZE, 1)

    # import data from binary files
    else:
        for i, nu_iter in enumerate(nu_values):
            for j, I_iter in enumerate(I_values):
                stride = i*j+j
                with open(f"data/data_nu{nu_iter}_I{I_iter}.bin", "br") as f:
                    for t in range(T):
                        A[:, stride*t:stride*(t+1)] += np.frombuffer(f.read(SIZE*DOUBLE_C_SIZE)).reshape(SIZE, 1)


def get_reduced_A(
    A:np.ndarray, threshold:float=0.99, return_U_S_V:bool=False
    )->np.ndarray|tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        ```get_reduced_A``` returns either the reduced form of A with ```threshold```% of variance or the correponding
        decomposition (U,S,D) triplet using the SVD decomposition 
    """

    U, s, V= alg.svd(A)
    S = np.zeros((U.shape[0], V.shape[0]))
    variance_percentage = np.cumsum(s)/np.sum(s)
    i = 0
    while i < len(variance_percentage) and variance_percentage[i]<threshold:
        S[i,i] = s[i]
        i+=1    
    
    if return_U_S_V:
        return U, S, V
    Ak = U@S@V
    return Ak

if __name__ == "__main__":
    #generate_data()
    import_data(A, False)
    Ak = get_reduced_A(A)
    print("A_shape", A.shape)
    print("Ak_shape", Ak.shape)
    #rbf = gp.RBF()
    #gp = gp.GaussianProcessRegressor(rbf, random_state=RANDOM_STATE)

