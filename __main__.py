import numpy as np
import subprocess
import scipy.linalg as alg
import scipy.sparse as sp
import sklearn.gaussian_process as gp
from GPR import GPR
import matplotlib.pyplot as plt

# parameters
DOUBLE_C_SIZE = 8

# set random settings
SEED = 42
BIT_GENERATOR = np.random.PCG64(SEED)
RANDOM_STATE = np.random.RandomState(BIT_GENERATOR)
RANDOM_NUMBER_GENERATOR = np.random.default_rng(BIT_GENERATOR)

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

def generate_data()->None:
    """
        ```generate_data``` calls the C executable to compute data and store it as binary files
    """
    for nu_iter in nu_values:
            for I_iter in I_values:
                subprocess.run(["./main", str(nu_iter), str(I_iter), f"./data/data_nu{nu_iter}_I{I_iter}.bin"])


def import_data(A:np.ndarray, from_stdout:bool|None=None)->None:
    """
        ```import_data``` imports data either on-the-fly from the executable stdout or from binary files
    """

    # compute data and directly imported but is not stored
    
    if from_stdout is None: from_stdout = True

    if from_stdout:
        for i, nu_iter in enumerate(nu_values):
            for j, I_iter in enumerate(I_values):
                output = subprocess.run(["./main", str(nu_iter), str(I_iter)], capture_output=True)
                stride = i*len(I_values)*T+j*T
                for t in range(T):
                    A[:, stride+t: stride+t+1] = np.frombuffer(output.stdout)[t*SIZE: (t+1)*SIZE].reshape(SIZE, 1)

    # import data from binary files
    else:
        for i, nu_iter in enumerate(nu_values):
            for j, I_iter in enumerate(I_values):
                stride = i*len(I_values)*T+j*T
                with open(f"data/data_nu{nu_iter}_I{I_iter}.bin", "br") as f:
                    for t in range(T):
                        A[:, stride+t: stride+t+1] += np.frombuffer(f.read(SIZE*DOUBLE_C_SIZE)).reshape(SIZE, 1)


def get_reduced_A(
    A:np.ndarray, threshold:float=0.99, return_Uk_Sk_Vk:bool|None=None
    )->np.ndarray|tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        ```get_reduced_A``` returns either the reduced form Ak of A with ```threshold```% of variance or the correponding
        reduced decomposition (Uk, Sk, Dk) triplet using the SVD decomposition 
    """

    if return_Uk_Sk_Vk is None: return_Uk_Sk_Vk = False

    U, s, V= alg.svd(A)
    variance_percentage = np.cumsum(s)/np.sum(s)
    i = 0
    while i < len(variance_percentage) and variance_percentage[i]<threshold:
        i+=1
    S = sp.diags_array(s[:i+1])
    if return_Uk_Sk_Vk:
        return U[:,:i+1], S, V[:i+1,:]
    Ak = U@S@V
    return Ak


def get_params_array(nu_values:list[float], I_values:list[float], t_values:list[float],)->np.ndarray:
    nu_val_len = len(nu_values)
    I_val_len = len(I_values)


    params_array = np.zeros((nu_val_len*I_val_len*T, 3))
    for i, nu_iter in enumerate(nu_values):
        for j, I_iter in enumerate(I_values):
            stride = i*len(I_values)*T+j*T
            for k, t_iter in enumerate(t_values):
                params_array[stride+k,:] += [nu_iter, I_iter, t_iter]
    return params_array


if __name__ == "__main__":
    #generate_data()
    import_data(A, from_stdout=True)
    Uk, Sk, Vk = get_reduced_A(A, return_Uk_Sk_Vk=True)

    a = A.T@Uk
    params_array = get_params_array(nu_values, I_values, t_values)
    gpr = GPR(gp.kernels.RBF())
    gpr.train(params_array, a, sigma=0)
    #print(gpr.predict_new_ak(params_array))
    print(gpr.score(params_array, a))
    test = Uk@gpr.predict_new_ak(get_params_array([0.01], [19], t_values)).T
    for i in range(T):    
        plt.imshow(np.reshape(test[:, i], (102, 102)))
        plt.colorbar()
        plt.show()
