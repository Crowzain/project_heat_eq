import matplotlib.pyplot as plt
import numpy as np
import subprocess
import scipy.linalg as alg
import scipy.sparse as sp
import sklearn.gaussian_process as gp


from GPR import GPR
from likelihood import maxlogLikelihood

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

def generate_data(
        nu_values:list[float],
        I_values:list[float]
        )->None:
    """
        ```generate_data``` calls the C executable to compute data and store it as binary files
    """
    for nu_iter in nu_values:
            for I_iter in I_values:
                subprocess.run(["./main", str(nu_iter), str(I_iter), f"./data/data_nu{nu_iter}_I{I_iter}.bin"])
    return None


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

def import_data(
        A:np.ndarray,
        nu_values:list[float],
        I_values:list[float],
        from_stdout:bool|None=None)->None:
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
    return None


def get_reduced_A(
    A:np.ndarray, 
    threshold:float=0.99, 
    return_Uk_Sk_Vk:bool=False
    )->np.ndarray|tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        ```get_reduced_A``` returns either the reduced form Ak of A with ```threshold```% of variance or the correponding
        reduced decomposition (Uk, Sk, Dk) triplet using the SVD decomposition 
    """

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


def generate_example(
        gpr:GPR
    ):
    m, s = gpr.predict_new_ak(get_params_array([0.06], [21], t_values), return_cov=True)
    pred_coeff_upper95 = m + 2*np.reshape(np.sqrt(np.maximum(np.diag(s), 0)), (len(m),1))
    pred_coeff_lower95 = m - 2*np.reshape(np.sqrt(np.maximum(np.diag(s), 0)), (len(m),1))
    pred = Uk@m.T

    # since coefficients are negative, temperature predictions bounds are switched
    pred_temperature_lower95 = Uk@pred_coeff_upper95.T
    pred_temperature_upper95 = Uk@pred_coeff_lower95.T

    return pred, pred_temperature_lower95, pred_temperature_upper95

def plot_3D(
        pred:np.ndarray,
        pred_temperature_lower95:np.ndarray,
        pred_temperature_upper95:np.ndarray,
        )->None:
    for i in range(T):    

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # Plot a basic wireframe.
        xv, yv = np.meshgrid(np.linspace(0, 1, 102), np.linspace(0, 1, 102), indexing='ij')
        ax.plot_wireframe(xv, yv, np.reshape(pred_temperature_lower95[:, i], (102, 102)), rstride=10, cstride=10, alpha=0.3, label="95% confidence interval lower bound")
        ax.plot_wireframe(xv, yv, np.reshape(pred_temperature_upper95[:, i], (102, 102)), rstride=10, cstride=10, color="r", alpha=0.3, label="95% confidence interval upper bound")
        ax.plot_surface(xv, yv, np.reshape(pred[:, i], (102, 102)), alpha=0.15, color="g", label="mean prediction")
        
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_zlabel(r'temperature ($\degree C$)')
        ax.legend()
        plt.show()
    return None


if __name__ == "__main__":
    #generate_data(nu_values, I_values)
    import_data(A, nu_values, I_values, from_stdout=True)
    Uk, Sk, Vk = get_reduced_A(A, return_Uk_Sk_Vk=True)

    a = A.T@Uk
    params_array = get_params_array(nu_values, I_values, t_values)
    
    sigma = 1e-5
    thetas, _ = maxlogLikelihood(params_array, a, sigma=sigma, verbose=False)
    gpr = GPR(gp.kernels.RBF(length_scale=thetas), thetas)

    gpr.fit(params_array, a, sigma=sigma)
    print(gpr.score(params_array, a))
    
    plot_3D(*generate_example(gpr))
    
    
