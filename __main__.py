import matplotlib.pyplot as plt
import numpy as np
import DoE.simulation_data as sd
import scipy.linalg as alg
import scipy.sparse as sp

from GPR.GPR_RBF import GPR_RBF
from GPR.likelihood import maxlogLikelihood

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
        gpr:GPR_RBF
    ):
    m, s = gpr.predict(sd.get_params_array([0.06], [21], t_values), return_cov=True)
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
    #sd.generate_data(nu_values, I_values)
    sd.import_data(A, nu_values, I_values, from_stdout=True)
    Uk, Sk, Vk = get_reduced_A(A, return_Uk_Sk_Vk=True)

    a = A.T@Uk
    params_array = sd.get_params_array(nu_values, I_values, t_values)
    
    sigma = 1e-5
    thetas, _ = maxlogLikelihood(params_array, a, sigma=sigma, verbose=False)
    gpr = GPR_RBF(thetas)

    gpr.fit(params_array, a, sigma=sigma)
    print(gpr.score(params_array, a))
    
    plot_3D(*generate_example(gpr))
    
    
