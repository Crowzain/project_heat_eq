import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as alg
import scipy.sparse as sp

T = 20

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

#plot_3D(*generate_example(gpr))

