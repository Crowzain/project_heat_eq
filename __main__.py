import numpy as np
import matplotlib.pyplot as plt
import subprocess
import copy

# define constants
m = 100
n = 100
T = 20

N_x = n+2
N_y = m+2
size = N_x*N_y

# domain definition

nu_values = {0.01, 0.03, 0.06, 0.07}
I_values = {1, 5, 20, 20} 


data = np.empty((N_x*N_y, len(nu_values)*len(I_values)))

for i, nu_iter in enumerate(nu_values):
    for j, I_iter in enumerate(I_values):
        output = subprocess.run(["./main", str(nu_iter), str(I_iter)], capture_output=True)
        stride = i*j+j
        for t in range(T):
            data[:, stride*t:stride*(t+1)] += np.frombuffer(output.stdout)[t*size: (t+1)*size].reshape(size, 1)


