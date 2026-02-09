import numpy as np
import subprocess
import sys

# define constants
m = 100
n = 100
T = 20

N_x = n+2
N_y = m+2
SIZE = N_x*N_y

DOUBLE_C_SIZE = 8

# domain definition

nu_values = {0.01, 0.03, 0.06, 0.07}
I_values = {1, 5, 20, 20} 


data = np.empty((N_x*N_y, len(nu_values)*len(I_values)))

def generate_data()->None:
    for nu_iter in nu_values:
            for I_iter in I_values:
                subprocess.run(["./main", str(nu_iter), str(I_iter), f"./data/data_nu{nu_iter}_I{I_iter}.bin"])


def import_data(from_stdout:bool=True)->None:
    # compute data and directly imported but is not stored
    if from_stdout:
        for i, nu_iter in enumerate(nu_values):
            for j, I_iter in enumerate(I_values):
                output = subprocess.run(["./main", str(nu_iter), str(I_iter)], capture_output=True)
                stride = i*j+j
                for t in range(T):
                    data[:, stride*t:stride*(t+1)] += np.frombuffer(output.stdout)[t*SIZE: (t+1)*SIZE].reshape(SIZE, 1)
    
    else:
        for i, nu_iter in enumerate(nu_values):
            for j, I_iter in enumerate(I_values):
                stride = i*j+j
                with open(f"data/data_nu{nu_iter}_I{I_iter}.bin", "br") as f:
                    for t in range(T):
                        data[:, stride*t:stride*(t+1)] += np.frombuffer(f.read(SIZE*DOUBLE_C_SIZE)).reshape(SIZE, 1)


if __name__ == "__main__":
    #generate_data()
    import_data(False)
    print(data)


