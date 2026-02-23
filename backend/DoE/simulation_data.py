import subprocess
import numpy as np
import pathlib

# parameters
DOUBLE_C_SIZE = 8

m = 100
n = 100
T = 20

N_x = n+2
N_y = m+2
SIZE = N_x*N_y


ROOT = pathlib.Path(".")
EXE_PATH = None
WDIR = None

# Use next() with generator expressions for efficiency
exe_path_iterator = pathlib.Path.rglob(ROOT, "*main")
exe_path = next(exe_path_iterator, None)

if exe_path:
    EXE_PATH = exe_path
    WDIR = EXE_PATH.parent
else:
    make_path_iterator = pathlib.Path.rglob(ROOT, "*Makefile")
    make_path = next(make_path_iterator, None)
    if make_path:
        WDIR = make_path.parent


def generate_data(
        nu_values:list[float],
        I_values:list[float],
        )->None:
    """
        ```generate_data``` calls the C executable to compute data and store it as binary files
    """

    for nu_iter in nu_values:
            for I_iter in I_values:
                subprocess.run([EXE_PATH, str(nu_iter), str(I_iter), 
                                os.path.join(WDIR, f"./data/data_nu{nu_iter}_I{I_iter}.bin")])
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
                output = subprocess.run([EXE_PATH, str(nu_iter), str(I_iter)], capture_output=True)
                stride = i*len(I_values)*T+j*T
                for t in range(T):
                    A[:, stride+t: stride+t+1] = np.frombuffer(output.stdout)[t*SIZE: (t+1)*SIZE].reshape(SIZE, 1)

    # import data from binary files
    else:
        for i, nu_iter in enumerate(nu_values):
            for j, I_iter in enumerate(I_values):
                stride = i*len(I_values)*T+j*T
                with open(WDIR / f"./data/data_nu{nu_iter}_I{I_iter}.bin", "br") as f:
                    for t in range(T):
                        A[:, stride+t: stride+t+1] += np.frombuffer(f.read(SIZE*DOUBLE_C_SIZE)).reshape(SIZE, 1)
    return None