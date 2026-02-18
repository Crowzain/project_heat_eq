import sklearn.gaussian_process as gp
import numpy as np
import scipy.linalg as linalg


class GPR_RBF:
    def __init__(self, theta:int|np.ndarray=1):
        self.theta = theta
        self.kernel = gp.kernels.RBF(self.theta)


    def fit(
            self,
            X_obs:np.ndarray,
            y:np.ndarray,
            sigma:float=0
            )->None:
        n_rows = X_obs.shape[0]
    
        # kernel matrix
        self.X_obs = X_obs
        self.K = np.eye(n_rows)*sigma+self.kernel(self.X_obs, self.X_obs)
        c, low = linalg.cho_factor(self.K)

        # matrix product K^{-1} by y
        self.invK_y = linalg.cho_solve((c, low), y)


    def predict(
            self,
            x:np.ndarray,
            return_cov:bool=False
            )->tuple[np.ndarray, np.ndarray]|np.ndarray:
        
        mean = self.kernel(x, self.X_obs)@self.invK_y
        if return_cov:
            cov = self.kernel(x, x) - \
                self.kernel(x, self.X_obs)@linalg.solve(self.K, self.kernel(self.X_obs, x))
            return mean, cov
        return mean
    
    def score(self, X_input:np.ndarray, y:np.ndarray)->dict[str, float]:
        prediction = self.predict(X_input)
        y_len = len(y)

        return {
            "MAE":np.sum(np.abs(prediction-y))/y_len,
            "MSE":np.sum((prediction-y)**2)/y_len
        }