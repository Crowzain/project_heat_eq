import sklearn.gaussian_process as gp
import numpy as np
import scipy.linalg as linalg
import mlflow.pyfunc
from typing import Tuple, Union, Dict


class GPR_RBF(mlflow.pyfunc.PythonModel):
    def __init__(self, theta:Union[np.ndarray, None]=None, sigma:int=0):
        if theta is None:
            self.theta = np.ones((1,3))
        else:
            self.theta = theta
        self.kernel = gp.kernels.RBF(self.theta)
        self.sigma = sigma


    def fit(
            self,
            X_obs:np.ndarray,
            y:np.ndarray,
            Uk:np.ndarray
            )->None:
        n_rows = X_obs.shape[0]
        self.Uk = Uk
    
        # kernel matrix
        self.X_obs = X_obs
        self.K = np.eye(n_rows)*self.sigma+self.kernel(self.X_obs, self.X_obs)
        c, low = linalg.cho_factor(self.K)

        # matrix product K^{-1} by y
        self.invK_y = linalg.cho_solve((c, low), y)



    def predict_coeff(
            self,
            model_input:np.ndarray
            )->Tuple[np.ndarray, np.ndarray]:
        
        mean = self.kernel(model_input, self.X_obs)@self.invK_y
        cov = self.kernel(model_input, model_input) - \
            self.kernel(model_input, self.X_obs)@linalg.solve(self.K, self.kernel(self.X_obs, model_input))
        return mean, cov
    
    def predict(
            self,
            model_input:np.ndarray
        )->Tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        m, s = self.predict_coeff(model_input)

        pred_coeff_upper95 = m + 2*np.reshape(np.sqrt(np.maximum(np.diag(s), 0)), (len(m),1))
        pred_coeff_lower95 = m - 2*np.reshape(np.sqrt(np.maximum(np.diag(s), 0)), (len(m),1))
        pred = self.Uk@m.T

        # since coefficients are negative, temperature predictions bounds are switched
        pred_temperature_lower95 = self.Uk@pred_coeff_upper95.T
        pred_temperature_upper95 = self.Uk@pred_coeff_lower95.T

        return pred, pred_temperature_lower95, pred_temperature_upper95

    
    def score(self, X_input:np.ndarray, y:np.ndarray)->Dict[str, float]:
        prediction, _ = self.predict_coeff(X_input)
        y_len = len(y)

        return {
            "MAE":np.sum(np.abs(prediction-y))/y_len,
            "MSE":np.sum((prediction-y)**2)/y_len
        }
    