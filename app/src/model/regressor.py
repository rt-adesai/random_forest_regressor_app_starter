#Import required libraries
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
warnings.filterwarnings('ignore') 

from sklearn.ensemble import RandomForestRegressor


model_fname = "model.save"
MODEL_NAME = "reg_base_random_forest_sklearn"



class Regressor(): 
    
    def __init__(self, n_estimators = 250, max_features = 3, max_samples = 0.7, **kwargs) -> None:
        self.n_estimators = int(n_estimators)
        self.max_features = int(max_features)
        self.max_samples= np.float(max_samples)
        
        self.model = self.build_model()
        
        
        
    def build_model(self): 
        model = RandomForestRegressor(
            n_estimators= self.n_estimators, 
            max_features= self.max_features,
            max_samples= self.max_samples, 
            random_state=42, 
            bootstrap= True, 
            oob_score= True, 
            n_jobs=-1, 
            verbose=0
        )
        return model
    
    
    def fit(self, train_X, train_y):        
                 
    
        self.model.fit(
                X = train_X,
                y = train_y
            )
    
    
    def predict(self, X,): 
        preds = self.model.predict(X)
        return preds 
    

    def summary(self):
        self.model.get_params()
        
    
    def evaluate(self, x_test, y_test): 
        """Evaluate the model and return the loss and metrics"""
        if self.model is not None:
            return self.model.score(x_test, y_test)        

    
    def save(self, model_path): 
        joblib.dump(self, os.path.join(model_path, model_fname), compress=4)

    @classmethod
    def load(cls, model_path): 
        rf = joblib.load(os.path.join(model_path, model_fname))
        return rf


def save_model(model, model_path):    
    model.save(model_path) 
    

def load_model(model_path): 
    model = Regressor.load(model_path)     
    return model



def get_data_based_model_params(data): 
    ''' 
        Set any model parameters that are data dependent. 
        For example, number of layers or neurons in a neural network as a function of data shape.
    '''  
    return {"max_features": max(1, int(0.5 *data.shape[1]))}
