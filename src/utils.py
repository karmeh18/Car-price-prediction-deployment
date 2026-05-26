import os
import sys

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from src.exception import Custom_Exception

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise Custom_Exception(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models):
    try:
        report={}
        for i in range(len(list(models))):
            model_name=list(models.keys())[i]
            model=list(models.values())[i]
            model.fit(X_train,y_train)
            pred=model.predict(X_test)
            score=calculate_mape(y_test,pred)
            report[model_name]=score
        return report
    except Exception as e:
        raise Custom_Exception(e,sys)
    
def calculate_mape(actual, predicted):
    """
    Calculate Mean Absolute Percentage Error (MAPE) between actual and predicted values.

    Parameters:
    actual : array-like, shape (n_samples,)
        Array containing actual values.
    predicted : array-like, shape (n_samples,)
        Array containing predicted values.

    Returns:
    mape : float
        Mean Absolute Percentage Error (MAPE) value.
    """
    # Ensure actual and predicted arrays have the same length
    if len(actual) != len(predicted):
        raise ValueError("Length of actual and predicted arrays must be the same.")

    # Calculate absolute percentage error for each sample
    ape = abs((actual - predicted) / actual)

    # Calculate mean absolute percentage error
    mape = ape.mean() * 100

    return mape

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise Custom_Exception(e,sys)
