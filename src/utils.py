import os
import sys
import dill 

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path , obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , 'wb') as file_obj:
            dill.dump(obj , file_obj)

    except Exception as e:
        raise CustomException(e,sys)        
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param_grid):
    try:
        report = {}

        for model_name, model in models.items():
            print(f"\n🔍 Evaluating: {model_name}")
            params = param_grid.get(model_name, {})

            gs = GridSearchCV(model, params, cv=3, scoring='r2', n_jobs=-1)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            print(f" {model_name}  Train R2: {train_score:.4f} | Test R2: {test_score:.4f}")
            print(f"Best Params: {gs.best_params_}")

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

    

def load_object(file_path):
    try:
        with open (file_path , 'rb') as file_obj :
            return dill.load(file_obj)    
    except Exception as e:
        raise CustomException(e,sys)    