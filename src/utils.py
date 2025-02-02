import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

from src.exception import CustomException
import dill
from sklearn.model_selection import RandomizedSearchCV

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(x_train,y_train,x_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param_grid = params[list(models.keys())[i]]  # Corrected this line

            random_search = RandomizedSearchCV(model, param_grid, n_iter=10, scoring="r2", cv=5, n_jobs=-1, verbose=2, random_state=42)
            random_search.fit(x_train, y_train)

            model.set_params(**random_search.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score


            return report
    
    except Exception as e:
        raise CustomException(e,sys)