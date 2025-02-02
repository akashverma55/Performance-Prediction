import os
import sys 
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            xtrain,ytrain,xtest,ytest = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1],
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Linear Regression": {},
                "Lasso": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
                "Ridge": {"alpha": [0.001, 0.01, 0.1, 1, 10]},
                "K-Neighbors Regressor": {"n_neighbors": [3, 5, 7, 9]},
                "Decision Tree": {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10]},
                "Random Forest": {"n_estimators": [10, 50, 100], "max_depth": [None, 10, 20]},
                "Gradient Boosting": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.2]},
                "AdaBoost Regressor": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
            }

            model_report:dict = evaluate_models(x_train=xtrain,y_train=ytrain,x_test=xtest,y_test=ytest,models=models,params=params)

            # to get the best model score from dict
            best_model_score = max(sorted(model_report.values()))

            # to get the best model name from dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            else:
                logging.info(f"Best found model on both training and testing dataset is {best_model_name}")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(xtest)
            r2_square = r2_score(ytest,predicted)
            
            return r2_square

        except Exception as e:
            raise CustomException(e,sys)