import numpy as np
import pandas as pd
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
import os, sys, pickle
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest" : RandomForestRegressor(),
                "Xg boost": XGBRegressor(),
                "Cat Boost" : CatBoostRegressor(),
                "Ada Boost" : AdaBoostRegressor(),
                "Gradient Boosting" : GradientBoostingRegressor(),
            }
            params = {
                "Linear Regression": {},
                "K-Neighbors Regressor":{
                    "n_neighbors": [1, 2, 5, 7, 8],
                    "weights":['uniform', 'distance'],
                    "algorithm" : ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                "Decision Tree": {
                    "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    "splitter": ['best', 'random'],
                    "max_features": ['sqrt', 'log2']
                },
                "Random Forest": {
                    "n_estimators": [50,100,150,200],
                    "criterion": ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
                    
                },
                "Xg boost":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Cat Boost":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },
                "Ada Boost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                }

            }



            model_report:dict = evaluate_model(X_train=X_train, y_train=y_train,X_test=X_test,y_test=y_test, models= models, param = params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found")
            logging.info("Best found model on both training and testing")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            return r2


        except Exception as e:
            raise CustomException(e, sys)