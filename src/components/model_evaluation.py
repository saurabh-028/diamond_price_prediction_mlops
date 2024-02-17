import os
import sys

import pandas as ps
import numpy as np
import mlflow
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.logger.logfile import logging
from src.exception.exception import customexception
from urllib.parse import urlparse
from src.utils.utils import load_object

class ModelEvaluation:
    def __init__(self) -> None:
        logging.info("Model Evaluation initiated")
    
    def eval_metrics(self, actual, preds):
        rmse = mean_squared_error(preds, actual) ** 0.5
        mae = mean_absolute_error(preds, actual)
        r_squared = r2_score(preds, actual)

        return rmse, mae, r_squared
    
    def model_evaluate(self, X_test, y_test):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)

            logging.info("model loaded Successfully")

            # mlflow.set_registry_uri("")

            tracking_url_type_store=urlparse(mlflow.get_registry_uri()).scheme

            print(tracking_url_type_store)
            
            with mlflow.start_run():
                preds = model.predict(X_test)
                rmse,mae,r_squared = self.eval_metrics(y_test, preds)

                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("R_squared", r_squared)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
                logging.info("Model Evaluation Successfull")

        except Exception as e:
            logging.info()
            raise customexception(e,sys)
