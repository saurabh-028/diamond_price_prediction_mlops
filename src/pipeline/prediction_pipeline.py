import os
import sys
from src.logger.logfile import logging
from src.exception.exception import customexception
import pandas as pd
from src.utils.utils import load_object

class PredictPipeline:
    def __init__(self) -> None:
        print("init the object")

    def predict(self,data):
        try:
            preprocessor_path = os.path.join("artifatct", "preprocessor.pkl")
            model_path = os.path.join("artifatct", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            preprocessed_data = preprocessor.transform(data)
            prediction = model.predict(preprocessed_data)

            return prediction

        except Exception as e:
            raise customexception(e,sys)
        
class MakeDataframe:
    def __init__(self) -> None:
        pass
    def get_data_as_dataframe(self):
        pass