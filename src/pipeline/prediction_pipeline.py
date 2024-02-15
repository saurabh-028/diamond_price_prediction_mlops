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
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model_path = os.path.join("artifacts", "model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            preprocessed_data = preprocessor.transform(data)
            prediction = model.predict(preprocessed_data)

            return prediction

        except Exception as e:
            raise customexception(e,sys)
        
class CustomeData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        
        self.carat=carat
        self.depth=depth
        self.table=table
        self.x=x
        self.y=y
        self.z=z
        self.cut = cut
        self.color = color
        self.clarity = clarity
            
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
                }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df   
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise customexception(e,sys)