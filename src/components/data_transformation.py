import pandas as pd
import numpy as np

import os
import sys
from src.logger.logfile import logging
from src.exception.exception import customexception
from dataclasses import dataclass
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OrdinalEncoder 

from sklearn.pipeline  import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.utils import save_object

@dataclass
class DataTansformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transfrormation_config = DataTansformationConfig()
    
    def get_data_transformation(self):
        try:
            logging.info("data transformation initiated")
            
            cat_cols = ['cut', 'color', 'clarity']
            num_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info("Pipeline Initiated")
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer()),
                    ("scaler", StandardScaler())
                ]   
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(categories=[cut_categories, color_categories,clarity_categories]))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("cat_cols", cat_pipeline, cat_cols),
                    ("num_cols", num_pipeline, num_cols)    
                ]
            )
            
            return preprocessor


        except Exception as e:
            logging.info("Exception occured in the get_data_transformation")
            raise customexception(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("read train and test data complete")
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            preprocessor = self.get_data_transformation()

            target_column = "price"
            X_train = train_df.drop([target_column, "id"], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop([target_column, "id"], axis=1)
            y_test = test_df[target_column]

            logging.info("Applying preprocessing object on training and testing datasets.")
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            save_object(
                file_path=self.data_transfrormation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )

            logging.info("preprocessing pickle file saved")

            return (
                X_train,
                X_test,
                y_train,
                y_test
            )

        except Exception as e:
            logging.info("Exeption occured while initiating data transformation")
            raise customexception(e, sys)
