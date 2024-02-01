import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import os
import sys
from src.logger.logfile import logging
from src.exception.exception import customexception
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    raw_data_path:str = os.path.join("artifacts","raw.csv")
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def ingestion(self):
        logging.info("data ingestion started")
        try:
            data = pd.read_csv("C:/Saurabh/mlops ineuron/End to End Project/Data/train.csv")
            logging.info("reading data")

            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path))),
            data.to_csv(self.ingestion_config.raw_data_path)

            train_data, test_data = train_test_split(data, test_size=0.25)
            logging.info("data split completed")

            train_data.to_csv(self.ingestion_config.train_data_path)
            test_data.to_csv(self.ingestion_config.test_data_path)

            logging.info("data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info("here is the error")
            raise customexception(e,sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.ingestion()