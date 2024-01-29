import pandas as ps
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
    def __init__(self):
        raw_data_path:str =o s.path.join("artifacts","raw.csv")
        train_data_path:str = os.path.join("artifacts","train.csv")
        test_data_path:str = os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self) -> None:
        pass
    def ingestion():
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e,sys)