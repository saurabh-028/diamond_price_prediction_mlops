import pandas as ps
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

@dataclass
class DataIngestionConfig:
    pass

class DataIngestion:
    def __init__(self) -> None:
        pass
    def ingestion():
        try:
            pass
        except Exception as e:
            logging.info()
            raise customexception(e,sys)