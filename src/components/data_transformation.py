# Import required
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Exception & Logging
from src.exception import CustomException
from src.logger import logging

# Data Transformation Class

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path =os.path.join('artifacts', "preprocessor.pkl")


class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns =['writing_score', 'reading_score']
            categorical_columns =[
                'gender', 
                'race_ethnicity',
                'parental_level_of_education',
                'test_preparation_course'
            ]

            # Numerical Pipeline
            num_pipeline =Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())

                ]
            )

            # Categorical Pipeline

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent') ),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler())

                ]
            )
        except:
            pass

