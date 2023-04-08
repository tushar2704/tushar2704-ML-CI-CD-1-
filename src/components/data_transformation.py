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

            # Logging
            logging.info(f"Numerical Columns Standard Sclaing Completed: {categorical_columns}")
            logging.info(f"Categorical Columns Encoding Completed: {numerical_columns}")


            # Column Transformer
            preprocessor = ColumnTransformer(
                [
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

