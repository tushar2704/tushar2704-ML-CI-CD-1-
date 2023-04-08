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
        '''
        This function is responsible for Data Transformation
        '''
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
        


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df =pd.read_csv(train_path)
            test_df =pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name ='math_score'
            numerical_columns =['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = test_df[target_column_name]

            input_feature_test_df =test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df =test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr =preprocessor_obj.transform(input_feature_train_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)]
            
            test_arr = np.c_[
                input_feature_test_arr np.array(target_feature_test_df)]
            logging.info(f"saved preprocessing object")

            save_object(
                file_path =self.data_transformation_config.preprocessor_ob_file_path,
                obj =preprocessor_obj)


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )


        except:
            pass


