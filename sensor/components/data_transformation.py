import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


from sensor.constant.training_pipeline import TARGET_COLUMN
from sensor.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact,
)

from sensor.entity.config_entity import DataTransformationConfig
from sensor.exception import SensorCustomException
from sensor.logger import logging

from sensor.ml.model.estimator import TargetValueMapping
from sensor.utils.main_utils import save_numpy_array_data, save_object



class DataTransformation:


    def __init__(self,data_validation_artifact: DataValidationArtifact, 
                    data_transformation_config: DataTransformationConfig,):
        """
        The data_validation_artifact : Output reference of data ingestion artifact stage
        The data_transformation_config : configuration for data transformation
        """

        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise SensorCustomException(e, sys)


    # Static method that reads data from a file path and returns a pandas DataFrame.
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            # Return's as pandas dataframe
            return pd.read_csv(file_path)
        
        except Exception as e:
            raise SensorCustomException(e, sys)
        
    # Method to get the transform object 
    @classmethod
    def get_data_transformer_object(cls)->Pipeline:
        """
        Creates a pipeline of transformations, SimpleImputer() for imputing missing values 
        with zeros and RobustScaler() for keep features in the same range 
        and handle outliers.

        """
        try:
            robust_scaler = RobustScaler()
            simple_imputer = SimpleImputer(strategy="constant", fill_value=0)
            preprocessor = Pipeline(
                steps=[
                    ("Imputer", simple_imputer), # replace missing values with zero
                    ("RobustScaler", robust_scaler) # keep every feature in same range and handle outlier
                    ]
            )
            
            # Return preprocessor pipeline for transformation
            return preprocessor

        except Exception as e:
            raise SensorCustomException(e, sys) from e
   


    # Combining all the functions to to start the transformation process step wise.
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            # Read the train and test data
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            # Getting the preprocessor object and store it for further use
            preprocessor = self.get_data_transformer_object()

            # Removes the target column from train_df to get Input features for Model Training
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)

            # Create target variable for training
            target_feature_train_df = train_df[TARGET_COLUMN]

            # Convert target values to numerical format for model training
            # Replaces categorical target values with numerical values using a mapping dictionary.
            target_feature_train_df = target_feature_train_df.replace(TargetValueMapping().to_dict())

            # Create Input features for Testing dataframe for Model Training
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)

            # Create target variable for testing
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            # Convert target values to numerical format for testing
            # Replaces categorical target values with numerical values using the same mapping dictionary.
            target_feature_test_df = target_feature_test_df.replace(TargetValueMapping().to_dict())
            
            # Fit the preprocessor to the training input features for both training and testing dataframe
            """ fit() does not transform the data, it only learns the necessary parameters."""

            preprocessor_object = preprocessor.fit(input_feature_train_df)

            # Transform the train input features using the fitted preprocessor (Imputation and scaling transformation)
            """ transform() method applies a learned transformation to the data."""

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            
            # Transform the test input features using the fitted preprocessor (Imputation and scaling transformation)
            transformed_input_test_feature =preprocessor_object.transform(input_feature_test_df)

            # Handling Imbalance data problem
            """ SMOTETomek combines SMOTE (Synthetic Minority Over-sampling Technique) and Tomek links to balance the dataset. """
            smt = SMOTETomek(sampling_strategy="minority")

            # Applying SMOTE to training data
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                transformed_input_train_feature, target_feature_train_df
            )

            # Applying SMOTE to testing data
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                transformed_input_test_feature, target_feature_test_df
            )
            
            # Combining the data 
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final) ]
            test_arr = np.c_[ input_feature_test_final, np.array(target_feature_test_final) ]

            # Save numpy array data
            save_numpy_array_data( self.data_transformation_config.transformed_train_file_path, array=train_arr, )
            save_numpy_array_data( self.data_transformation_config.transformed_test_file_path,array=test_arr,)

            # Save Preprocessor object for further use
            save_object( self.data_transformation_config.transformed_object_file_path, preprocessor_object,)
            
            
            # preparing artifact for transformed file and preprocessor object
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )

            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}")

            return data_transformation_artifact
        
        
        except Exception as e:
            raise SensorCustomException(e, sys) from e
        
        