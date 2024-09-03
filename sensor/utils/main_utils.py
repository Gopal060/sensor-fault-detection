import yaml
import pandas as pd
import numpy as np
import os
import dill
import sys
from sensor.exception import SensorCustomException
from sensor.logger import logging


def read_yaml_file(file_path:str)->dict:
    try:

        with open(file_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        

    except Exception as e:
        raise SensorCustomException(e,sys)
    


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise SensorCustomException(e, sys) 
    

def save_numpy_array_data(file_path: str, array: np.array):
    """ Save numpy array data to file_path: str this location """

    try:

        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise SensorCustomException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """ Load the numpy array data form file_path: str location and return np.array loaded data."""

    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise SensorCustomException(e, sys) from e
    


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of main utils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        
        logging.info("Exited the save_object method form main utils class")
    
    except Exception as e:
        raise SensorCustomException(e, sys) from e 
    

def load_object(file_path: str) -> object:
    
    try:
        # Checks if the file path is exists
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not not exists.")
        
        # Open the file and load the file object
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
        
    except Exception as e:
        raise SensorCustomException(e, sys)