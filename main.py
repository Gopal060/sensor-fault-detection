from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorCustomException
from sensor.logger import logging
import sys
import os
# from sensor.utils2 import dump_csv_file_to_mongodb_collection
from sensor.pipeline.training_pipeline import TrainPipeline


if __name__=="__main__":
    
    # file_path="/Users/Gopal/sensorlive/aps_failure_training.csv"
    # database_name="sensordata"
    # collection_name ="sensor"
    # dump_csv_file_to_mongodb_collection(file_path,database_name,collection_name)
   
    training_pipeline = TrainPipeline()
    training_pipeline.run_pipeline()
    