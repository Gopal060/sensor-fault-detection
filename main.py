from sensor.pipeline.training_pipeline import TrainPipeline
from sensor.utils.main_utils import load_object
from sensor.ml.model.estimator import ModelResolver,TargetValueMapping
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.exception import SensorCustomException
import os,sys
from sensor.logger import logging
from sensor.pipeline import training_pipeline
from sensor.pipeline.training_pipeline import TrainPipeline
import os
from sensor.utils.main_utils import read_yaml_file
from sensor.constant.training_pipeline import SAVED_MODEL_DIR


from  fastapi import FastAPI
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, File, UploadFile, Response
import pandas as pd

from  fastapi import FastAPI
from sensor.constant.application import APP_HOST, APP_PORT
from starlette.responses import RedirectResponse
from uvicorn import run as app_run
from fastapi.responses import Response
from sensor.utils.main_utils import load_object
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi import FastAPI, File, UploadFile, Response
import pandas as pd


app = FastAPI()



origins = ["*"]
#Cross-Origin Resource Sharing (CORS) 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/",tags=["authentication"])
async def  index():
    return RedirectResponse(url="/docs")





@app.get("/train")
async def train():
    try:

        training_pipeline = TrainPipeline()

        if training_pipeline.is_pipeline_running:
            return Response("Training pipeline is already running.")
        
        training_pipeline.run_pipeline()
        return Response("Training successfully completed!")
    
    except Exception as e:
        return Response(f"Error Occurred! {e}")
        


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded CSV file
        df = pd.read_csv(file.file)
        
    
        model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
        if not model_resolver.is_model_exists():
            return Response("Model is not available")
        
        best_model_path = model_resolver.get_best_model_path()
        model = load_object(file_path=best_model_path)
        y_pred = model.predict(df)
        df['predicted_column'] = y_pred
        df['predicted_column'].replace(TargetValueMapping().reverse_mapping(), inplace=True)
        
        # Return the predicted CSV file
        return Response(df.to_csv(index=False), media_type="text/csv")
    
    except Exception as e:
        raise Response(f"Error Occurred! {e}")


def main():
    try: 
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()

    except Exception as e:
        print(e)
        logging.exception(e)


if __name__=="__main__":
    
    # file_path="/Users/Gopal/sensorlive/aps_failure_training.csv"
    # database_name="sensordata"
    # collection_name ="sensor"
    # dump_csv_file_to_mongodb_collection(file_path,database_name,collection_name)
    
    app_run(app, host=APP_HOST, port=APP_PORT)