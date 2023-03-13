
import uvicorn
from fastapi import FastAPI
import sys
import pandas as pd
import numpy as np

from data_model import DataModel


sys.path.insert(0, './../../')

import app.src.preprocessing.pipeline as pipeline
import app.src.model.regressor as regressor


artifacts_path = "./../artifacts/"

# Create app 
app = FastAPI()
# load preprocessors
inputs_pipeline = pipeline.load_preprocessor(artifacts_path)
# load model 
model = regressor.load_model(artifacts_path)


@app.get("/ping")
def ping() -> dict:
    '''
    This endpoint is used for health check. Returns a message indicating health of service. 
    '''
    return { "message": "Random Forest prediction service is running!" }


# Expose the prediction functionality, make a prediction from the passed
# JSON data and return the predicted diamond value
@app.post('/predict')
def predict(input_: DataModel) -> dict:
    '''
    Returns predicted price of a diamond given input features. \n
    Input is a json object with following keys and corresponding value types: \n 
    * Id: str field used to represent the unique id of the record. It is ignored by the prediction model.
    * Carat Weight: float in range 0.75 to 3.0
    * Cut: Optional str with categorical values: [ "Fair", "Good", "Ideal", "Signature-Ideal", "Very Good", ]
    * Color: Optional str with categorical values: [ "D", "E", "F", "G", "H", "I"]
    * Clarity: Optional str with categorical values: [ 'FL', 'IF', 'SI1', 'VS1', 'VS2', 'VVS1', 'VVS2', ]
    * Polish: Optional str with categorical values: [ 'EX', 'G', 'ID', 'VG',  ]
    * Symmetry: Optional str with categorical values: [ 'EX', 'G', 'ID', 'VG',  ]
    * Report: Optional str with categorical values: [ 'AGSL', 'GIA', ]
    '''
        
    df = pd.DataFrame.from_records([input_.dict(by_alias=True)])
    # print(df)    
    
    processed_data =  inputs_pipeline.transform(df)    
    prediction = model.predict(processed_data)[0]    
    return {
        "data": {**input_.dict(by_alias=True)}, 
        "prediction": np.round(prediction, 4)
    }


if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=80)             