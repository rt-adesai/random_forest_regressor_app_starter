

import os, warnings, sys
import json
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import linregress


sys.path.insert(0, './../../')


import app.src.preprocessing.pipeline as pipeline
import app.src.model.regressor as regressor
import app.src.utils as utils


test_data_path = "./../data/processed_data/testing/"
schema_path = "./../data/data_config/"
artifacts_path = "./../artifacts/"
results_path = "./../results/"

results_fname = "results.json"


def run_evaluation(): 
        
    # read train_data
    test_data = utils.get_data(test_data_path)
    # print(test_data.shape)
    
    # get data schema
    data_schema = utils.get_data_schema(schema_path)

    # load preprocessors
    inputs_pipeline = pipeline.load_preprocessor(artifacts_path)
    
    # preprocess test inputs
    processed_test_inputs = inputs_pipeline.transform(test_data)
    
    # load model 
    model = regressor.load_model(artifacts_path)
    
    # make predictions
    predictions = model.predict(processed_test_inputs)    
    
    scores = get_scores(test_data, predictions, data_schema)
    print(scores)
    with open(os.path.join(results_path, results_fname), "w") as outfile:
        json.dump(scores, outfile, indent=2)
    
    test_data["predictions"] = predictions
    test_data.to_csv(f"{results_path}predictions.csv", index=False)



def get_scores(test_data, predictions, data_schema):     
    # actuals
    Y = test_data[data_schema["inputDatasets"]["regressionBaseMainInput"]["targetField"]]
    # predictions
    Y_hat = np.squeeze(predictions)
    rmse = mean_squared_error(Y, Y_hat, squared=False)
    mae = mean_absolute_error(Y, Y_hat)
    q3, q1 = np.percentile(Y, [75, 25])
    iqr = q3 - q1
    nmae = mae / iqr
    _, _, r_value, _, _  = linregress(Y, Y_hat)
    r2 = r_value * r_value
    scores = {
        "rmse": np.round(rmse,4), 
        "mae": np.round(mae,4),
        "nmae": np.round(nmae,4),
        "r2": np.round(r2,4)
        }
    return scores



if __name__ == "__main__": 
    
    run_evaluation()