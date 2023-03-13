#!/usr/bin/env python

import os, warnings, sys
warnings.filterwarnings('ignore') 

import pprint

sys.path.insert(0, './../../')

import app.src.preprocessing.pipeline as pipeline
import app.src.preprocessing.preprocess_utils as pp_utils
import app.src.utils as utils
import app.src.model.regressor as regressor

train_data_path = "./../data/processed_data/training/"
schema_path = "./../data/data_config/"
model_path = "./../artifacts/"


# get model configuration parameters 
model_cfg = utils.get_model_config()

def run_training():  
    
    # set random seeds
    utils.set_seeds()
    
    # read train_data
    train_data = utils.get_data(train_data_path)
    
    # get data schema
    data_schema = utils.get_data_schema(schema_path)
        
    # preprocess data
    print("Pre-processing data...")
    processed_inputs, processed_target, inputs_pipeline = preprocess_data(train_data, data_schema)    
                 
    # Create and train model     
    print('Training model ...')  
    model= train_model(train_X=processed_inputs, train_y = processed_target)         
    
    # save preprocessors
    pipeline.save_preprocessor(inputs_pipeline, model_path)
    
    # save model
    regressor.save_model(model=model, model_path=model_path)
    
    print('Done training and saving model ...')  
    


def preprocess_data(train_data, data_schema):
    # print('Preprocessing train_data of shape...', train_data.shape)
    pp_params = pp_utils.get_preprocess_params(train_data, data_schema, model_cfg)   
    
    inputs_pipeline = pipeline.get_inputs_pipeline(pp_params, model_cfg)
    inputs = train_data.loc[:, train_data.columns != pp_params["target_attr_name"]]
    processed_inputs = inputs_pipeline.fit_transform(inputs)
    
    # we are not doing any transformation on the targets, but we could have (e.g. standard scaling)
    processed_target = train_data[[pp_params["target_attr_name"]]]
    print("Processed train X/y data shape", processed_inputs.shape, processed_target.shape)
          
    return processed_inputs, processed_target, inputs_pipeline


def train_model(train_X, train_y):                 
    # get model hyper-parameters that are dependent on the data shape 
    data_based_params = regressor.get_data_based_model_params(train_X)        
    # Create and train model   
    model = regressor.Regressor(**data_based_params)  
    # model.summary()  
    model.fit( train_X=train_X, train_y=train_y )  
    
    return model






if __name__ == "__main__": 
    
    run_training()