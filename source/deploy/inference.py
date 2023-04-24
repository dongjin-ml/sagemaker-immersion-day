import io
import os
import csv
import time
import json
import pickle as pkl
import numpy as np
import pandas as pd
from io import BytesIO
import xgboost as xgb
import sagemaker_xgboost_container.encoder as xgb_encoders
from sagemaker.serializers import CSVSerializer
from io import StringIO

#For Gunicorn/Flask xgboost image, we need to ensure input and output encoding match exactly for model monitor (CSV or JSON)
from flask import Response 

NUM_FEATURES = 58
CSV_SERIALIZER = CSVSerializer(content_type='text/csv')

def model_fn(model_dir):
    """
    Deserialize and return fitted model.
    """
    model_file = "xgboost-model"
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, model_file))
    return model
                     

def input_fn(request_body, request_content_type):
    """
    The SageMaker XGBoost model server receives the request data body and the content type,
    and invokes the `input_fn`.
    Return a DMatrix (an object that can be passed to predict_fn).
    """

    print (f'Input, Content_type: {request_content_type}')
    if request_content_type == "application/x-npy":        
        stream = BytesIO(request_body)
        array = np.frombuffer(stream.getvalue())
        array = array.reshape(int(len(array)/NUM_FEATURES), NUM_FEATURES)
        return xgb.DMatrix(array)
    
    elif request_content_type == "text/csv":
        return xgb_encoders.csv_to_dmatrix(request_body.rstrip("\n"))
    
    elif request_content_type == "text/libsvm":
        return xgb_encoders.libsvm_to_dmatrix(request_body)
    
    else:
        raise ValueError(
            "Content type {} is not supported.".format(request_content_type)
        )

def predict_fn(input_data, model):
    """
    SageMaker XGBoost model server invokes `predict_fn` on the return value of `input_fn`.

    Return a two-dimensional NumPy array (predictions and scores)
    """
    start_time = time.time()
    y_probs = model.predict(input_data)
    print("--- Inference time: %s secs ---" % (time.time() - start_time))    
    y_preds = [1 if e >= 0.5 else 0 for e in y_probs] 
    #return np.vstack((y_preds, y_probs))
    y_probs = np.array(y_probs).reshape(1, -1)
    y_preds = np.array(y_preds).reshape(1, -1)   
    output = np.concatenate([y_probs, y_preds], axis=1)
    
    return output


def output_fn(predictions, content_type="text/csv"):
    """
    After invoking predict_fn, the model server invokes `output_fn`.
    """
    print (f'Output, Content_type: {content_type}')
    
    if content_type == "text/csv":
        outputs = CSV_SERIALIZER.serialize(predictions)
        print (outputs)
        return Response(outputs, mimetype=content_type)

    elif content_type == "application/json":

        outputs = json.dumps({
            'pred': predictions[0][0],
            'prob': predictions[0][1]
        })                
        #return outputs
        return Response(outputs, mimetype=content_type)
    else:
        raise ValueError("Content type {} is not supported.".format(content_type))
