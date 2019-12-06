import pandas as pd
from flask import Flask, jsonify, request
import json
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import keras

# load model
model = keras.models.load_model('new_data.h5')
model._make_predict_function() 
# app
app = Flask(__name__)

# routes
@app.route("/")
def greet():
    return "Hi, I am a Bayesian NN model!"

@app.route('/req/', methods=['POST'])

def predict():
    # get data
    
    data = request.get_json(force=True)
    print(data)
    # convert data into dataframe
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    
    mat =[]
    for i in range(0,200):
        mat.append(model.predict(data_df))
    a = np.mean(mat,axis=0)
    b = np.std(mat,axis=0)
    
#    # send back to browser
    y_keys = ['Lvr Area','Kit area','WC area','Br area','Br para','Mbr Area',
          'Storage Area','Public circulation ratio',
          'Material Infill 1_Vol','Material Infill 2_Vol','Material Support 1_Vol','Material Support 2_Vol','Material Support Slab_Vol']

    output={}
    for i in range(0,len(y_keys)):
        output[y_keys[i]] = list(a[:,i])
    
    output = pd.DataFrame.from_dict(a)

    output = output.to_json(orient='records')

    # return data
    return output

if __name__ == '__main__':
#    app.run(port = 5000, debug=True)
    app.run()