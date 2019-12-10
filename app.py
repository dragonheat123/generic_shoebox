import pandas as pd
from flask import Flask, jsonify, request
import json
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import keras

# load models and information
model = keras.models.load_model('model.h5')
model._make_predict_function() 

modelc = keras.models.load_model('modelc.h5')
modelc._make_predict_function() 

y_max = {'Br area': 18.15,
 'Br para': 2.891673403,
 'Kit area': 31.5,
 'Lvr Area': 39.278,
 'Material Infill 1_Vol': 323.4,
 'Material Infill 2_Vol': 420.66,
 'Material Support 1_Vol': 1094.4,
 'Material Support 2_Vol': 150.0,
 'Material Support Slab_Vol': 1131.5,
 'Mbr Area': 19.695,
 'Public circulation ratio': 0.355973626,
 'Storage Area': 3.7,
 'Support_ratio': 0.557413601,
 'WC area': 6.8}

x_max = {'Building Floor Area': 2828.8,
 'Building para': 2.397474391,
 'Construction System_Column Beam_Concrete_Glulam_Slab_Concrete': 1.0,
 'Construction System_Column Beam_Concrete_Glulam_Slab_Glulam': 1.0,
 'Construction System_Column Beam_Concrete_None_Slab_Concrete': 1.0,
 'Construction System_Column Beam_Concrete_Steel_Slab_Concrete': 1.0,
 'Construction System_Shear Wall_CLT_None_Slab_CLT': 1.0,
 'Construction System_Shear Wall_Concrete_CLT_Slab_Concrete': 1.0,
 'Construction System_Shear Wall_Concrete_Glulam_Slab_Glulam': 1.0,
 'Construction System_Shear Wall_Concrete_None_Slab_Concrete': 1.0,
 'Material Infill 1_Gypsum ': 1.0,
 'Material Infill 1_Light Weight Concrete': 1.0,
 'Material Infill 2 (Windows)_Double Pane IGU': 1.0,
 'No of Br': 4.0,
 'No of Mbr': 2.0,
 'No of Storage': 3.0,
 'No of WC': 3.0,
 'No. of units in floor': 38.0}

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
    x_input = pd.DataFrame.from_dict(data)
    
    
    x_input={'Building Floor Area': 1027,
              'Building para': 0.64,
              'No of Br': 2,
              'No of Mbr': 0,
              'No of Storage': 1,
              'No of WC': 2,
              'No. of units in floor': 12,
              'Material Infill 1_Gypsum ': 1.0,
              'Material Infill 1_Light Weight Concrete': 0.0,
              'Material Infill 2 (Windows)_Double Pane IGU': 1.0,
              'Construction System_Column Beam_Concrete_Glulam_Slab_Concrete': 1,
              'Construction System_Column Beam_Concrete_Glulam_Slab_Glulam': 0,
              'Construction System_Column Beam_Concrete_None_Slab_Concrete': 0,
              'Construction System_Column Beam_Concrete_Steel_Slab_Concrete': 0,
              'Construction System_Shear Wall_CLT_None_Slab_CLT':0,
              'Construction System_Shear Wall_Concrete_CLT_Slab_Concrete': 0,
              'Construction System_Shear Wall_Concrete_Glulam_Slab_Glulam': 0,
              'Construction System_Shear Wall_Concrete_None_Slab_Concrete': 0}

    x_area = pd.DataFrame.from_dict([x_input]).ix[:,['Building Floor Area', 'Building para', 'No of WC', 'No of Br',\
                          'No of Mbr', 'No of Storage', 'No. of units in floor',\
                          'Material Infill 1_Gypsum ','Material Infill 1_Light Weight Concrete',\
                          'Material Infill 2 (Windows)_Double Pane IGU']]
    x_vol = pd.DataFrame.from_dict([x_input]).ix[:,['Building Floor Area', 'Building para', 'No of WC', 'No of Br',\
                              'No of Mbr', 'No of Storage', 'No. of units in floor',\
                              'Construction System_Column Beam_Concrete_Glulam_Slab_Concrete',\
                               'Construction System_Column Beam_Concrete_Glulam_Slab_Glulam',\
                               'Construction System_Column Beam_Concrete_None_Slab_Concrete',\
                               'Construction System_Column Beam_Concrete_Steel_Slab_Concrete',\
                               'Construction System_Shear Wall_CLT_None_Slab_CLT',\
                               'Construction System_Shear Wall_Concrete_CLT_Slab_Concrete',\
                               'Construction System_Shear Wall_Concrete_Glulam_Slab_Glulam',\
                               'Construction System_Shear Wall_Concrete_None_Slab_Concrete']]
    y_max_area = pd.DataFrame.from_dict([y_max]).ix[:,['Lvr Area', 'Kit area', 'WC area', 'Br area', 'Br para', 'Mbr Area',
       'Storage Area', 'Public circulation ratio', 'Material Infill 1_Vol',
       'Material Infill 2_Vol']]
    x_max_area = pd.DataFrame.from_dict([x_max]).ix[:,['Building Floor Area', 'Building para', 'No of WC', 'No of Br',\
                              'No of Mbr', 'No of Storage', 'No. of units in floor',\
                              'Material Infill 1_Gypsum ','Material Infill 1_Light Weight Concrete',\
                              'Material Infill 2 (Windows)_Double Pane IGU']]
    y_max_vol = pd.DataFrame.from_dict([y_max]).ix[:,['Material Support 1_Vol', 'Support_ratio', 'Material Support Slab_Vol']]
    x_max_vol = pd.DataFrame.from_dict([x_max]).ix[:,['Building Floor Area', 'Building para', 'No of WC', 'No of Br',\
                              'No of Mbr', 'No of Storage', 'No. of units in floor',\
                              'Construction System_Column Beam_Concrete_Glulam_Slab_Concrete',\
                               'Construction System_Column Beam_Concrete_Glulam_Slab_Glulam',\
                               'Construction System_Column Beam_Concrete_None_Slab_Concrete',\
                               'Construction System_Column Beam_Concrete_Steel_Slab_Concrete',\
                               'Construction System_Shear Wall_CLT_None_Slab_CLT',\
                               'Construction System_Shear Wall_Concrete_CLT_Slab_Concrete',\
                               'Construction System_Shear Wall_Concrete_Glulam_Slab_Glulam',\
                               'Construction System_Shear Wall_Concrete_None_Slab_Concrete']]

    x_area = x_area/x_max_area.values
    x_vol = x_vol/x_max_vol.values
    
    # predictions
    
    ckey = ['Construction System_Column Beam_Concrete_Glulam_Slab_Concrete',
             'Construction System_Column Beam_Concrete_Glulam_Slab_Glulam',
             'Construction System_Column Beam_Concrete_None_Slab_Concrete',
             'Construction System_Column Beam_Concrete_Steel_Slab_Concrete',
             'Construction System_Shear Wall_CLT_None_Slab_CLT',
             'Construction System_Shear Wall_Concrete_CLT_Slab_Concrete',
             'Construction System_Shear Wall_Concrete_Glulam_Slab_Glulam',
             'Construction System_Shear Wall_Concrete_None_Slab_Concrete']
    change = [0,0,0,0,0,0,0,0]
    
    y_area_mat=[]
    y_area = {}
    for i in range(0,400):
        y_area_mat.append(model.predict(x_area)*y_max_area.values)
    y_area_a = np.mean(y_area_mat,axis=0)
    y_area_b = np.std(y_area_mat,axis=0)
    y_area['y_area_a']=dict(zip(y_max_area.keys().values,y_area_a[0]))
    y_area['y_area_u']=dict(zip(y_max_area.keys().values,y_area_b[0]))
    
    y_vol_holder = []
      
    for j in range(0,len(ckey)):
        change = [0,0,0,0,0,0,0,0]
        change[j] = 1
        x_vol[ckey] = change
        y_vol_mat=[]
        y_vol = {}  
        for i in range(0,400):
            y_vol_mat.append(modelc.predict(x_vol)*y_max_vol.values)
        y_vol_a = np.mean(y_vol_mat,axis=0)
        y_vol_b = np.std(y_vol_mat,axis=0)
        y_vol['y_vol_a'] = dict(zip(y_max_vol.keys().values,y_vol_a[0]))
        y_vol['y_vol_u'] = dict(zip(y_max_vol.keys().values,y_vol_b[0]))
        y_vol_holder.append(y_vol)
    
    output = json.dumps([y_area,y_vol_holder])
    

    # return data
    return output

if __name__ == '__main__':
#    app.run(port = 5000, debug=True)
    app.run()