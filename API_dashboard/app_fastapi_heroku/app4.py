# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:54:18 2023

@author: kbouh
"""

# 1. Library imports
import uvicorn
#import plotly
from fastapi import FastAPI
import pickle
import gzip
from pathlib import Path
#import json
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go

#from sklearn.model_selection import train_test_split

#from lime import lime_tabular

#import gzip
import pandas as pd
#import joblib
#from joblib import load


from pydantic import BaseModel



def application(environ, start_response):
  if environ['REQUEST_METHOD'] == 'OPTIONS':
    start_response(
      '200 OK',
      [
        ('Content-Type', 'application/json'),
        ('Access-Control-Allow-Origin', '*'),
        ('Access-Control-Allow-Headers', 'Authorization, Content-Type'),
        ('Access-Control-Allow-Methods', 'POST'),
      ]
    )
    return ''


origin_data = pd.read_csv('data_reduced3.csv')
data = origin_data.drop(['Unnamed: 0','TARGET'], axis=1)
liste_id = data['SK_ID_CURR'].tolist()
#data1 = data.drop("SK_ID_CURR", axis=1)
#data1 = data.drop('SK_ID_CURR',axis=1)


class LoanID(BaseModel):
    SK_ID_CURR:int

        
app = FastAPI()
with gzip.open(Path(__file__).resolve().parent / "RFClassifier3.pkl.gz", "rb") as f:
    classifier = pickle.load(f)
#classifier = joblib.load(open('RFClassifier3.joblib', 'rb'))
#classifier = pickle.loads(gzip.open("RFClassifier3.pkl","rb").read())
#classifier = pickle.loads(gzip.open("RFClassifier3.pkl.gz","rb").read())

@app.get('/')
def index():
    return {'message': 'Bienvenue !'}
    
@app.post('/predict')
def predict_target(input : LoanID):
    
    inp1 = input.SK_ID_CURR
     
    if (inp1 in liste_id):
        
        
        x = data.iloc[(data[data['SK_ID_CURR']==inp1]).index[0]]
        inp2 = x[1]
        inp3 = x[2]
        inp4 = x[3]
        inp5 = x[4]
        inp6 = x[5]
        inp7 = x[6]
        inp8 = x[7]
        result = classifier.predict_proba([[inp2,inp3,inp4,inp5,inp6,inp7,inp8]])[0]
        
          
        return {
            result[1]
        }
        
             
    else:
        return {'Client pas identifié'}
  
        
    if __name__ == '__main__':
        uvicorn.run(app, host='0.0.0.0', port=5000)