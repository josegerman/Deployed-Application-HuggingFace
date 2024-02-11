import gradio as gr
import sklearn
import pandas pd
import numpy as np

def make_prediction(gender, Partner, Dependents, tenure, MultipleLines,
       InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, PaperlessBilling, PaymentMethod,
       MonthlyCharges, TotalCharges):
   input_data = pd.DataFrame({'gender':[gender], 'Partner':[Partner], 'Dependents':[Dependents], 'tenure':[tenure], 'MultipleLines':[MultipleLines],
       'InternetService':[InternetService], 'OnlineSecurity':[OnlineSecurity], 'OnlineBackup':[OnlineBackup], 'DeviceProtection':[DeviceProtection],
       'TechSupport':[TechSupport], 'Contract':[Contract], 'PaperlessBilling':[PaperlessBilling], 'PaymentMethod':[PaymentMethod],
       'MonthlyCharges':[MonthlyCharges], 'TotalCharges':[TotalCharges]})
   
   #load already saved pipeline and make predictions
    with open("ml_model.pkl", "rb") as f:
        model = pickle.load(f)
        predt = model.predict(input_data) 
    #return prediction 
    return predt