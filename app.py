# Import required libraries

import pandas as pd
import numpy as np
import pickle
import gradio as gr
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

st.title("Predict House Price")

# Load saved model using pickle
with open('randomforestregressor_model.pkl', 'rb') as file1:
    model1 = pickle.load(file1)

# Load saved encoder objects using pickle
with open('BldgType_le.pkl','rb') as f1:
    BldgType_le = pickle.load(f1)

with open('HouseStyle_le.pkl','rb') as f2:
    HouseStyle_le = pickle.load(f2)

with open('RoofStyle_le.pkl','rb') as f3:
    RoofStyle_le = pickle.load(f3)

with open('RoofMaterial_le.pkl','rb') as f4:
    RoofMaterial_le = pickle.load(f4)

with open('Foundation_le.pkl','rb') as f5:
    Foundation_le = pickle.load(f5)

with open('Heating_le.pkl','rb') as f6:
    Heating_le = pickle.load(f6)

with open('MiscFeature_le.pkl','rb') as f7:
    MiscFeature_le = pickle.load(f7)

with open('CentralAir_le.pkl','rb') as f8:
    CentralAir_le = pickle.load(f8)

with open('GarageType_le.pkl','rb') as f9:
    GarageType_le = pickle.load(f9)

# Define prediction function
def predict_house_price(LotFrontageSF,LotAreaSF,OverallQual,OverallCond,YearBuilt,
                        YearRemodAdd,ExterQual,ExterCond,BsmtQual,BsmtCond,
                        BsmtFinSF,TotalBsmtSF,HeatingQC,FirstFlrSF,SecondFlrSF,
                        GrLivArea,GarageYrBlt,GarageCars,GarageSF,GarageQual,
                        WoodDeckSF,OpenPorchSF,BldgType,HouseStyle,RoofStyle,
                        RoofMaterial,Foundation,Heating,MiscFeature,CentralAir,
                        GarageType):
    
    BldgType_encoded = BldgType_le.transform([BldgType])[0]
    HouseStyle_encoded = HouseStyle_le.transform([HouseStyle])[0]
    RoofStyle_encoded = RoofStyle_le.transform([RoofStyle])[0]
    RoofMaterial_encoded = RoofMaterial_le.transform([RoofMaterial])[0]
    Foundation_encoded = Foundation_le.transform([Foundation])[0]
    Heating_encoded = Heating_le.transform([Heating])[0]
    MiscFeature_encoded = MiscFeature_le.transform([MiscFeature])[0]
    CentralAir_encoded = CentralAir_le.transform([CentralAir])[0]
    GarageType_encoded = GarageType_le.transform([GarageType])[0]
    
    

    input_data = pd.DataFrame({'LotFrontageSF': [LotFrontageSF], 'LotAreaSF': [LotAreaSF], 'OverallQual': [OverallQual], 'OverallCond': [OverallCond], 'YearBuilt': [YearBuilt],
                               'YearRemodAdd': [YearRemodAdd], 'ExterQual': [ExterQual], 'ExterCond': [ExterCond], 'BsmtQual': [BsmtQual], 'BsmtCond': [BsmtCond],
                               'BsmtFinSF': [BsmtFinSF], 'TotalBsmtSF': [TotalBsmtSF], 'HeatingQC': [HeatingQC],  'FirstFlrSF': [FirstFlrSF], 'SecondFlrSF': [SecondFlrSF],
                               'GrLivArea': [GrLivArea],  'GarageYrBlt': [GarageYrBlt], 'GarageCars': [GarageCars], 'GarageSF': [GarageSF], 'GarageQual': [GarageQual],
                               'WoodDeckSF': [WoodDeckSF], 'OpenPorchSF': [OpenPorchSF], 'BldgType_Encoded': [BldgType_encoded], 'HouseStyle_Encoded': [HouseStyle_encoded], 'RoofStyle_Encoded': [RoofStyle_encoded],
                               'RoofMaterial_Encoded': [RoofMaterial_encoded], 'Foundation_Encoded': [Foundation_encoded], 'Heating_Encoded': [Heating_encoded], 'MiscFeature_Encoded': [MiscFeature_encoded], 'CentralAir_Encoded': [CentralAir_encoded],
                               'GarageType_Encoded': [GarageType_encoded]})
    
    predict_house_price = model1.predict(input_data)

    return predict_house_price[0]

#create the input components for gradio
LotFrontageSF_input = gr.Number(label='Lot Frontage (Squared Feet)',value = 1000)
LotAreaSF_input = gr.Number(label='Lot Total Area (Squared Feet)',value=70)
BldgType_input = gr.Dropdown(label='Build Type',choices =['Single Family Detached', 'Duplex'], value='Single Family Detached')
HouseStyle_input = gr.Dropdown(label='House Style',choices =['1 Story', '2 Story'], value='1 Story')
OverallQual_input = gr.Number(label='Overall House Quality (1 to 10 scale; 10=best)',value=8)
OverallCond_input = gr.Number(label='Overall House Condition (1 to 10 scale; 10=best)',value=8)
YearBuilt_input = gr.Number(label='Year Build',value=1989)
YearRemodAdd_input = gr.Number(label='Year Remodelled (if any)',value=1999)
RoofStyle_input = gr.Dropdown(label='Roof Style',choices =['Gable', 'Hip'],value='Gable')
RoofMaterial_input = gr.Dropdown(label='Roof Material',choices =['Standard Composite Shingle'], value='Standard Composite Shingle')
ExterQual_input = gr.Number(label='Exterior Quality (1 to 10 scale; 10=best)',value=7)
ExterCond_input = gr.Number(label='Exterior Condition (1 to 10 scale; 10=best)',value=6)
Foundation_input = gr.Dropdown(label='Foundation Type',choices =['Slab'], value='Slab')
BsmtQual_input = gr.Number(label='Basement Quality (1 to 10 scale; 10=best)',value=5)
BsmtCond_input = gr.Number(label='Basement Condition (1 to 10 scale; 10=best)',value=5)
BsmtFinSF_input = gr.Number(label='Basement Finished Area (Squared Feed)',value=500)
TotalBsmtSF_input = gr.Number(label='Total Basement Area (Squared Feet)',value=920)
Heating_input = gr.Dropdown(label='Heating Type',choices =['GasA'], value='GasA')
HeatingQC_input = gr.Number(label='Heating Quality (1 to 10 scale; 10=best)',value=7)
CentralAir_input = gr.Dropdown(label='Central Air Conditioning',choices =['Yes','No'], value='No')
FirstFlrSF_input = gr.Number(label='First Floor Area (Squared Feet)',value=856)
SecondFlrSF_input = gr.Number(label='Second Floor Area (Square Feet)(If none, enter 0)',value=854)
GrLivArea_input = gr.Number(label='Great Living Area (Squared Feet)',value=1700)
GarageType_input = gr.Dropdown(label='Garage Type',choices =['Attached', 'Detached'], value='Attached')
GarageYrBlt_input = gr.Number(label='Year Garage Built',value=1989)
GarageCars_input = gr.Number(label='Garage Size (Number of Cars)',value=2)
GarageSF_input = gr.Number(label='Garage Area (Squared Feet)',value=600)
GarageQual_input = gr.Number(label='Garage Quality (1 to 10 scale; 10=best)',value=8)
WoodDeckSF_input = gr.Number(label='Wood Deck Area (Squared Feet)',value=0)
OpenPorchSF_input = gr.Number(label='Open Porch Area (Squared Feet)',value=30)
MiscFeature_input = gr.Dropdown(label='Miscellaneous Feature',choices =['None', 'Shed'], value = 'None')
output = gr.Textbox(label='Predicted House Price') 


#create the interface component
app = gr.Interface(fn = predict_house_price,inputs = [LotFrontageSF_input,
                                                LotAreaSF_input,
                                                OverallQual_input,
                                                OverallCond_input,
                                                YearBuilt_input,
                                                YearRemodAdd_input,
                                                ExterQual_input,
                                                ExterCond_input,
                                                BsmtQual_input,
                                                BsmtCond_input,
                                                BsmtFinSF_input,
                                                TotalBsmtSF_input,
                                                HeatingQC_input,
                                                FirstFlrSF_input,
                                                SecondFlrSF_input,
                                                GrLivArea_input,
                                                GarageYrBlt_input,
                                                GarageCars_input,
                                                GarageSF_input,
                                                GarageQual_input,
                                                WoodDeckSF_input,
                                                OpenPorchSF_input,
                                                BldgType_input,
                                                HouseStyle_input,
                                                RoofStyle_input,
                                                RoofMaterial_input,
                                                Foundation_input,
                                                Heating_input,
                                                MiscFeature_input,
                                                CentralAir_input,
                                                GarageType_input
                                                ],
                   title ="House Price Generator",
                       description="Enter House details below and click the submit button to generate house price",
                   outputs = output)

app.launch()
#app.launch(share=True)



