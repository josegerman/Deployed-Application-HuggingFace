import gradio as gr
#import sklearn
import pycaret
import pandas as pd
import numpy as np
import pickle

def make_prediction(LotFrontageSF, LotAreaSF, BldgType, HouseStyle,
                   OverallQual, OverallCond, YearBuilt, YearRemodAdd,
                   RoofStyle, RoofMaterial, ExterQual, ExterCond,
                   Foundation, BsmtQual, BsmtCond, BsmtFinSF, TotalBsmtSF,
                   Heating, HeatingQC, CentralAir, FirstFlrSF, SecondFlrSF,
                   GrLivArea, GarageType, GarageYrBlt, GarageCars, GarageSF,
                   GarageQual, WoodDeckSF, OpenPorchSF, MiscFeature):
    input_data = pd.DataFrame({'LotFrontageSF': [LotFrontageSF], 'LotAreaSF': [LotAreaSF], 'BldgType': [BldgType], 'HouseStyle': [HouseStyle],
                   'OverallQual': [OverallQual], 'OverallCond': [OverallCond], 'YearBuilt': [YearBuilt], 'YearRemodAdd': [YearRemodAdd],
                   'RoofStyle': [RoofStyle], 'RoofMaterial': [RoofMaterial], 'ExterQual': [ExterQual], 'ExterCond': [ExterCond],
                   'Foundation': [Foundation], 'BsmtQual': [BsmtQual], 'BsmtCond': [BsmtCond], 'BsmtFinSF': [BsmtFinSF], 'TotalBsmtSF': [TotalBsmtSF],
                   'Heating': [Heating], 'HeatingQC': [HeatingQC], 'CentralAir': [CentralAir], 'FirstFlrSF': [FirstFlrSF], 'SecondFlrSF': [SecondFlrSF],
                   'GrLivArea': [GrLivArea], 'GarageType': [GarageType], 'GarageYrBlt': [GarageYrBlt], 'GarageCars': [GarageCars], 'GarageSF': [GarageSF],
                   'GarageQual': [GarageQual], 'WoodDeckSF': [WoodDeckSF], 'OpenPorchSF': [OpenPorchSF], 'MiscFeature': [MiscFeature]})

    #load already saved pipeline and make predictions
    with open("houseprice_best_model.pkl", "rb") as f:
        model = pickle.load(f)
        predt = model.predict(input_data) 
    #return prediction 
    return predt

#create the input components for gradio
LotFrontageSF_input = gr.Number(value = 1000)
LotAreaSF_input = gr.Number(value=70)
BldgType_input = gr.Dropdown(choices =['Single Family Detached', 'Duplex'], value='Single Family Detached')
HouseStyle_input = gr.Dropdown(choices =['1 Story', '2 Story'], value='1 Story')
OverallQual_input = gr.Number(value=8)
OverallCond_input = gr.Number(value=8)
YearBuilt_input = gr.Number(value=1989)
YearRemodAdd_input = gr.Number(value=1999)
RoofStyle_input = gr.Dropdown(choices =['Gable', 'Hip'],value='Gable')
RoofMaterial_input = gr.Dropdown(choices =['Standard Composite Shingle', value='Standard Composite Shingle'])
ExterQual_input = gr.Number(value=7)
ExterCond_input = gr.Number(value=6)
Foundation_input = gr.Dropdown(choices =['Slab'], value='Slab')
BsmtQual_input = gr.Number(value=5)
BsmtCond_input = gr.Number(value=5)
BsmtFinSF_input = gr.Number(value=500)
TotalBsmtSF_input = gr.Number(value=920)
Heating_input = gr.Dropdown(choices =['GasA'], value='GasA')
HeatingQC_input = gr.Number(value=7)
CentralAir_input = gr.Number(value=1)
FirstFlrSF_input = gr.Number(value=856)
SecondFlrSF_input = gr.Number(value=854)
GrLivArea_input = gr.Number(value=1700)
GarageType_input = gr.Dropdown(choices =['Attached', 'Detached'], value='Attached')
GarageYrBlt_input = gr.Number(value=1989)
GarageCars_input = gr.Number(value=2)
GarageSF_input = gr.Number(value=600)
GarageQual_input = gr.Number(value=8)
WoodDeckSF_input = gr.Number(value=0)
OpenPorchSF_input = gr.Number(value=30)
MiscFeature_input = gr.Dropdown(choices =['None', 'Shed'], value = 'None')
output = gr.Textbox(label='House Price') 


#create the interface component
app = gr.Interface(fn =make_prediction,inputs =[LotFrontageSF_input,
                                                LotAreaSF_input,
                                                BldgType_input,
                                                HouseStyle_input,
                                                OverallQual_input,
                                                OverallCond_input,
                                                YearBuilt_input,
                                                YearRemodAdd_input,
                                                RoofStyle_input,
                                                RoofMaterial_input,
                                                ExterQual_input,
                                                ExterCond_input,
                                                Foundation_input,
                                                BsmtQual_input,
                                                BsmtCond_input,
                                                BsmtFinSF_input,
                                                TotalBsmtSF_input,
                                                Heating_input,
                                                HeatingQC_input,
                                                CentralAir_input,
                                                FirstFlrSF_input,
                                                SecondFlrSF_input,
                                                GrLivArea_input,
                                                GarageType_input,
                                                GarageYrBlt_input,
                                                GarageCars_input,
                                                GarageSF_input,
                                                GarageQual_input,
                                                WoodDeckSF_input,
                                                OpenPorchSF_input,
                                                MiscFeature_input],
                   title ="House Price Generator",
                       description="Enter the feilds Below and click the submit button to Make Your Prediction",
                   outputs = output)


app.launch()



