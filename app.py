import gradio as gr
import sklearn
import pandas pd
import numpy as np

def make_prediction(LotFrontageSF, LotAreaSF, BldgType, HouseStyle,
                   OverallQual, OverallCond, YearBuilt, YearRemodAdd,
                   RoofStyle, RoofMaterial, ExterQual, ExterCond,
                   Foundation, BsmtQual, BsmtCond, BsmtFinSF, TotalBsmtSF,
                   Heating, HeatingQC, CentralAir, 1stFlrSF, 2ndFlrSF,
                   GrLivArea, GarageType, GarageYrBlt, GarageCars, GarageSF,
                   GarageQual, WoodDeckSF, OpenPorchSF, MiscFeature):
    input_data = pd.DataFrame({'LotFrontageSF': [LotFrontageSF], 'LotAreaSF': [LotAreaSF], 'BldgType': [BldgType], 'HouseStyle': [HouseStyle],
                               'OverallQual': [OverallQual], 'OverallCond': [OverallCond], 'YearBuilt': [YearBuilt], 'YearRemodAdd': [YearRemodAdd],
                               'RoofStyle': [RoofStyle], 'RoofMaterial': [RoofMaterial], 'ExterQual': [ExterQual], 'ExterCond': [ExterCond],
                               'Foundation': [Foundation], 'BsmtQual': [BsmtQual], 'BsmtCond': [BsmtCond], 'BsmtFinSF': [BsmtFinSF], 'TotalBsmtSF': [TotalBsmtSF],
                               'Heating': [Heating], 'HeatingQC': [HeatingQC], 'CentralAir': [CentralAir], '1stFlrSF': [1stFlrS], '2ndFlrSF': [2ndFlrSF],
                               'GrLivArea': [GrLivArea], 'GarageType': [GarageType], 'GarageYrBlt': [GarageYrBlt], 'GarageCars': [GarageCars], 'GarageSF': [GarageSF],
                               'GarageQual': [GarageQual], 'WoodDeckSF': [WoodDeckSF], 'OpenPorchSF': [OpenPorchSF], 'MiscFeature': [MiscFeature]})

           
   
#load already saved pipeline and make predictions
with open("house_price_api.pkl", "rb") as f:
    model = pickle.load(f)
    predt = model.predict(input_data) 
#return prediction 
return predt

#create the input components for gradio
LotFrontageSF_input = gr.number()
LotAreaSF_input = gr.number()
BldgType_input = gr.inputs.Drowdown(choices =['Single Family Detached', 'Duplex'])
HouseStyle_input = gr.inputs.Drowdown(choices =['1 Story', '2 Story']



output = gr.Textbox(label='Prediction') 