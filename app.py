import gradio as gr
#import sklearn
import pandas as pd
import numpy as np

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
    with open("housing_price_api.pkl", "rb") as f:
        model = pickle.load(f)
        predt = model.predict(input_data) 
    #return prediction 
    return predt

#create the input components for gradio
LotFrontageSF_input = gr.Number()
LotAreaSF_input = gr.Number()
BldgType_input = gr.Dropdown(choices =['Single Family Detached', 'Duplex'])
HouseStyle_input = gr.Dropdown(choices =['1 Story', '2 Story'])
OverallQual_input = gr.Number()
OverallCond_input = gr.Number()
YearBuilt_input = gr.Number()
YearRemodAdd_input = gr.Number()
RoofStyle_input = gr.Dropdown(choices =['Gable', 'Hip'])
RoofMaterial_input = gr.Dropdown(choices =['Standard Composite Shingle'])
ExterQual_input = gr.Number()
ExterCond_input = gr.Number()
Foundation_input = gr.Dropdown(choices =['Slab'])
BsmtQual_input = gr.Number()
BsmtCond_input = gr.Number()
BsmtFinSF_input = gr.Number()
TotalBsmtSF_input = gr.Number()
Heating_input = gr.Dropdown(choices =['GasA'])
HeatingQC_input = gr.Number()
CentralAir_input = gr.Number()
FirstFlrSF_input = gr.Number()
SecondFlrSF_input = gr.Number()
GrLivArea_input = gr.Number()
GarageType_input = gr.Dropdown(choices =['Attached', 'Detached'])
GarageYrBlt_input = gr.Number()
GarageCars_input = gr.Number()
GarageSF_input = gr.Number()
GarageQual_input = gr.Number()
WoodDeckSF_input = gr.Number()
OpenPorchSF_input = gr.Number()
MiscFeature_input = gr.Dropdown(choices =['None', 'Shed'])
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



