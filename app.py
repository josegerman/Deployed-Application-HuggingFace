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
    with open("house_price_api.pkl", "rb") as f:
        model = pickle.load(f)
        predt = model.predict(input_data) 
    #return prediction 
    return predt

#create the input components for gradio
LotFrontageSF_input = gr.number()
LotAreaSF_input = gr.number()
BldgType_input = gr.inputs.Drowdown(choices =['Single Family Detached', 'Duplex'])
HouseStyle_input = gr.inputs.Drowdown(choices =['1 Story', '2 Story'])
OverallQual_input = gr.number()
YearBuilt_input = gr.number()
YearRemodAdd_input = gr.number()
RoofStyle_input = gr.inputs.Drowdown(choices =['Gable', 'Hip'])
RoofMaterial_input = gr.inputs.Drowdown(choices = ['Standard Composite Shingle'])
ExterQual_input = gr.number()
ExterCond_input = gr.number()
Foundation_input = gr.inputs.Drowdown(choices = ['Slab'])
BsmtQual_input = gr.number()
BsmtCond_input = gr.number()
BsmtFinSF_input = gr.number()
TotalBsmtSF_input = gr.number()
Heating_input = gr.inputs.Drowdown(choices = ['GasA'])
HeatingQC_input = gr.number()
CentralAir_input = gr.number()
FirstFlrSF_input = gr.number()
SecondFlrSF_input = gr.number()
GrLivArea_input = gr.number()
GarageType_input = gr.number()
GarageYrBlt_input = gr.number()
GarageCars_input = gr.number()
GarageSF_input = gr.number()
GarageQual_input = gr.number()
WoodDeckSF_input = gr.number()
OpenPorchSF_input = gr.number()
MiscFeature_input = gr.inputs.Drowdown(choices = ['None', 'Shed'])

output = gr.Textbox(label='House Price') 


#create the interface component
app = gr.Interface(fn =make_prediction,inputs =[LotFrontageSF_input,
                                                LotAreaSF_input,
                                                BldgType_input,
                                                HouseStyle_input,
                                                OverallQual_input,
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


app.launch(share = True)



