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
    input_data = pd.DataFrame( )

    #load already saved pipeline and make predictions
    with open("houseprice_best_model.pkl", "rb") as f:
        model = pickle.load(f)
        print(model)
        #predt = model.predict(input_data) 
    #return prediction 
    #return predt

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
                       description="Enter House details below and click the submit button to generate house price",
                   outputs = output)

app.launch()
#app.launch(share=True)



