# -*- coding: utf-8 -*-

import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn
from pydantic import create_model

# Create the app
app = FastAPI()

# Load trained Pipeline
model = load_model("housing_price_api")

# Create input/output pydantic models
input_model = create_model("housing_price_api_input", **{'LotFrontageSF': 57.0, 'LotAreaSF': 8846, 'BldgType': 'Single Family Detached', 'HouseStyle': 'Split Foyer', 'OverallQual': 5, 'OverallCond': 5, 'YearBuilt': 1996, 'YearRemodAdd': 1996, 'RoofStyle': 'Gable', 'RoofMaterial': 'Standard Composite Shingle', 'ExterQual': 4, 'ExterCond': 3, 'Foundation': 'Poured Concrete', 'BsmtQual': 4, 'BsmtCond': 3, 'BsmtFinSF': 298, 'TotalBsmtSF': 870, 'Heating': 'GasA', 'HeatingQC': 5, 'CentralAir': 1, '1stFlrSF': 914, '2ndFlrSF': 0, 'GrLivArea': 914, 'GarageType': 'Detached', 'GarageYrBlt': 1998, 'GarageCars': 2, 'GarageSF': 576, 'GarageQual': 3, 'WoodDeckSF': 0, 'OpenPorchSF': 0, 'MiscFeature': 'None'})
output_model = create_model("housing_price_api_output", prediction=148000)


# Define predict function
@app.post("/predict", response_model=output_model)
def predict(data: input_model):
    data = pd.DataFrame([data.dict()])
    predictions = predict_model(model, data=data)
    return {"prediction": predictions["prediction_label"].iloc[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
