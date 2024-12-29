from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, List
import pandas as pd
import joblib
from training.ml.data import process_data
from training.ml.model import inference

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and encoders
model = joblib.load("model/model.joblib")
encoder = joblib.load("model/encoder.joblib")
lb = joblib.load("model/lb.joblib")

class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }

@app.get("/")
async def root():
    return {"message": "Welcome to the income prediction API!"}

@app.get("/docs")  # Explicitly add docs route if needed
async def get_docs():
    return app.openapi()

@app.post("/predict")  # Make sure it's exactly "/predict"
async def predict(data: InputData):
    """Make predictions with the model"""
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict(by_alias=True)])
    
    # Get the features used during training
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the input data
    X, _, _, _ = process_data(
        input_df, 
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Get prediction
    prediction = inference(model, X)
    pred_label = lb.inverse_transform(prediction)[0]

    return {
        "prediction": pred_label
    }
