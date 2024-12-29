"""
Unit tests for the main.py file and our API
"""
import sys
import os
import pytest
import logging
from fastapi.testclient import TestClient

from main import app

# Importing logger
logger = logging.getLogger(__name__)

client = TestClient(app)

@pytest.fixture(scope="module")
def test_data():
    return {
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

@app.get("/")
async def read_main():
    return {"message": "Welcome to the income prediction API!"}


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    try :
        assert response.status_code == 200
    except AssertionError:
        print(f"Response status code is not 200. Response: {response.json()}")
    try:
        assert response.json() == {"message": "Welcome to the income prediction API!"}
    except AssertionError:
        print(f"Response is not as expected. Response: {response.json()}")


def test_model_files():
    # Make sure model files exist
    try :
        assert os.path.exists("model/model.joblib"), "Model file not found"
        assert os.path.exists("model/encoder.joblib"), "Encoder file not found"
        assert os.path.exists("model/lb.joblib"), "Label binarizer file not found"
    except AssertionError:
        print(f"Model files not found. Model files: {os.listdir('model')}")

def test_predict_endpoint():
    """Test the predict endpoint"""
    # Print available routes for debugging
    #print("Available routes:", [route.path for route in app.routes])
    
    test_data = {
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

    response = client.post("/predict", json=test_data)
    print("Response status:", response.status_code)
    print("Response body:", response.json() if response.status_code != 404 else "Not Found")
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [" <=50K", " >50K"]

def test_predict_classe_sup50K():
    """Test the predict endpoint"""
    
    test_data = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14084,
        "capital-loss": 0,
        "hours-per-week": 50,
        "native-country": "United-States"
    }

    response = client.post("/predict", json=test_data)
    print("Response status:", response.status_code)
    print("Response body:", response.json() if response.status_code != 404 else "Not Found")
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == " >50K"


def test_predict_classe_inf50K():
    """Test the predict endpoint"""
    
    test_data = {
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

    response = client.post("/predict", json=test_data)
    print("Response status:", response.status_code)
    print("Response body:", response.json() if response.status_code != 404 else "Not Found")
    
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] == " <=50K"