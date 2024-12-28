"""
This script is used to test the training pipeline.
Author : Christophe Bourgoin
Date : 2024-12-28
"""

# Importing libraries
import os
import pytest
import joblib
import pandas as pd
import logging

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from training.ml.model import inference, compute_model_metrics
from training.ml.data import process_data

# Importing logger
logger = logging.getLogger(__name__)

# Fixtures
@pytest.fixture(scope="module")
def data_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'clean_data.csv')

@pytest.fixture(scope="module")
def data(data_path):
    return pd.read_csv(data_path)

@pytest.fixture(scope="module")
def model_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'model.joblib')

@pytest.fixture(scope="module")
def encoder_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'encoder.joblib')

@pytest.fixture(scope="module")
def lb_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'lb.joblib')


@pytest.fixture(scope="module")
def model(model_path):
    return joblib.load(model_path)

@pytest.fixture(scope="module")
def encoder(encoder_path):
    return joblib.load(encoder_path)

@pytest.fixture(scope="module")
def lb(lb_path):
    return joblib.load(lb_path)

@pytest.fixture(scope="module")
def features():
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    return categorical_features, numerical_features

@pytest.fixture(scope="module")
def processed_datasets(data, features):
    categorical_features = features[0]

    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['salary'])
    
    X_train, y_train, fitted_encoder, fitted_lb = process_data(train, categorical_features, label="salary", training=True)
    X_test, y_test, _, _ = process_data(test, categorical_features, label="salary", training=False, encoder=fitted_encoder, lb=fitted_lb)

    return X_train, y_train, X_test, y_test, fitted_encoder, fitted_lb



# Tests :
def test_read_data(data_path):  
    try:
        _ = pd.read_csv(data_path)
    except FileNotFoundError as e:
        logger.error("Error in data: file not found %s", e)
        raise e


def test_data(data):
    try:
        assert data.shape[0] > 0
        assert data.shape[1] > 0
    except Exception as e:
        logger.error("Error in data: incorrect shape %s", e)
        raise e

    try :
        print(type(data))
        assert isinstance(data, pd.DataFrame)
    except Exception as e:
        logger.error("Error in data: incorrect type %s", e)
        raise e
    try :
        assert 'salary' in data.columns
    except KeyError as e:
        logger.error("Error in data: no salary column in data%s", e)
        raise e
    try:
        expected_values = [' <=50K', ' >50K']
        actual_values = data['salary'].unique()
        assert set(expected_values) == set(actual_values)
    except AssertionError as e:
        logger.error("Error in data: unexpected salary values %s", e)
        raise e

def test_features(features):
    try :
        assert len(features)==2
    except AssertionError as e:
        logger.error("Error in features: Check the number of features %s", e)
        raise e
    try :
        assert len(features[0])==8
    except AssertionError as e:
        logger.error("Error in features: Check the number of categorical features %s", e)
        raise e
    try :
        assert len(features[1])==6
    except AssertionError as e:
        logger.error("Error in features: Check the number of numerical features %s", e)
        raise e

def test_model(model_path):
    try :
        loaded_model = joblib.load(model_path)
        assert isinstance(loaded_model, HistGradientBoostingClassifier)
    except AssertionError as e:
        logger.error("Error in model type: model is not a HistGradientBoostingClassifier %s", e)
        raise e



def test_encoder(encoder_path):
    try:
        loaded_encoder = joblib.load(encoder_path)
        assert loaded_encoder is not None
    except AssertionError as e:
        logger.error("Error in encoder loading: Failed to load encoder %s", e)
        raise e


def test_lb(lb_path):
    try:
        loaded_lb = joblib.load(lb_path)
        assert loaded_lb is not None
    except AssertionError as e:
        logger.error("Error in lb loading: Failed to load lb %s", e)
        raise e
    
def test_predictions(model_path, model,processed_datasets):
    loaded_model = joblib.load(model_path)
    X_test  = processed_datasets[2]
    y_test = processed_datasets[3]

    try :
        predictions = loaded_model.predict(X_test)
        assert predictions.shape == y_test.shape
    except AssertionError as e:
        logger.error("Error in predictions: Bad predictions shape %s", e)
        raise e
    try : 
        valid_classes = model.classes_
        assert all(pred in valid_classes for pred in predictions)
    except AssertionError as e:
        logger.error("Error in predictions: Bad predictions classes %s", e)
        raise e
    

def test_evaluation(model_path, processed_datasets):
    loaded_model = joblib.load(model_path)
    X_test  = processed_datasets[2]
    y_test = processed_datasets[3]

    try:
        predictions = inference(loaded_model, X_test)
        assert predictions.shape == y_test.shape
    except AssertionError as e:
        logger.error("Error in inference: Bad predictions shape  %s", e)
        raise e
    try :
        precision, recall, fbeta = compute_model_metrics(y_test, predictions)
    except Exception as e:
        logger.error("Error in evaluation: Failed to compute loaded_model metrics %s", e)
        raise e