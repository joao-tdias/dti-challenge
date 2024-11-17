import pytest
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.prediction import predict_new_data

@pytest.fixture
def scaler():
    data = np.random.rand(100, 10)
    scaler = StandardScaler()
    scaler.fit(data)
    return scaler

@pytest.fixture
def model():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model = LogisticRegression()
    model.fit(X, y)
    return model

def test_predict_new_data_valid_input(model, scaler):
    data = [
        [1000025, 5, 1, 1, 1, 2, 1, 3, 1, 1]
    ]
    pred = predict_new_data(model, scaler, data)
    assert pred is not None
    assert len(pred) == 1

def test_predict_new_data_missing_columns(model, scaler):
    data = [
        [1000025, 5, 1, 1, 1, 2, 1, 3, 1]
    ]
    pred = predict_new_data(model, scaler, data)
    assert pred is None

def test_predict_new_data_invalid_scaler(model):
    data = [
        [1000025, 5, 1, 1, 1, 2, 1, 3, 1, 1]
    ]
    scaler = None
    pred = predict_new_data(model, scaler, data)
    assert pred is None

def test_predict_new_data_invalid_model(scaler):
    data = [
        [1000025, 5, 1, 1, 1, 2, 1, 3, 1, 1]
    ]
    model = None
    pred = predict_new_data(model, scaler, data)
    assert pred is None

def test_predict_new_data_scaler_exception(model, scaler):
    data = [
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    ]
    pred = predict_new_data(model, scaler, data)
    assert pred is None

def test_predict_new_data_model_exception(scaler):
    data = [
        [1000025, 5, 1, 1, 1, 2, 1, 3, 1, 1]
    ]

    class FaultyModel:
        def predict(self, X):
            raise ValueError("Model prediction failed")

    model = FaultyModel()
    pred = predict_new_data(model, scaler, data)
    assert pred is None
