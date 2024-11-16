import pytest
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.train_functions import train_and_evaluate_model, train_final_model

@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_classes=2,
        random_state=42
    )
    X = pd.DataFrame(X)
    y = pd.Series(y)
    return X, y

@pytest.fixture
def invalid_data():
    X = pd.DataFrame()
    y = pd.Series()
    return X, y

def test_train_and_evaluate_model(sample_data):
    X, y = sample_data
    test_size = 0.2
    eta0 = 1.0
    random_seed_array = [42, 43, 44]

    accuracy_array, precision_array = train_and_evaluate_model(
        X, y, test_size, eta0, random_seed_array
    )

    assert len(accuracy_array) == len(random_seed_array), "Accuracy array length mismatch."
    assert len(precision_array) == len(random_seed_array), "Precision array length mismatch."
    for acc in accuracy_array:
        assert 0 <= acc <= 1, "Accuracy out of bounds."
    for prec in precision_array:
        assert 0 <= prec <= 1, "Precision out of bounds."

def test_train_and_evaluate_model_invalid_test_size(sample_data):
    X, y = sample_data
    test_size = -0.1
    eta0 = 1.0
    random_seed_array = [42]

    accuracy_array, precision_array = train_and_evaluate_model(
        X, y, test_size, eta0, random_seed_array
    )

    assert len(accuracy_array) == 0, "Accuracy array should be empty due to invalid test_size."
    assert len(precision_array) == 0, "Precision array should be empty due to invalid test_size."

def test_train_and_evaluate_model_empty_random_seed_array(sample_data):
    X, y = sample_data
    test_size = 0.2
    eta0 = 1.0
    random_seed_array = []

    accuracy_array, precision_array = train_and_evaluate_model(
        X, y, test_size, eta0, random_seed_array
    )

    assert len(accuracy_array) == 0, "Accuracy array should be empty when random_seed_array is empty."
    assert len(precision_array) == 0, "Precision array should be empty when random_seed_array is empty."

def test_train_final_model(sample_data):
    X, y = sample_data
    eta0 = 1.0

    model, scaler = train_final_model(X, y, eta0)

    assert isinstance(model, Perceptron), "Model should be an instance of Perceptron."
    assert isinstance(scaler, StandardScaler), "Scaler should be an instance of StandardScaler."
    assert hasattr(model, "coef_"), "Model should be trained and have coefficients."

def test_train_final_model_invalid_eta0(sample_data):
    X, y = sample_data
    eta0 = -1.0

    with pytest.raises(ValueError):
        train_final_model(X, y, eta0)

def test_train_final_model_empty_data(invalid_data):
    X, y = invalid_data
    eta0 = 1.0

    with pytest.raises(ValueError):
        train_final_model(X, y, eta0)

def test_train_and_evaluate_model_exception_handling():
    X = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    y = pd.Series([0, 1])

    test_size = 0.5
    eta0 = 1.0
    random_seed_array = ['invalid_seed']

    accuracy_array, precision_array = train_and_evaluate_model(
        X, y, test_size, eta0, random_seed_array
    )

    assert len(accuracy_array) == 0, "Accuracy array should be empty due to exception."
    assert len(precision_array) == 0, "Precision array should be empty due to exception."
