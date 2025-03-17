import pytest
import pickle
import numpy as np
from app import app  # Import Flask app from app.py

# Load the trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@pytest.fixture
def client():
    """Creates a test client for the Flask app."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Test if the home route ("/") loads successfully."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Iris Flower Classifier" in response.data


def test_valid_prediction(client):
    """Test model prediction with valid inputs."""
    valid_data = {
        "sepal_length": "5.1",
        "sepal_width": "3.5",
        "petal_length": "1.4",
        "petal_width": "0.2"
    }
    response = client.post("/", data=valid_data)
    assert response.status_code == 200
    assert any(species in response.data for species in [b"Setosa", b"Versicolor", b"Virginica"])


def test_invalid_input(client):
    """Test handling of non-numeric input."""
    invalid_data = {
        "sepal_length": "abc",
        "sepal_width": "xyz",
        "petal_length": "1.4",
        "petal_width": "0.2"
    }
    response = client.post("/", data=invalid_data)
    assert response.status_code == 200
    assert b"Invalid input! Please enter numeric values." in response.data
