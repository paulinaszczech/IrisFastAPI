import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.main import app
from fastapi.testclient import TestClient

# Stworzenie instancji klienta testowego dla FastAPI.
client = TestClient(app)


# Test przewidywania dla Iris-setosa
def test_predict_iris_setosa():
    response = client.post(
        "/predict",
        json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, 
              "petal_width": 0.2}
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "Iris-setosa"}


# Test przewidywania dla Iris-versicolor
def test_predict_iris_versicolor():
    response = client.post(
        "/predict",
        json={"sepal_length": 6.0, "sepal_width": 2.9, "petal_length": 4.5, 
              "petal_width": 1.5}
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "Iris-versicolor"}


# Test przewidywania dla Iris-virginica
def test_predict_iris_virginica():
    response = client.post(
        "/predict",
        json={"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, 
              "petal_width": 2.5}
    )
    assert response.status_code == 200
    assert response.json() == {"prediction": "Iris-virginica"}