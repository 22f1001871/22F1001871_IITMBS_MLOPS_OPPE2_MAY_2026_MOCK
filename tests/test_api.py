# tests/test_api.py
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Iris Species Prediction API"
    assert body["status"] == "running"

def test_predict_endpoint():
    # Example input values for Iris flower
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()

    # Check that keys exist in response
    assert "prediction" in body
    assert "probability" in body
    assert "latency_ms" in body

    # Validate types
    assert isinstance(body["prediction"], str)
    assert isinstance(body["probability"], float)
    assert isinstance(body["latency_ms"], float)
