from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import time


app = FastAPI(
    title="Iris Species Prediction API",
    description="Predicts Iris flower species",
    version="1.0"
)

model = joblib.load("model/iris_model.joblib")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def root():
    return {
        "message": "Iris Species Prediction API",
        "status": "running"
    }


@app.post("/predict")
def predict(data: IrisInput):

    start_time = time.perf_counter()

    X = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]

    prediction = model.predict(X)[0]

    probabilities = model.predict_proba(X)[0]

    classes = model.named_steps["classifier"].classes_

    probability = float(
        probabilities[list(classes).index(prediction)]
    )

    latency = (
        time.perf_counter() - start_time
    ) * 1000

    return {
        "prediction": prediction,
        "probability": probability,
        "latency_ms": latency
    }