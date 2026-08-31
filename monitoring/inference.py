import pandas as pd
import requests
import random
import time
import json


API_URL = "http://34.9.52.121/predict"

df = pd.read_csv("data/iris.csv")

samples = df.sample(
    n=100,
    random_state=42
)


logs = []


for _, row in samples.iterrows():

    payload = {
        "sepal_length": float(row["sepal_length"]),
        "sepal_width": float(row["sepal_width"]),
        "petal_length": float(row["petal_length"]),
        "petal_width": float(row["petal_width"])
    }

    start = time.perf_counter()

    response = requests.post(
        API_URL,
        json=payload
    )

    end = time.perf_counter()

    latency = (end - start) * 1000

    result = response.json()

    logs.append({
        "input": payload,
        "actual": row["species"],
        "prediction": result["prediction"],
        "probability": result["probability"],
        "latency_ms": latency
    })


logs_df = pd.DataFrame(logs)

logs_df.to_csv(
    "monitoring/inference_logs.csv",
    index=False
)


print("100 inference requests completed.")

print(
    "Average latency:",
    logs_df["latency_ms"].mean(),
    "ms"
)

print(
    "P95 latency:",
    logs_df["latency_ms"].quantile(0.95),
    "ms"
)

print("\nPrediction counts:")
print(logs_df["prediction"].value_counts())