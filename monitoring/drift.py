import pandas as pd
import numpy as np

from scipy.stats import ks_2samp


FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]


# Training data
train_df = pd.read_csv(
    "data/iris.csv"
)


# Create synthetic/new production data
np.random.seed(42)

drift_df = train_df[FEATURES].copy()

# Introduce distribution changes
drift_df["sepal_length"] += np.random.normal(
    0.3,
    0.1,
    len(drift_df)
)

drift_df["sepal_width"] += np.random.normal(
    -0.1,
    0.05,
    len(drift_df)
)

drift_df["petal_length"] += np.random.normal(
    0.5,
    0.15,
    len(drift_df)
)

drift_df["petal_width"] += np.random.normal(
    0.2,
    0.05,
    len(drift_df)
)


print("KS Test Results")
print("=" * 50)

results = []


for feature in FEATURES:

    statistic, p_value = ks_2samp(
        train_df[feature],
        drift_df[feature]
    )

    drift_detected = p_value < 0.05

    results.append({
        "feature": feature,
        "KS_statistic": statistic,
        "p_value": p_value,
        "drift_detected": drift_detected
    })


results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(
    "monitoring/drift_results.csv",
    index=False
)