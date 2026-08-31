import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference
)

# --------------------------------------------------
# Load data
# --------------------------------------------------
df = pd.read_csv("data/iris.csv")

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

X = df[FEATURES]

# Binary target:
# Virginica = 1
# Other species = 0
y = (df["species"] == "virginica").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Load model trained on the original 3-class Iris target
# --------------------------------------------------
model = joblib.load("model/iris_model.joblib")

# Get original species predictions
pred_species = model.predict(X_test)

# Convert predictions to binary
y_pred = (pred_species == "virginica").astype(int)

# --------------------------------------------------
# Synthetic audit-only sensitive attribute (random)
# --------------------------------------------------
# IMPORTANT:
# This is NOT a real demographic attribute.
# It is only used to demonstrate Fairlearn.

np.random.seed(42)  # reproducibility
audit_group = np.random.randint(0, 2, size=len(X_test))

print("Audit group counts:")
print(pd.Series(audit_group).value_counts())

# --------------------------------------------------
# Fairness metrics
# --------------------------------------------------
dp_difference = demographic_parity_difference(
    y_test,
    y_pred,
    sensitive_features=audit_group
)

eo_difference = equalized_odds_difference(
    y_test,
    y_pred,
    sensitive_features=audit_group
)

print("\nFairness Results")
print("-----------------------")
print("Demographic Parity Difference:", dp_difference)
print("Equalized Odds Difference:", eo_difference)
