import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# Load data
df = pd.read_csv("data/iris.csv")

FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

TARGET = "species"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load model
model = joblib.load("model/iris_model.joblib")

# Get scaler and classifier
scaler = model.named_steps["scaler"]
classifier = model.named_steps["classifier"]

# Scale test data
X_test_scaled = scaler.transform(X_test)

# SHAP LinearExplainer
explainer = shap.LinearExplainer(
    classifier,
    scaler.transform(X_train)
)

shap_values = explainer(X_test_scaled)

print("SHAP analysis completed.")

# Feature importance
mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))

importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": mean_abs_shap
}).sort_values(
    "mean_abs_shap",
    ascending=False
)

print("\nFeature importance:")
print(importance)

# Summary plot
shap.summary_plot(
    shap_values,
    X_test_scaled,
    feature_names=FEATURES,
    show=False
)

plt.tight_layout()
plt.savefig(
    "explainability/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

print("SHAP plot saved.")