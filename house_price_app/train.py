# train.py

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib
import os

# ============================================================
# Load California Housing Dataset
# ============================================================
print("Loading California Housing dataset...")
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# ============================================================
# Train/Test Split
# ============================================================
print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# Build Training Pipeline
# ============================================================
model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])

# ============================================================
# Train Model
# ============================================================
print("Training model...")
model.fit(X_train, y_train)

# ============================================================
# Score Model
# ============================================================
score = model.score(X_test, y_test)
print(f"Model R² Score: {score:.4f}")

# ============================================================
# Save Model
# ============================================================
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")

print("Model saved to models/best_model.pkl")
