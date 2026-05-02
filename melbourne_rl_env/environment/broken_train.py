import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

df = pd.read_csv("data/Melbourne_housing_FULL.csv")
df = df.dropna(subset=["Price"])

FEATURES = ["Rooms", "Distance", "Landsize"]

df[FEATURES] = df[FEATURES].fillna(0)

X = df[FEATURES]
y = df["Price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.5, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

val_pred = model.predict(X_val)
r2 = r2_score(y_val, val_pred)
print(f"Validation R2: {r2:.4f}")
print(f"(Target for hidden test: R2 >= 0.72)")

joblib.dump(model, "model/saved_model.pkl")
print("Model saved -> model/saved_model.pkl")

test_df = pd.read_csv("data/test_features.csv")

test_X = test_df[FEATURES]

predictions = model.predict(test_X)

output = pd.DataFrame({"predictions": predictions})
output.to_csv("model/test_predictions.csv", index=False)
print("Predictions saved -> model/test_predictions.csv")
