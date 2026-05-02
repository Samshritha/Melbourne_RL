import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/Melbourne_housing_FULL.csv")
df = df.dropna(subset=["Price"])

q01 = df["Price"].quantile(0.01)
q99 = df["Price"].quantile(0.99)
df  = df[(df["Price"] >= q01) & (df["Price"] <= q99)]

CAT_COLS = ["Type", "Regionname", "Method"]

# Fit encoders on full training data so test categories are handled consistently
encoders = {}
for col in CAT_COLS:
    le = LabelEncoder()
    df[col] = df[col].fillna("Unknown")
    le.fit(df[col].astype(str))
    df[col + "_enc"] = le.transform(df[col].astype(str))
    encoders[col] = le


def engineer_features(df, encoders):
    df = df.copy()
    df["House_Age"]    = (2024 - df["YearBuilt"]).clip(0, 200)
    df["Rooms_x_Bath"] = df["Rooms"] * df["Bathroom"].fillna(1)
    df["AreaPerRoom"]  = df["BuildingArea"] / (df["Rooms"] + 1)
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)
            # map unseen categories to a safe fallback
            df[col] = df[col].apply(
                lambda v: v if v in le.classes_ else le.classes_[0]
            )
            df[col + "_enc"] = le.transform(df[col])
    return df


df = engineer_features(df, encoders)

FEATURES = [
    "Rooms", "Distance", "Bedroom2", "Bathroom", "Car",
    "Landsize", "BuildingArea", "House_Age", "AreaPerRoom",
    "Rooms_x_Bath", "Lattitude", "Longtitude",
    "Propertycount", "Type_enc", "Regionname_enc", "Method_enc"
]

train_medians = {}
for col in FEATURES:
    if col in df.columns and df[col].isnull().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        train_medians[col] = median_val

X = df[FEATURES]
y = df["Price"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

val_preds = model.predict(X_val)
r2 = r2_score(y_val, val_preds)
print(f"Validation R2: {r2:.4f}  (target: >= 0.72)")

joblib.dump(model, "model/saved_model.pkl")
print("Model saved -> model/saved_model.pkl")


def prepare_test(test_df):
    test_df = engineer_features(test_df, encoders)
    for col in FEATURES:
        if col in test_df.columns and test_df[col].isnull().any():
            test_df[col] = test_df[col].fillna(train_medians.get(col, 0))
        elif col not in test_df.columns:
            test_df[col] = 0
    return test_df[FEATURES]


test_raw    = pd.read_csv("data/test_features.csv")
test_X      = prepare_test(test_raw)
predictions = np.clip(model.predict(test_X), 0, None)

output = pd.DataFrame({"PredictedPrice": predictions})
output.to_csv("model/test_predictions.csv", index=False)
print(f"Predictions saved -> model/test_predictions.csv ({len(output)} rows)")
print(f"Prediction range: ${predictions.min():,.0f} - ${predictions.max():,.0f}")
