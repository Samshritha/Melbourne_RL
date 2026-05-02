# Melbourne Housing Price Prediction — Fix the ML Pipeline

## Your Task

You are an ML engineer. A housing price prediction pipeline at `/model/train.py`
has been implemented but is performing poorly (R² ≈ 0.38).

Your task is to **diagnose and fix the pipeline** so it achieves **R² ≥ 0.72**
on the hidden test set.

---

## What You Have Access To

| Path | Description |
|---|---|
| `/model/train.py` | The broken pipeline. **Fix this file.** |
| `/data/Melbourne_housing_FULL.csv` | Training dataset (21,368 rows, 21 columns) |
| `/data/test_features.csv` | Test set features — no prices (5,343 rows) |

## What You Must Produce

When you run `python /model/train.py`, it must create:

| Output File | Requirement |
|---|---|
| `/model/saved_model.pkl` | A trained scikit-learn model saved with `joblib.dump()` |
| `/model/test_predictions.csv` | Predictions for every row in `test_features.csv` |

### Exact format required for `test_predictions.csv`:

```
PredictedPrice
485000.0
620000.0
...
```

- **Column name must be exactly:** `PredictedPrice` (case-sensitive)
- **Row count must match** the number of rows in `test_features.csv` (5,343 rows)
- **All values must be positive finite numbers** (no NaN, no negative values)

---

## Dataset Columns Available

```
Suburb, Address, Rooms, Type, Method, SellerG, Date,
Distance, Postcode, Bedroom2, Bathroom, Car,
Landsize, BuildingArea, YearBuilt, CouncilArea,
Lattitude, Longtitude, Regionname, Propertycount, Price
```

`Price` is the target column (only present in training data, not in test_features.csv).

---

## Scoring

Your solution will be evaluated automatically using R² score against hidden test prices.

| R² Achieved | Score |
|---|---|
| < 0.50 | 0.0 |
| 0.50 – 0.65 | 0.1 – 0.5 (linear) |
| 0.65 – 0.72 | 0.5 – 1.0 (linear) |
| ≥ 0.72 | 1.0 |

---

## Rules

- You **may** use: `pandas`, `numpy`, `scikit-learn`, `joblib`
- You **may not** use: `xgboost`, `lightgbm`, `catboost`, `tensorflow`, `torch`, `keras`,
  or any library not installed in the VM
- The file `/model/train.py` must run **without errors** when called as:
  `python /model/train.py`
- You **may** add helper functions or additional code inside `train.py`
- You **may not** manually write predictions to the CSV — they must
  come from `model.predict()`
- You **may not** read from `/data/test_prices_HIDDEN.csv` — that file
  does not exist from your perspective

---

## Hints

- Inspect the dataset columns and their distributions before modelling
- Think carefully about what makes a good feature for housing prices
- Review your preprocessing choices — small decisions here have large downstream effects

---

## Suggested Workflow

```bash
cat /prompt.md                    # Read this file
cat /model/train.py               # Read the current broken script
python /model/train.py            # Run it — observe errors and low R²
nano /model/train.py              # Edit the file
python /model/train.py            # Run again after your fixes
# Repeat until R² on your validation set exceeds 0.72
```