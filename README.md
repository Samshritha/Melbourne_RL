# Melbourne Housing RL Environment

An RL training environment where a language model must diagnose and fix
a broken ML pipeline.

---

## What the LLM Must Do

The LLM is given a broken `train.py` that achieves **R² ≈ 0.38**.
It must fix it to achieve **R² ≥ 0.72** on a hidden test set using only
`pandas`, `numpy`, `scikit-learn`, and `joblib`.

---

## Project Structure

```
melbourne_rl_env/
│
└── environment/
    ├── prompt.md              ← What the LLM reads as its task
    ├── broken_train.py        ← Deliberately buggy starter script (7 bugs)
    ├── setup_data.py          ← Creates the hidden train/test split (run once)
    ├── setup_vm.sh            ← VM initialization script (run once, Linux only)
    ├── judge.py               ← Automated 7-step judge (0.0 – 1.0)
    ├── reference_solution.py  ← DO NOT show to LLM (for testing only)
    └── README.md              ← This file
```

At runtime inside the VM, the layout becomes:

```
/
├── prompt.md                        ← LLM reads this
├── data/
│   ├── Melbourne_housing_FULL.csv   ← LLM has access (21,368 rows, training only)
│   ├── test_features.csv            ← LLM has access (5,343 rows, no prices)
│   └── test_prices_HIDDEN.csv       ← HIDDEN — judge only
└── model/
    ├── train.py                     ← LLM edits this
    ├── saved_model.pkl              ← LLM must produce this
    ├── test_predictions.csv         ← LLM must produce this
    └── judge_score.txt              ← Judge writes score here
```

---

## Dataset Facts (Verified)

| Stat | Value |
|---|---|
| Raw dataset rows | 34,857 |
| After dropping missing Price | 27,247 |
| After outlier removal (1st–99th percentile) | 26,711 |
| Price range after cleaning | $310,000 – $3,400,540 |
| Train set saved to Melbourne_housing_FULL.csv (seed 99) | 21,368 rows |
| Test set (seed 99) — hidden from LLM | 5,343 rows |

**Note:** `Melbourne_housing_FULL.csv` contains only the training portion (21,368 rows). The 5,343 test rows are never written to any LLM-accessible file.

---

## Deployment: Step by Step

### Step 1 — Get the Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/anthonypino/melbourne-housing-market

File name: `Melbourne_housing_FULL.csv`

Place it in the `data/` folder before running setup.

### Step 2 — Run setup_data.py (creates the hidden split)
```bash
python environment/setup_data.py
```

This creates four files in `data/`:
- `Melbourne_housing_FULL.csv` (cleaned)
- `test_features.csv` (LLM can see this)
- `test_prices_HIDDEN.csv` (judge only)
- `test_full_HIDDEN.csv` (judge only)

### Step 3 — On Linux VM only: run setup_vm.sh
```bash
chmod +x /environment/setup_vm.sh
bash /environment/setup_vm.sh
```

### Step 4 — Start the LLM's task
Tell the LLM:
```
cat /prompt.md
```

### Step 5 — Run the judge after LLM finishes
```bash
python environment/judge.py
cat model/judge_score.txt    # The reward score: 0.0 – 1.0
```

---

## The 8 Bugs in `broken_train.py`

| # | Bug | Impact |
|---|-----|--------|
| 1 | Only 3 features used (Rooms, Distance, Landsize) | Misses most signal |
| 2 | Missing values filled with 0 | Distorts distributions |
| 3 | No price outlier removal | Extreme values hurt model fit |
| 4 | `test_size=0.5` wastes half the data | Less training data |
| 5 | Uses `LinearRegression` on non-linear data | Wrong model class |
| 6 | Test features not imputed before predict | Crash on NaN input |
| 7 | Wrong column name: `predictions` not `PredictedPrice` | Judge format fail |
| 8 | Data leakage trap for future development | Subtle generalisation failure |

The reference solution using `RandomForestRegressor` achieves:
- Validation R²: **0.8077**
- Hidden test R²: **0.8911**

---

## The 8 Judge Checks

| Step | Check | Fail → Score |
|------|-------|-------------|
| 1 | `model/train.py` exists | 0.0 |
| 2 | No banned libraries (xgboost, lightgbm, catboost, tensorflow, torch, keras) | 0.0 |
| 3 | `train.py` runs without error (300s timeout) | 0.0 |
| 4 | `saved_model.pkl` exists and is a valid sklearn estimator (has fit and predict methods) | 0.0 |
| 5 | `test_predictions.csv` exists | 0.0 |
| 6 | Column named `PredictedPrice`, row count = 5,343, all finite and positive | 0.0 |
| 7 | Predictions are non-trivial (std > $10,000) | 0.0 |
| 8 | R² vs hidden prices → piecewise linear score | 0.0 – 1.0 |

---

## Scoring Table

| R² Achieved | Score |
|-------------|-------|
| < 0.50 | 0.0 |
| 0.50 – 0.65 | 0.1 – 0.5 (linear) |
| 0.65 – 0.72 | 0.5 – 1.0 (linear) |
| ≥ 0.72 | 1.0 |

---

## Reward Hacking Analysis

| Attack | Defence |
|--------|---------|
| LLM joins `test_features.csv` with `Melbourne_housing_FULL.csv` to look up prices | `Melbourne_housing_FULL.csv` contains **only the training 80%** (21,368 rows) — test rows are absent, so no join is possible |
| LLM saves empty or untrained model | Check 4 verifies valid sklearn estimator. Check 8 evaluates actual R² against hidden prices — any model that cannot predict well scores 0.0 regardless |
| LLM returns constant median prediction | Check 7 requires std > $10,000 — constant predictions score 0.0 |
| LLM uses banned libraries (xgboost, torch, etc.) | Check 2 scans all import patterns including string literals and dynamic imports — any banned library → 0.0 |
| LLM inflates predictions to match training distribution | Evaluated against true hidden prices — cannot be gamed |

---

## Reward Denial Analysis

| Risk | Defence |
|------|---------|
| Floating point variance | Threshold is 0.72, not 1.0 — tolerant of minor numerical variation |
| Different valid feature sets | Any valid sklearn model that generalises will pass |
| Model trains slowly on CPU | 300-second timeout is generous for Random Forest on this dataset |
| Column name confusion | Exact name `PredictedPrice` stated explicitly in prompt.md and this README |

---

## Testing Before Deployment

### Test 1 — Reference solution should score 1.0
```bash
copy environment\reference_solution.py model\train.py
python model/train.py
python environment/judge.py
cat model/judge_score.txt    # Expected: 1.0
```

### Test 2 — Broken script should score 0.0
```bash
copy environment\broken_train.py model\train.py
python environment/judge.py
cat model/judge_score.txt    # Expected: 0.0
```

---

## Packages Required

```
pandas
numpy
scikit-learn
joblib
```

No GPU required. All computation runs on CPU.
