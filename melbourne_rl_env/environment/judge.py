import os
import sys
import subprocess
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score

TRAIN_PY        = "model/train.py"
SAVED_MODEL     = "model/saved_model.pkl"
PREDICTIONS_CSV = "model/test_predictions.csv"
TEST_FEATURES   = "data/test_features.csv"
HIDDEN_PRICES   = "data/test_prices_HIDDEN.csv"
SCORE_OUTPUT    = "model/judge_score.txt"

BANNED_IMPORTS  = ["xgboost", "lightgbm", "catboost", "tensorflow", "torch", "keras"]

R2_ZERO    = 0.50
R2_MID_LOW = 0.65
R2_FULL    = 0.72


def write_score(score: float):
    score = round(float(score), 4)
    with open(SCORE_OUTPUT, "w") as f:
        f.write(str(score))
    return score


def fail(reason: str) -> float:
    print(f"\nFAILED: {reason}")
    print("Score: 0.0")
    write_score(0.0)
    return 0.0


def passed(check_name: str):
    print(f"PASSED: {check_name}")


def r2_to_score(r2: float) -> float:
    if r2 < R2_ZERO:
        return 0.0
    elif r2 < R2_MID_LOW:
        frac = (r2 - R2_ZERO) / (R2_MID_LOW - R2_ZERO)
        return 0.1 + frac * (0.5 - 0.1)
    elif r2 < R2_FULL:
        frac = (r2 - R2_MID_LOW) / (R2_FULL - R2_MID_LOW)
        return 0.5 + frac * (1.0 - 0.5)
    else:
        return 1.0


def check_1_file_exists():
    if not os.path.exists(TRAIN_PY):
        return fail("train.py not found at model/train.py")
    passed("Check 1: train.py exists")
    return True


def check_2_no_banned_imports():
    with open(TRAIN_PY, "r") as f:
        source = f.read().lower()
    for lib in BANNED_IMPORTS:
        # catches: import x, from x, __import__("x"), import_module("x"), import_module('x')
        patterns = [
            f"import {lib}",
            f"from {lib}",
            f'"{lib}"',
            f"'{lib}'",
        ]
        if any(p in source for p in patterns):
            return fail(f"Banned library detected: {lib}")
    passed("Check 2: No banned libraries")
    return True


def check_3_runs_without_error():
    print("   Running train.py (timeout: 300s)...")
    try:
        result = subprocess.run(
            ["python", TRAIN_PY],
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode != 0:
            return fail(f"train.py exited with error:\n{result.stderr[-1000:]}")
    except subprocess.TimeoutExpired:
        return fail("train.py exceeded 300 second timeout")
    except Exception as e:
        return fail(f"Unexpected error: {e}")
    passed("Check 3: train.py runs without error")
    return True


def check_4_model_saved():
    """
    Verify saved_model.pkl exists and is a valid sklearn model.
    We only check that it loads and has a predict method.
    We do NOT call predict() here because the model expects
    engineered features that are created inside train.py.
    The real prediction check happens in Check 8 via the CSV.
    """
    if not os.path.exists(SAVED_MODEL):
        return fail("saved_model.pkl not found at model/saved_model.pkl")

    try:
        model = joblib.load(SAVED_MODEL)
    except Exception as e:
        return fail(f"saved_model.pkl could not be loaded: {e}")

    if not hasattr(model, "predict"):
        return fail("Loaded object has no .predict() method — not a valid model")

    if not hasattr(model, "fit"):
        return fail("Loaded object has no .fit() method — not a valid model")

    passed("Check 4: Model exists, loads, and is a valid sklearn estimator")
    return True


def check_5_predictions_exist():
    if not os.path.exists(PREDICTIONS_CSV):
        return fail("test_predictions.csv not found at model/test_predictions.csv")
    passed("Check 5: test_predictions.csv exists")
    return True


def check_6_predictions_format():
    try:
        preds_df = pd.read_csv(PREDICTIONS_CSV)
    except Exception as e:
        return fail(f"Could not read test_predictions.csv: {e}"), None

    if "PredictedPrice" not in preds_df.columns:
        return fail(
            f"Column 'PredictedPrice' not found.\n"
            f"   Found: {list(preds_df.columns)}\n"
            f"   Must be named exactly: PredictedPrice"
        ), None

    test_df       = pd.read_csv(TEST_FEATURES)
    expected_rows = len(test_df)
    actual_rows   = len(preds_df)

    if actual_rows != expected_rows:
        return fail(
            f"Row count mismatch: expected {expected_rows}, got {actual_rows}"
        ), None

    preds = preds_df["PredictedPrice"]

    if not np.isfinite(preds).all():
        return fail(f"{(~np.isfinite(preds)).sum()} NaN or infinite values found"), None

    if (preds <= 0).any():
        return fail(f"{(preds <= 0).sum()} non-positive predictions found"), None

    passed(f"Check 6: Format valid ({actual_rows} rows, all positive and finite)")
    return True, preds


def check_7_not_trivial(preds):
    std = preds.std()
    if std < 10_000:
        return fail(f"Predictions appear constant (std=${std:,.0f})")
    passed(f"Check 7: Predictions non-trivial (std=${std:,.0f})")
    return True


def check_8_r2_score(preds):
    hidden = pd.read_csv(HIDDEN_PRICES)["Price"]
    r2     = r2_score(hidden, preds)
    score  = r2_to_score(r2)

    print(f"\n{'='*50}")
    print(f"  R2 Score:     {r2:.4f}")
    print(f"  Reward Score: {score:.4f}")
    print(f"  Threshold:    R2 >= {R2_FULL} -> score = 1.0")
    print(f"{'='*50}")

    if score == 1.0:
        print("  PASSED - Full reward awarded")
    elif score > 0:
        print(f"  PARTIAL - Score {score:.2f}. Improve R2 to reach 1.0")
    else:
        print(f"  FAILED - R2 {r2:.3f} below threshold {R2_ZERO}")

    return score


def run_judge():
    print("="*50)
    print("  RL ENVIRONMENT JUDGE - Melbourne Housing")
    print("="*50)

    if not check_1_file_exists():      return write_score(0.0)
    if not check_2_no_banned_imports(): return write_score(0.0)
    if not check_3_runs_without_error(): return write_score(0.0)
    if not check_4_model_saved():      return write_score(0.0)
    if not check_5_predictions_exist(): return write_score(0.0)

    format_ok, preds = check_6_predictions_format()
    if not format_ok:                  return write_score(0.0)
    if not check_7_not_trivial(preds): return write_score(0.0)

    score = check_8_r2_score(preds)
    write_score(score)
    print(f"\nScore written to {SCORE_OUTPUT}")
    return score


if __name__ == "__main__":
    run_judge()