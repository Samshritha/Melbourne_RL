import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
 
print("Loading Melbourne_housing_FULL.csv...")
df = pd.read_csv("data/Melbourne_housing_FULL.csv")
print(f"  Raw dataset: {len(df):,} rows, {len(df.columns)} columns")
 
df = df.dropna(subset=["Price"])
print(f"  After dropping missing Price: {len(df):,} rows")
 
p1  = df["Price"].quantile(0.01)
p99 = df["Price"].quantile(0.99)
df  = df[(df["Price"] >= p1) & (df["Price"] <= p99)]
print(f"  After outlier removal: {len(df):,} rows")
print(f"  Price range: ${p1:,.0f} - ${p99:,.0f}")
 
train_df, test_df = train_test_split(df, test_size=0.20, random_state=99)
print(f"\n  Train set: {len(train_df):,} rows")
print(f"  Test set:  {len(test_df):,} rows")
 
train_df.to_csv("data/Melbourne_housing_FULL.csv", index=False)
print("\n  Saved training dataset (test rows excluded)")
 
test_df.drop(columns=["Price"]).to_csv("data/test_features.csv", index=False)
print(f"  Saved test features ({len(test_df):,} rows)")
 
test_df[["Price"]].reset_index(drop=True).to_csv("data/test_prices_HIDDEN.csv", index=False)
print("  Saved hidden prices (HIDDEN FROM LLM)")
 
test_df.reset_index(drop=True).to_csv("data/test_full_HIDDEN.csv", index=False)
 
print("\nSetup complete. Random seed: 99")