"""Retrain RandomForest with room_per_m2 as the new engineered feature."""
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
csv_path = ROOT / "week4" / "apartments_data_enriched_with_new_features.csv"
out_path = Path(__file__).resolve().parent / "random_forest_regression.pkl"

df = pd.read_csv(csv_path)

for col in ["pop", "pop_dens", "frg_pct", "emp", "tax_income", "rooms", "area", "price"]:
    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace("'", "", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )

# Standardize engineered feature definition across training and inference.
df["room_per_m2"] = (df["area"] / df["rooms"]).round(2)

features = ["rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income", "room_per_m2"]
df = df[features + ["price"]].dropna().drop_duplicates()
print(f"Training rows: {len(df)}")

X = df[features]
y = df["price"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
print("feature_names_in_:", model.feature_names_in_.tolist())

with open(out_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to: {out_path}")
