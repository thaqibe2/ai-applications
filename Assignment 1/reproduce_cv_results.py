import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def summarize(cvres: dict) -> dict:
    r2 = cvres["test_r2"]
    rmse = -cvres["test_rmse"]
    mae = -cvres["test_mae"]
    return {
        "mean_r2": float(np.mean(r2)),
        "std_r2": float(np.std(r2, ddof=1)),
        "mean_rmse": float(np.mean(rmse)),
        "std_rmse": float(np.std(rmse, ddof=1)),
        "mean_mae": float(np.mean(mae)),
        "std_mae": float(np.std(mae, ddof=1)),
    }


def main() -> None:
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    results = {}

    # Iteration 1 (week1 baseline)
    df1 = pd.read_csv("../week1/original_apartment_data_analytics_hs24.csv", encoding="utf-8")
    df1 = df1.dropna().drop_duplicates()
    features1 = ["rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income"]
    X1 = df1[features1]
    y1 = df1["price"]

    results["iteration1"] = {
        "dataset": "week1/original_apartment_data_analytics_hs24.csv",
        "rows_after_cleaning": int(df1.shape[0]),
        "features": features1,
        "LinearRegression": summarize(cross_validate(LinearRegression(), X1, y1, cv=cv, scoring=scoring)),
        "RandomForestRegressor_random_state_42": summarize(
            cross_validate(RandomForestRegressor(random_state=42), X1, y1, cv=cv, scoring=scoring)
        ),
    }

    # Iteration 2 (week4 improved)
    df2 = pd.read_csv("../week4/apartments_data_enriched_with_new_features.csv", encoding="utf-8")
    df2 = df2.dropna().drop_duplicates()

    features2_notebook = [
        "rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income",
        "room_per_m2", "luxurious", "temporary", "furnished", "area_cat_ecoded",
        "zurich_city", "avg_price_postal_rooms_area",
    ]
    features2_available = [c for c in features2_notebook if c in df2.columns]
    missing = [c for c in features2_notebook if c not in df2.columns]

    X2 = df2[features2_available]
    y2 = df2["price"]

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=(16, 16),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=42,
                validation_fraction=0.1,
            ),
        ),
    ])

    results["iteration2"] = {
        "dataset": "week4/apartments_data_enriched_with_new_features.csv",
        "rows_after_cleaning": int(df2.shape[0]),
        "features_from_notebook": features2_notebook,
        "missing_notebook_features_in_current_dataset": missing,
        "features_used_for_reproducible_cv": features2_available,
        "RandomForestRegressor_random_state_42": summarize(
            cross_validate(RandomForestRegressor(random_state=42), X2, y2, cv=cv, scoring=scoring)
        ),
        "MLPRegressor_16_16_scaled": summarize(cross_validate(mlp, X2, y2, cv=cv, scoring=scoring)),
    }

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
