from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import gradio as gr
import numpy as np
import pandas as pd


APP_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATE_PATHS = [
    APP_DIR / "random_forest_regression.pkl",
    APP_DIR / "week3" / "apartment" / "random_forest_regression.pkl",
    APP_DIR.parent / "week3" / "apartment" / "random_forest_regression.pkl",
]

BFS_CANDIDATE_PATHS = [
    APP_DIR / "bfs_municipality_and_tax_data.csv",
    APP_DIR / "week3" / "apartment" / "bfs_municipality_and_tax_data.csv",
    APP_DIR.parent / "week3" / "apartment" / "bfs_municipality_and_tax_data.csv",
]

REQUIRED_BFS_COLUMNS = ["bfs_number", "bfs_name", "pop", "pop_dens", "frg_pct", "emp", "tax_income"]
FALLBACK_FEATURE_ORDER = ["rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income", "room_per_m2"]
ZURICH_BFS_NUMBERS = {
    1, 2, 3, 4, 6, 9, 10, 11, 13, 25, 27, 28, 29, 31, 38, 39, 51, 52, 53, 54,
    56, 58, 59, 60, 62, 66, 67, 68, 69, 72, 83, 84, 86, 89, 90, 91, 92, 94,
    95, 96, 98, 100, 102, 112, 113, 115, 116, 117, 118, 120, 121, 131, 135,
    136, 138, 139, 141, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 172,
    173, 177, 178, 180, 191, 193, 194, 195, 196, 198, 199, 200, 214, 219, 221,
    223, 224, 225, 227, 228, 230, 231, 241, 242, 243, 244, 245, 247, 248, 249,
    250, 251, 261, 295, 296, 297,
}


def find_existing_file(candidate_paths: list[Path], file_label: str) -> Path:
    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path
    searched = "\n".join(f"- {p}" for p in candidate_paths)
    raise FileNotFoundError(
        f"{file_label} not found. Searched:\n{searched}\n"
        "Place the required file next to app.py for Hugging Face Spaces deployment."
    )


def clean_numeric_series(series: pd.Series) -> pd.Series:
    # Handles values like "108'788", "108 788", and mixed object types.
    return pd.to_numeric(
        series.astype(str).str.replace("'", "", regex=False).str.replace(" ", "", regex=False),
        errors="coerce",
    )


def load_model(model_path: Path) -> Any:
    with model_path.open("rb") as f:
        model = pickle.load(f)
    return model


def load_bfs_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8")

    missing_columns = [c for c in REQUIRED_BFS_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(f"BFS file is missing required columns: {missing_columns}")

    for col in ["pop", "pop_dens", "frg_pct", "emp", "tax_income"]:
        df[col] = clean_numeric_series(df[col])

    # Keep only municipalities used in the original Zürich assignment notebook.
    df = df[df["bfs_number"].isin(ZURICH_BFS_NUMBERS)].copy()
    df = df[df["bfs_name"].notna()].copy()
    df["bfs_name"] = df["bfs_name"].astype(str).str.strip()

    # Fill missing municipal indicators using column medians for prediction robustness.
    numeric_cols = ["pop", "pop_dens", "frg_pct", "emp", "tax_income"]
    for col in numeric_cols:
        if df[col].isna().all():
            raise ValueError(f"BFS column '{col}' contains only missing values.")
        df[col] = df[col].fillna(df[col].median())

    return df


def get_model_feature_order(model: Any) -> list[str]:
    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None:
        return [str(f) for f in feature_names]

    # Fallback based on the verified schema in apartment.ipynb.
    return FALLBACK_FEATURE_ORDER.copy()


def validate_user_inputs(rooms: float, area: float, town: str) -> str | None:
    if rooms is None or area is None:
        return "Rooms and area are required."
    if rooms <= 0:
        return "Rooms must be greater than 0."
    if area <= 0:
        return "Area must be greater than 0."
    if not town or not str(town).strip():
        return "Please select a municipality."
    return None


def build_single_input_row(
    rooms: float,
    area: float,
    town: str,
    bfs_df: pd.DataFrame,
    expected_features: list[str],
) -> pd.DataFrame:
    town_clean = str(town).strip()

    if town_clean not in set(bfs_df["bfs_name"]):
        raise ValueError(f"Invalid municipality: '{town_clean}'.")

    row = bfs_df[bfs_df["bfs_name"] == town_clean].copy()
    if row.empty:
        raise ValueError(f"No BFS row found for municipality '{town_clean}'.")

    # If duplicates exist, use first but keep behavior explicit.
    if len(row) > 1:
        row = row.sort_values(by="bfs_number", ascending=True).head(1).copy()

    row.loc[:, "rooms"] = float(rooms)
    row.loc[:, "area"] = float(area)
    # New engineered feature: average m² per room (introduced in week 2 feature engineering).
    row.loc[:, "room_per_m2"] = round(float(area) / float(rooms), 2)


    available_cols = set(row.columns)
    missing_feature_values = [c for c in expected_features if c not in available_cols]
    if missing_feature_values:
        raise ValueError(
            "Feature mismatch: model expects columns not present in constructed input: "
            f"{missing_feature_values}"
        )

    feature_df = row.reindex(columns=expected_features)

    if feature_df.isna().any(axis=None):
        na_cols = feature_df.columns[feature_df.isna().any()].tolist()
        raise ValueError(f"Feature dataframe contains missing values in columns: {na_cols}")

    return feature_df


def format_chf(amount: float) -> str:
    rounded = int(np.round(amount, 0))
    formatted = f"{rounded:,}".replace(",", "'")
    return f"Estimated monthly rent: CHF {formatted}"


def create_predictor(model: Any, bfs_df: pd.DataFrame):
    expected_features = get_model_feature_order(model)

    missing_in_bfs = [
        col for col in expected_features if col not in {"rooms", "area", "room_per_m2"} and col not in bfs_df.columns
    ]
    if missing_in_bfs:
        raise ValueError(
            "BFS data does not contain required model features: "
            f"{missing_in_bfs}."
        )

    def predict_rent(rooms: float, area: float, town: str) -> str:
        validation_error = validate_user_inputs(rooms, area, town)
        if validation_error:
            return f"Input error: {validation_error}"

        try:
            input_df = build_single_input_row(rooms, area, town, bfs_df, expected_features)
            pred = model.predict(input_df)
            if len(pred) == 0:
                return "Prediction error: model returned an empty output."
            return format_chf(float(pred[0]))
        except Exception as exc:  # Robust app-level error handling
            return f"Prediction error: {exc}"

    return predict_rent


def build_interface() -> gr.Interface:
    model_path = find_existing_file(MODEL_CANDIDATE_PATHS, "Model file (random_forest_regression.pkl)")
    bfs_path = find_existing_file(BFS_CANDIDATE_PATHS, "BFS municipality data file")

    model = load_model(model_path)
    bfs_df = load_bfs_data(bfs_path)

    towns = sorted(bfs_df["bfs_name"].dropna().astype(str).unique().tolist())
    predictor = create_predictor(model, bfs_df)

    return gr.Interface(
        fn=predictor,
        inputs=[
            gr.Number(label="Number of rooms", value=3.0, precision=1),
            gr.Number(label="Living area (m²)", value=80.0, precision=1),
            gr.Dropdown(choices=towns, label="Municipality / Town", value="Zürich" if "Zürich" in towns else towns[0]),
        ],
        outputs=gr.Textbox(label="Prediction"),
        title="Apartment Price Prediction – Canton of Zurich",
        description=(
            "Predict monthly apartment rent in CHF based on apartment inputs and municipality-level BFS indicators."
        ),
        examples=[
            [2.5, 65, "Winterthur" if "Winterthur" in towns else towns[0]],
            [3.5, 90, "Zürich" if "Zürich" in towns else towns[0]],
            [4.5, 120, "Dietlikon" if "Dietlikon" in towns else towns[0]],
        ],
    )


def main() -> None:
    app = build_interface()
    app.launch()


if __name__ == "__main__":
    main()
