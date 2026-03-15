# Apartment Price Prediction – Canton of Zurich

## 1) Project Overview
This repository contains a machine learning regression application that predicts **monthly apartment rent (CHF)** for municipalities in the canton of Zurich.

The deployed app uses a trained `RandomForestRegressor` and combines:
- apartment-level inputs (`rooms`, `area`)
- municipality-level BFS socioeconomic variables

## 2) Application
Public app link:

**ADD_YOUR_HUGGING_FACE_SPACE_LINK_HERE**

## 3) Project Files
- `app.py`  
  Gradio application for rent prediction (inference).
- `random_forest_regression.pkl`  
  Trained final regression model artifact used by the app.
- `bfs_municipality_and_tax_data.csv`  
  Municipality-level BFS data used during inference.
- `apartment.ipynb` (week 3)  
  Main deployment notebook reference: model loading, feature schema, BFS merge logic, Gradio prototype.
- `week1/apartment_regression_price_prediction.ipynb`  
  Baseline modeling comparison (Linear Regression vs Random Forest).
- `week4/apartment_regression_price_prediction.ipynb`  
  Extended experiments (Random Forest, `MLPRegressor`, PyTorch regressor).
- `requirements.txt`  
  Dependencies for local run and Hugging Face Spaces.

## 4) Dataset and Features
### Apartment-level features
- `rooms`
- `area`

### Municipality-level BFS features
- `pop`
- `pop_dens`
- `frg_pct`
- `emp`
- `tax_income`

### Feature schema used by the deployed model artifact
From `apartment.ipynb` and model metadata (`feature_names_in_`):

`['rooms', 'area', 'pop', 'pop_dens', 'frg_pct', 'emp', 'tax_income', 'room_per_m2']`

## 5) Preprocessing
### Training-side preprocessing (verified from week 1 / week 4 notebooks)
- Load apartment dataset from CSV
- Remove missing rows (`dropna`)
- Remove duplicate rows (`drop_duplicates`)
- Select modeling feature sets
- Use `train_test_split(..., test_size=0.20, random_state=42)`

### Inference-side preprocessing (verified from `apartment.ipynb` and implemented in `app.py`)
- Load BFS municipality CSV
- Restrict municipality options to the verified canton of Zurich municipalities used in the original apartment notebook
- Clean `tax_income` by removing apostrophes (`'`) and converting to numeric
- Convert `pop`, `pop_dens`, `frg_pct`, `emp`, `tax_income` to numeric
- Handle missing BFS numeric values with median imputation (robust inference)
- Match selected municipality to BFS row
- Inject user-provided `rooms` and `area`
- Compute engineered feature `room_per_m2 = area / rooms` (average m² per room)
- Reorder columns to exact model feature order (`model.feature_names_in_`)

## 6) Feature Engineering (New Feature Requirement)
### New feature: `room_per_m2`

**Definition:** `room_per_m2 = area / rooms` — the average living area per room in m².

**Origin:** Introduced in the Week 2 feature engineering notebook (`week2/feature_engineering.ipynb`). Not present in the Week 1 baseline.

**Rationale:** `rooms` and `area` individually are informative, but their ratio captures apartment density more precisely — a 3-room 120m² apartment is fundamentally different from a 3-room 60m² apartment. This ratio helps the model distinguish spacious from cramped apartments of the same room count.

**Usage in app:** Computed at inference time from user inputs (`area / rooms`) before constructing the model input row. No additional user input is required.

## 7) Iterative Modeling Process
> Cross-validation results below were reproduced now from existing notebooks and datasets using **5-fold KFold CV** (`n_splits=5`, `shuffle=True`, `random_state=42`) with sklearn scoring for `r2`, `neg_root_mean_squared_error`, and `neg_mean_absolute_error`.

| Iteration | Objective | Dataset & feature subset | Data preprocessing | Models used | Hyperparameters (verified) | 5-fold CV results (mean ± std) | Outcome |
|---|---|---|---|---|---|---|---|
| 1 (Week 1 baseline) | Establish baseline with core apartment + BFS variables | Dataset: `week1/original_apartment_data_analytics_hs24.csv` (819 rows after cleaning). Features: `rooms, area, pop, pop_dens, frg_pct, emp, tax_income` | `dropna`, `drop_duplicates` | 1) `LinearRegression` 2) `RandomForestRegressor` | Linear Regression: defaults. RF: `random_state=42` (default `n_estimators=100`) | **LinearRegression**: R² = **0.5360 ± 0.0901**, RMSE = **857.4671 ± 160.7545**, MAE = **523.8686 ± 48.7087**. **RandomForestRegressor**: R² = **0.4908 ± 0.1162**, RMSE = **891.5083 ± 151.1379**, MAE = **528.6937 ± 37.8436**. | In 5-fold CV, Linear Regression is slightly better than default Random Forest on this baseline 7-feature setup. |
| 2 (Week 4 improved) | Improve predictive quality with richer non-linear feature set | Dataset: `week4/apartments_data_enriched_with_new_features.csv` (2344 rows after cleaning). Notebook feature list contains 14 variables, but `avg_price_postal_rooms_area` is missing in this dataset file; reproducible CV used the 13 available features: `rooms, area, pop, pop_dens, frg_pct, emp, tax_income, room_per_m2, luxurious, temporary, furnished, area_cat_ecoded, zurich_city`. | `dropna`, `drop_duplicates`; `StandardScaler` for MLP pipeline | 1) `RandomForestRegressor` 2) `MLPRegressor` | RF: `random_state=42`. MLP: `hidden_layer_sizes=(16,16)`, `activation='relu'`, `solver='adam'`, `max_iter=200`, `validation_fraction=0.1`, `random_state=42` | **RandomForestRegressor**: R² = **0.6766 ± 0.0277**, RMSE = **611.9815 ± 20.1678**, MAE = **406.8635 ± 15.1592**. **MLPRegressor (scaled)**: R² = **0.5198 ± 0.0512**, RMSE = **746.0197 ± 49.1082**, MAE = **529.3717 ± 24.4519**. | RF clearly outperforms MLP in reproduced 5-fold CV and remains the strongest verified model for deployment. |

## 8) Final Selected Model
- **Selected model:** `RandomForestRegressor`
- **Deployed artifact:** `random_forest_regression.pkl`
- **Artifact parameters (verified):**
  - `n_estimators=100`
  - `random_state=42`
  - remaining parameters at sklearn defaults
- **Input features (exact order):**
  - `rooms`, `area`, `pop`, `pop_dens`, `frg_pct`, `emp`, `tax_income`, `room_per_m2` *(engineered)*

The deployed application uses the verified 8-feature `RandomForestRegressor` artifact (7 base features + engineered `room_per_m2`) available in the repository, ensuring consistency between the documented application and the production inference pipeline.

## 9) Evaluation Method
- The iterative section reports reproduced **5-fold cross-validation** metrics (`R^2`, RMSE, MAE) with mean and standard deviation.
- CV protocol used for reproducibility: `KFold(n_splits=5, shuffle=True, random_state=42)`.
- Hold-out train/test metrics from notebooks are still useful as supplementary diagnostics but are not the only evidence anymore.
- Reproducibility caveat: in week 4, one notebook feature (`avg_price_postal_rooms_area`) is absent from the current dataset file, so CV was computed on the verified available subset and documented explicitly.

## 10) Application Description
The app flow is:
1. User enters `rooms`, `area`, and `town`.
2. App retrieves the matching BFS municipality row from the verified Zurich municipality subset.
3. App cleans and validates data, then constructs the final input row in exact model feature order.
4. App returns formatted predicted monthly rent in CHF.

## 11) Limitations
- Deployed model uses 8 features (including the engineered `room_per_m2`); further categorical features from week 4 experiments (`luxurious`, `furnished`, etc.) are not included in the deployed artifact.
- Municipality-level indicators are proxies and do not fully capture micro-location effects.
- Important housing characteristics (condition, renovation quality, floor, amenities) are not included.
- A model pickle created with one sklearn version can show compatibility warnings in another version.

## 12) Author
Besfort Thaqi

---

## Submission Checklist
- [ ] Replace `ADD_YOUR_HUGGING_FACE_SPACE_LINK_HERE` with the public Space URL
- [ ] (Optional) If you regenerate week 4 data with `avg_price_postal_rooms_area`, rerun CV and update Iteration 2 to the full 14-feature setup
- [ ] Ensure `random_forest_regression.pkl` and `bfs_municipality_and_tax_data.csv` are present at Space runtime
- [ ] Confirm the app starts successfully on Hugging Face Spaces
- [ ] Verify a few sample predictions after deployment
- [ ] Do one final spelling and formatting review
