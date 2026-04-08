# Credit Card CIBIL Score Prediction System

This project builds a complete credit score prediction workflow in Python with:

- Simulated user and monthly credit-behavior data
- Regression modeling for current CIBIL score prediction
- Time-series style forecasting for future score trend
- Explainable AI using SHAP when available
- Risky-user anomaly detection
- Visualizations and exported result tables

## Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `shap` (optional but recommended)

## Project Files

- `cibil_prediction_system.py`: Main end-to-end pipeline
- `streamlit_app.py`: Interactive Streamlit dashboard
- `outputs/plots/`: Saved charts
- `outputs/tables/`: Saved datasets, feature importance, metrics, and anomaly results
- `outputs/summary.txt`: Final summary of results

## How It Works

1. Simulates a dataset for users with demographics, credit behavior, utility and rent payment patterns.
2. Creates 12 months of time-series-style behavior data for each user.
3. Handles missing values, encoding, and scaling through a preprocessing pipeline.
4. Engineers important credit features such as:
   - credit utilization ratio
   - payment-to-income ratio
   - late payment frequency
   - spending, payment, utilization, and score trends
5. Trains a `RandomForestRegressor` to predict the current CIBIL score.
6. Trains a sequence-based forecasting model using the prior monthly behavior to estimate the next score point.
7. Uses SHAP for explainability if installed, otherwise falls back to model feature importance.
8. Flags risky users using `IsolationForest`.

## Run

```bash
python cibil_prediction_system.py
```

## Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard includes:

- model metrics
- global explainability
- risky-user anomaly monitoring
- plot gallery
- live score prediction for a new user

## Optional Install

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
```

## Output Metrics

The script prints and saves:

- RMSE
- R²
- Predicted score for an example new user
- Feature importance plots
- Sample user time-series trends
- Anomaly detection output

## Notes

- The dataset is simulated because real CIBIL data is typically private and regulated.
- SHAP is optional. If it is not installed, the project still runs with a fallback explanation path.
- The time-series component is implemented as a sequence-learning style forecast using 12 months of historical behavior.
