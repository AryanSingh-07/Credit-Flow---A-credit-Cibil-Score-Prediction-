import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(os.path.join("outputs", ".mplconfig")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


OUTPUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")


@dataclass
class Config:
    n_users: int = 1000
    months: int = 12
    random_state: int = 42
    test_size: float = 0.2


def ensure_directories() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def linear_trend(values: np.ndarray) -> float:
    x_axis = np.arange(len(values))
    return float(np.polyfit(x_axis, values, 1)[0])


def simulate_credit_dataset(config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.random_state)
    occupations = np.array(
        ["Salaried", "Self-Employed", "Student", "Business", "Retired", "Freelancer"]
    )

    user_rows: List[Dict] = []
    monthly_rows: List[Dict] = []

    for user_id in range(1, config.n_users + 1):
        age = int(rng.integers(21, 60))
        occupation = rng.choice(
            occupations, p=[0.42, 0.18, 0.10, 0.14, 0.06, 0.10]
        )

        base_income = {
            "Salaried": rng.normal(75000, 18000),
            "Self-Employed": rng.normal(90000, 30000),
            "Student": rng.normal(25000, 8000),
            "Business": rng.normal(110000, 35000),
            "Retired": rng.normal(50000, 12000),
            "Freelancer": rng.normal(65000, 22000),
        }[occupation]
        income = float(np.clip(base_income, 15000, 300000))

        credit_limit = float(np.clip(income * rng.uniform(0.25, 0.8), 10000, 250000))
        utility_payment_ratio = float(np.clip(rng.normal(0.9, 0.1), 0.45, 1.05))
        rent_payment_ratio = float(np.clip(rng.normal(0.88, 0.14), 0.30, 1.05))
        existing_loans = int(rng.integers(0, 5))

        risk_bias = (
            0.5 * (occupation == "Student")
            + 0.25 * (occupation == "Freelancer")
            - 0.15 * (occupation == "Retired")
        )
        late_payment_lambda = np.clip(0.8 + risk_bias + (70000 - income) / 100000, 0.1, 3.5)
        missed_base = int(rng.poisson(late_payment_lambda))

        spend_level = income * rng.uniform(0.12, 0.45)
        payment_strength = rng.uniform(0.75, 1.15)

        utilization_list = []
        payment_ratio_list = []
        late_payment_list = []
        repayment_quality_list = []
        score_series = []

        running_score = 720 + rng.normal(0, 20)

        for month in range(1, config.months + 1):
            seasonal = 1 + 0.12 * np.sin((2 * np.pi * month) / 12)
            monthly_spend = float(
                np.clip(rng.normal(spend_level * seasonal, income * 0.05), 1000, credit_limit * 1.1)
            )
            utilization = float(np.clip(monthly_spend / max(credit_limit, 1), 0.02, 1.2))

            late_prob = sigmoid(
                np.array(
                    [
                        -1.1
                        + 2.2 * (utilization - 0.45)
                        + 0.6 * risk_bias
                        + 0.5 * (utility_payment_ratio < 0.8)
                        + 0.4 * (rent_payment_ratio < 0.8)
                    ]
                )
            )[0]
            late_payment = int(rng.random() < late_prob)
            late_payment += int(rng.random() < max(0.0, late_prob - 0.45))

            payment_ratio = float(
                np.clip(
                    payment_strength - 0.25 * utilization - 0.08 * late_payment + rng.normal(0, 0.08),
                    0.35,
                    1.25,
                )
            )
            monthly_payment = float(np.clip(monthly_spend * payment_ratio, 500, income * 0.65))

            repayment_quality = float(np.clip(payment_ratio - 0.1 * late_payment, 0.0, 1.2))

            score_change = (
                9.0 * (1 - utilization)
                + 18.0 * repayment_quality
                - 22.0 * late_payment
                + 11.0 * (utility_payment_ratio - 0.75)
                + 9.0 * (rent_payment_ratio - 0.75)
                - 3.5 * existing_loans
                + rng.normal(0, 9)
            )
            running_score = float(np.clip(running_score + score_change / 6.0, 300, 900))

            utilization_list.append(utilization)
            payment_ratio_list.append(monthly_payment / max(income, 1))
            late_payment_list.append(late_payment)
            repayment_quality_list.append(repayment_quality)
            score_series.append(running_score)

            monthly_rows.append(
                {
                    "user_id": user_id,
                    "month": month,
                    "monthly_spend": monthly_spend,
                    "monthly_payment": monthly_payment,
                    "monthly_utilization": utilization,
                    "late_payments_month": late_payment,
                    "repayment_quality": repayment_quality,
                    "score_month": running_score,
                }
            )

        utilization_arr = np.array(utilization_list)
        payment_income_arr = np.array(payment_ratio_list)
        late_arr = np.array(late_payment_list)
        score_arr = np.array(score_series)

        current_score = float(np.clip(score_arr[-1] + rng.normal(0, 8), 300, 900))

        user_rows.append(
            {
                "user_id": user_id,
                "age": age,
                "income": income,
                "occupation": occupation,
                "credit_limit": credit_limit,
                "utility_payment_ratio": utility_payment_ratio,
                "rent_payment_ratio": rent_payment_ratio,
                "existing_loans": existing_loans,
                "avg_monthly_spend": float(np.mean([row["monthly_spend"] for row in monthly_rows if row["user_id"] == user_id])),
                "avg_monthly_payment": float(np.mean([row["monthly_payment"] for row in monthly_rows if row["user_id"] == user_id])),
                "credit_utilization_ratio": float(np.mean(utilization_arr)),
                "payment_to_income_ratio": float(np.mean(payment_income_arr)),
                "late_payment_frequency": float(np.sum(late_arr) / config.months),
                "repayment_history_score": float(np.mean(repayment_quality_list)),
                "spending_trend": linear_trend(
                    np.array([row["monthly_spend"] for row in monthly_rows if row["user_id"] == user_id])
                ),
                "payment_trend": linear_trend(
                    np.array([row["monthly_payment"] for row in monthly_rows if row["user_id"] == user_id])
                ),
                "score_trend_last_12m": linear_trend(score_arr),
                "avg_score_last_12m": float(np.mean(score_arr)),
                "score_std_last_12m": float(np.std(score_arr)),
                "cibil_score": current_score,
                "future_score_1m": float(np.clip(current_score + score_arr[-1] - score_arr[-2] + rng.normal(0, 10), 300, 900)),
            }
        )

    user_df = pd.DataFrame(user_rows)
    monthly_df = pd.DataFrame(monthly_rows)

    user_df = inject_missingness(user_df, rng)
    monthly_df = inject_missingness(monthly_df, rng, fraction=0.01)
    return user_df, monthly_df


def inject_missingness(df: pd.DataFrame, rng: np.random.Generator, fraction: float = 0.03) -> pd.DataFrame:
    result = df.copy()
    protected_columns = {"user_id", "month", "cibil_score", "future_score_1m"}
    candidate_columns = [col for col in result.columns if col not in protected_columns]
    total_cells = int(len(result) * len(candidate_columns) * fraction)
    if total_cells <= 0:
        return result

    for _ in range(total_cells):
        row = int(rng.integers(0, len(result)))
        col = rng.choice(candidate_columns)
        result.at[row, col] = np.nan
    return result


def aggregate_monthly_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    grouped = monthly_df.sort_values(["user_id", "month"]).groupby("user_id")
    feature_rows = []

    for user_id, group in grouped:
        spend = group["monthly_spend"].interpolate(limit_direction="both")
        payment = group["monthly_payment"].interpolate(limit_direction="both")
        score = group["score_month"].interpolate(limit_direction="both")
        utilization = group["monthly_utilization"].interpolate(limit_direction="both")
        late = group["late_payments_month"].fillna(0)

        feature_rows.append(
            {
                "user_id": user_id,
                "monthly_spend_mean": float(spend.mean()),
                "monthly_payment_mean": float(payment.mean()),
                "monthly_spend_trend": linear_trend(spend.to_numpy()),
                "monthly_payment_trend": linear_trend(payment.to_numpy()),
                "monthly_score_trend": linear_trend(score.to_numpy()),
                "monthly_utilization_mean": float(utilization.mean()),
                "monthly_utilization_trend": linear_trend(utilization.to_numpy()),
                "monthly_late_payment_total": float(late.sum()),
                "monthly_late_payment_trend": linear_trend(late.to_numpy()),
            }
        )

    return pd.DataFrame(feature_rows)


def prepare_modeling_data(user_df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
    monthly_features = aggregate_monthly_features(monthly_df)
    merged = user_df.merge(monthly_features, on="user_id", how="left")
    return merged


def build_preprocessor(feature_df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = [col for col in feature_df.columns if col not in categorical_cols]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def train_regression_model(df: pd.DataFrame, config: Config):
    feature_cols = [col for col in df.columns if col not in {"user_id", "cibil_score", "future_score_1m"}]
    X = df[feature_cols]
    y = df["cibil_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )

    preprocessor, _, _ = build_preprocessor(X)
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=config.random_state,
        n_jobs=1,
    )
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = float(r2_score(y_test, predictions))

    return pipeline, X_train, X_test, y_train, y_test, predictions, {"rmse": rmse, "r2": r2}


def build_sequence_dataset(monthly_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_seq = []
    y_seq = []

    for _, group in monthly_df.sort_values(["user_id", "month"]).groupby("user_id"):
        group = group.copy()
        for col in ["monthly_spend", "monthly_payment", "monthly_utilization", "late_payments_month", "score_month"]:
            if group[col].isna().all():
                group[col] = 0
            else:
                group[col] = group[col].interpolate(limit_direction="both").fillna(group[col].median())

        sequence = group[
            ["monthly_spend", "monthly_payment", "monthly_utilization", "late_payments_month", "score_month"]
        ].to_numpy()
        X_seq.append(sequence[:-1])
        y_seq.append(sequence[-1, -1])

    return np.array(X_seq), np.array(y_seq)


def train_sequence_model(monthly_df: pd.DataFrame, config: Config):
    X_seq, y_seq = build_sequence_dataset(monthly_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=config.test_size, random_state=config.random_state
    )

    model_name = "Sequence Random Forest"
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    sequence_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=10,
        random_state=config.random_state,
        n_jobs=1,
    )
    sequence_model.fit(X_train_scaled, y_train)
    preds = sequence_model.predict(X_test_scaled)

    return {
        "model_name": model_name,
        "model": sequence_model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": preds,
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
    }


def explain_with_shap(pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    sample_train = X_train.sample(min(200, len(X_train)), random_state=42)
    sample_test = X_test.sample(min(50, len(X_test)), random_state=42)

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed_train = preprocessor.transform(sample_train)
    transformed_test = preprocessor.transform(sample_test)
    feature_names = preprocessor.get_feature_names_out()

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": mean_abs_shap}
        ).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        top_shap = importance_df.head(12).sort_values("importance")
        plt.barh(top_shap["feature"], top_shap["importance"], color="teal")
        plt.title("Top SHAP Feature Importances")
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "shap_feature_importance.png"))
        plt.close()
        return importance_df

    raw_importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": raw_importance}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    top_features = importance_df.head(12).sort_values("importance")
    plt.barh(top_features["feature"], top_features["importance"], color="darkorange")
    plt.title("Fallback Feature Importances (SHAP not installed)")
    plt.xlabel("Model importance")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_feature_importance.png"))
    plt.close()
    return importance_df


def explain_single_user(
    pipeline: Pipeline, data: pd.DataFrame, user_row: pd.Series
) -> Tuple[float, pd.DataFrame]:
    features = data.drop(columns=["user_id", "cibil_score", "future_score_1m"])
    feature_row = user_row[features.columns].to_frame().T
    prediction = float(pipeline.predict(feature_row)[0])

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(feature_row)
    feature_names = preprocessor.get_feature_names_out()

    if SHAP_AVAILABLE:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)[0]
        contribution_df = pd.DataFrame(
            {"feature": feature_names, "contribution": shap_values}
        ).sort_values("contribution", key=np.abs, ascending=False)
    else:
        dense_row = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
        contributions = dense_row * model.feature_importances_
        contribution_df = pd.DataFrame(
            {"feature": feature_names, "contribution": contributions}
        ).sort_values("contribution", key=np.abs, ascending=False)

    return prediction, contribution_df


def run_anomaly_detection(df: pd.DataFrame) -> pd.DataFrame:
    anomaly_features = df[
        [
            "credit_utilization_ratio",
            "payment_to_income_ratio",
            "late_payment_frequency",
            "utility_payment_ratio",
            "rent_payment_ratio",
            "score_trend_last_12m",
        ]
    ].copy()

    anomaly_features = anomaly_features.fillna(anomaly_features.median(numeric_only=True))
    detector = IsolationForest(contamination=0.08, random_state=42)
    labels = detector.fit_predict(anomaly_features)
    scores = detector.decision_function(anomaly_features)

    result = df[["user_id", "cibil_score"]].copy()
    result["risk_flag"] = np.where(labels == -1, "Risky", "Normal")
    result["anomaly_score"] = scores
    return result.sort_values("anomaly_score")


def create_visualizations(
    df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    evaluation: Dict[str, float],
    sequence_results: Dict[str, float],
) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["cibil_score"], kde=True, color="slateblue")
    plt.title("Distribution of Simulated CIBIL Scores")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "cibil_score_distribution.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="credit_utilization_ratio",
        y="cibil_score",
        hue="late_payment_frequency",
        palette="viridis",
        alpha=0.7,
    )
    plt.title("CIBIL Score vs Credit Utilization")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "score_vs_utilization.png"))
    plt.close()

    selected_ids = monthly_df["user_id"].drop_duplicates().sample(5, random_state=42)
    plt.figure(figsize=(10, 6))
    for user_id in selected_ids:
        user_series = monthly_df[monthly_df["user_id"] == user_id].sort_values("month")
        plt.plot(user_series["month"], user_series["score_month"], marker="o", label=f"User {user_id}")
    plt.title("12-Month Credit Score Trend for Sample Users")
    plt.xlabel("Month")
    plt.ylabel("Credit Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "sample_user_score_trends.png"))
    plt.close()

    metrics_df = pd.DataFrame(
        [
            {"model": "Random Forest Regression", "RMSE": evaluation["rmse"], "R2": evaluation["r2"]},
            {"model": sequence_results["model_name"], "RMSE": sequence_results["rmse"], "R2": sequence_results["r2"]},
        ]
    )
    metrics_df.to_csv(os.path.join(TABLES_DIR, "model_metrics.csv"), index=False)
    importance_df.head(20).to_csv(
        os.path.join(TABLES_DIR, "feature_importance.csv"), index=False
    )


def create_example_user() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "age": 31,
                "income": 85000,
                "occupation": "Salaried",
                "credit_limit": 45000,
                "utility_payment_ratio": 0.96,
                "rent_payment_ratio": 0.92,
                "existing_loans": 1,
                "avg_monthly_spend": 21000,
                "avg_monthly_payment": 19500,
                "credit_utilization_ratio": 0.47,
                "payment_to_income_ratio": 0.23,
                "late_payment_frequency": 0.08,
                "repayment_history_score": 0.89,
                "spending_trend": 250,
                "payment_trend": 180,
                "score_trend_last_12m": 3.6,
                "avg_score_last_12m": 742,
                "score_std_last_12m": 14,
                "monthly_spend_mean": 21000,
                "monthly_payment_mean": 19500,
                "monthly_spend_trend": 250,
                "monthly_payment_trend": 180,
                "monthly_score_trend": 3.6,
                "monthly_utilization_mean": 0.47,
                "monthly_utilization_trend": -0.01,
                "monthly_late_payment_total": 1,
                "monthly_late_payment_trend": -0.02,
            }
        ]
    )


def summarize_single_prediction(contribution_df: pd.DataFrame) -> str:
    top_positive = contribution_df.head(3)
    lines = []
    for _, row in top_positive.iterrows():
        direction = "increases" if row["contribution"] >= 0 else "decreases"
        lines.append(f"{row['feature']} {direction} the score")
    return "; ".join(lines)


def run_full_pipeline(config: Config | None = None, save_outputs: bool = True) -> Dict[str, object]:
    ensure_directories()
    config = config or Config()

    print("1. Generating simulated dataset...")
    user_df, monthly_df = simulate_credit_dataset(config)
    if save_outputs:
        user_df.to_csv(os.path.join(TABLES_DIR, "simulated_user_data.csv"), index=False)
        monthly_df.to_csv(os.path.join(TABLES_DIR, "simulated_monthly_data.csv"), index=False)

    print("2. Preparing features...")
    modeling_df = prepare_modeling_data(user_df, monthly_df)
    if save_outputs:
        modeling_df.to_csv(os.path.join(TABLES_DIR, "modeling_dataset.csv"), index=False)

    print("3. Training regression model for CIBIL score...")
    (
        regression_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
        reg_predictions,
        evaluation,
    ) = train_regression_model(modeling_df, config)

    print("4. Training sequence model for future score trend...")
    sequence_results = train_sequence_model(monthly_df, config)

    print("5. Running explainability analysis...")
    importance_df = explain_with_shap(regression_pipeline, X_train, X_test)

    print("6. Detecting risky users...")
    anomaly_df = run_anomaly_detection(modeling_df)
    if save_outputs:
        anomaly_df.to_csv(os.path.join(TABLES_DIR, "anomaly_detection_results.csv"), index=False)

    print("7. Creating visualizations...")
    if save_outputs:
        create_visualizations(modeling_df, monthly_df, importance_df, evaluation, sequence_results)

    print("8. Predicting an example new user...")
    example_user = create_example_user()
    example_prediction = float(regression_pipeline.predict(example_user)[0])
    _, contribution_df = explain_single_user(
        regression_pipeline,
        modeling_df,
        pd.concat([pd.Series({"user_id": -1, "cibil_score": np.nan, "future_score_1m": np.nan}), example_user.iloc[0]]),
    )
    if save_outputs:
        contribution_df.head(10).to_csv(
            os.path.join(TABLES_DIR, "example_user_explanation.csv"), index=False
        )

    summary_text = summarize_single_prediction(contribution_df)

    if save_outputs:
        with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as summary_file:
            summary_file.write("Credit Card CIBIL Score Prediction System Summary\n")
            summary_file.write("=" * 55 + "\n")
            summary_file.write(f"Regression RMSE: {evaluation['rmse']:.2f}\n")
            summary_file.write(f"Regression R^2: {evaluation['r2']:.4f}\n")
            summary_file.write(f"Sequence Model ({sequence_results['model_name']}) RMSE: {sequence_results['rmse']:.2f}\n")
            summary_file.write(f"Sequence Model ({sequence_results['model_name']}) R^2: {sequence_results['r2']:.4f}\n")
            summary_file.write(f"Example new user predicted CIBIL score: {example_prediction:.2f}\n")
            summary_file.write(f"Explanation summary: {summary_text}\n")
            summary_file.write("Most risky users are available in outputs/tables/anomaly_detection_results.csv\n")

    return {
        "config": config,
        "user_df": user_df,
        "monthly_df": monthly_df,
        "modeling_df": modeling_df,
        "regression_pipeline": regression_pipeline,
        "evaluation": evaluation,
        "sequence_results": sequence_results,
        "importance_df": importance_df,
        "anomaly_df": anomaly_df,
        "example_user": example_user,
        "example_prediction": example_prediction,
        "contribution_df": contribution_df,
        "summary_text": summary_text,
    }


def main() -> None:
    results = run_full_pipeline(save_outputs=True)

    print("\nProject complete.")
    print(f"Regression Model RMSE: {results['evaluation']['rmse']:.2f}")
    print(f"Regression Model R^2: {results['evaluation']['r2']:.4f}")
    print(f"Sequence Model ({results['sequence_results']['model_name']}) RMSE: {results['sequence_results']['rmse']:.2f}")
    print(f"Sequence Model ({results['sequence_results']['model_name']}) R^2: {results['sequence_results']['r2']:.4f}")
    print(f"Example new user predicted CIBIL score: {results['example_prediction']:.2f}")
    print("Top explanation drivers:")
    print(results["contribution_df"].head(5).to_string(index=False))
    print(f"\nArtifacts saved under: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
