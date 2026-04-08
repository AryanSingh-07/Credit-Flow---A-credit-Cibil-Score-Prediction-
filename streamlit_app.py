import altair as alt
import pandas as pd
import streamlit as st

from cibil_prediction_system import Config, explain_single_user, run_full_pipeline


st.set_page_config(
    page_title="CIBIL Score Prediction Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_pipeline_results(n_users: int, random_state: int):
    config = Config(n_users=n_users, random_state=random_state)
    return run_full_pipeline(config=config, save_outputs=False)


def score_band(score: float) -> str:
    if score >= 750:
        return "Excellent"
    if score >= 700:
        return "Good"
    if score >= 650:
        return "Fair"
    return "Risky"


def format_modeling_view(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()
    view["score_band"] = view["cibil_score"].apply(score_band)
    return view


def build_sidebar():
    st.sidebar.header("Dataset Controls")
    n_users = st.sidebar.slider("Number of Users", min_value=1000, max_value=20000, value=8000, step=1000)
    random_state = st.sidebar.number_input("Random Seed", min_value=1, max_value=9999, value=42, step=1)
    st.sidebar.caption("Larger datasets take longer because the models are retrained on demand.")
    return n_users, random_state


def render_overview(results, filtered_df: pd.DataFrame, monthly_filtered: pd.DataFrame):
    st.subheader("Interactive Portfolio View")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Users in View", f"{len(filtered_df):,}")
    col2.metric("Avg CIBIL Score", f"{filtered_df['cibil_score'].mean():.1f}")
    col3.metric("Avg Utilization", f"{filtered_df['credit_utilization_ratio'].mean():.2f}")
    col4.metric("Risky Users", f"{(filtered_df['risk_flag'] == 'Risky').sum():,}")

    hist = (
        alt.Chart(filtered_df)
        .mark_bar(color="#0f766e")
        .encode(
            alt.X("cibil_score:Q", bin=alt.Bin(maxbins=30), title="CIBIL Score"),
            alt.Y("count():Q", title="Users"),
            tooltip=["count():Q"],
        )
        .properties(height=320, title="Score Distribution")
    )

    scatter = (
        alt.Chart(filtered_df)
        .mark_circle(size=55, opacity=0.6)
        .encode(
            x=alt.X("credit_utilization_ratio:Q", title="Credit Utilization Ratio"),
            y=alt.Y("cibil_score:Q", title="CIBIL Score"),
            color=alt.Color("late_payment_frequency:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=["user_id", "cibil_score", "income", "late_payment_frequency", "occupation"],
        )
        .interactive()
        .properties(height=320, title="Score vs Utilization")
    )

    st.altair_chart(hist | scatter, use_container_width=True)

    trend_base = (
        monthly_filtered.groupby("month", as_index=False)
        .agg(
            avg_spend=("monthly_spend", "mean"),
            avg_payment=("monthly_payment", "mean"),
            avg_score=("score_month", "mean"),
        )
        .melt("month", var_name="metric", value_name="value")
    )
    trend_chart = (
        alt.Chart(trend_base)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("value:Q", title="Average Value"),
            color=alt.Color("metric:N", title="Metric"),
            tooltip=["month", "metric", "value"],
        )
        .interactive()
        .properties(height=350, title="Average Monthly Behavior Across Selected Users")
    )
    st.altair_chart(trend_chart, use_container_width=True)


def render_user_drilldown(filtered_df: pd.DataFrame, monthly_df: pd.DataFrame):
    st.subheader("User Drill-Down")
    user_ids = filtered_df["user_id"].sort_values().tolist()
    selected_user = st.selectbox("Select User ID", user_ids, index=0)

    user_row = filtered_df[filtered_df["user_id"] == selected_user].iloc[0]
    user_monthly = monthly_df[monthly_df["user_id"] == selected_user].sort_values("month")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Score", f"{user_row['cibil_score']:.1f}")
    col2.metric("Income", f"{user_row['income']:.0f}")
    col3.metric("Utilization", f"{user_row['credit_utilization_ratio']:.2f}")
    col4.metric("Risk Flag", user_row["risk_flag"])

    profile_cols = [
        "occupation",
        "age",
        "utility_payment_ratio",
        "rent_payment_ratio",
        "late_payment_frequency",
        "payment_to_income_ratio",
        "score_trend_last_12m",
    ]
    st.dataframe(user_row[profile_cols].to_frame("value"), use_container_width=True)

    line_data = user_monthly.melt(
        id_vars=["month"],
        value_vars=["monthly_spend", "monthly_payment", "score_month"],
        var_name="metric",
        value_name="value",
    )
    line_chart = (
        alt.Chart(line_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:O", title="Month"),
            y=alt.Y("value:Q", title="Value"),
            color="metric:N",
            tooltip=["month", "metric", "value"],
        )
        .interactive()
        .properties(height=360, title=f"12-Month Behavior for User {selected_user}")
    )
    st.altair_chart(line_chart, use_container_width=True)


def build_prediction_input(defaults: pd.Series) -> pd.DataFrame:
    with st.form("prediction_form"):
        st.subheader("Live Prediction")
        top1, top2, top3 = st.columns(3)

        with top1:
            age = st.number_input("Age", 18, 80, int(defaults["age"]))
            income = st.number_input("Monthly Income", 10000.0, 500000.0, float(defaults["income"]), 5000.0)
            occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Student", "Business", "Retired", "Freelancer"])
            existing_loans = st.number_input("Existing Loans", 0, 10, int(defaults["existing_loans"]))

        with top2:
            credit_limit = st.number_input("Credit Limit", 5000.0, 500000.0, float(defaults["credit_limit"]), 5000.0)
            utility_payment_ratio = st.slider("Utility Payment Ratio", 0.0, 1.2, float(defaults["utility_payment_ratio"]), 0.01)
            rent_payment_ratio = st.slider("Rent Payment Ratio", 0.0, 1.2, float(defaults["rent_payment_ratio"]), 0.01)
            credit_utilization_ratio = st.slider("Credit Utilization Ratio", 0.0, 1.2, float(defaults["credit_utilization_ratio"]), 0.01)

        with top3:
            payment_to_income_ratio = st.slider("Payment to Income Ratio", 0.0, 1.2, float(defaults["payment_to_income_ratio"]), 0.01)
            late_payment_frequency = st.slider("Late Payment Frequency", 0.0, 2.0, float(defaults["late_payment_frequency"]), 0.01)
            repayment_history_score = st.slider("Repayment History Score", 0.0, 1.2, float(defaults["repayment_history_score"]), 0.01)
            avg_score_last_12m = st.number_input("Avg Score Last 12 Months", 300.0, 900.0, float(defaults["avg_score_last_12m"]), 1.0)

        mid1, mid2, mid3 = st.columns(3)
        with mid1:
            avg_monthly_spend = st.number_input("Avg Monthly Spend", 500.0, 300000.0, float(defaults["avg_monthly_spend"]), 1000.0)
            avg_monthly_payment = st.number_input("Avg Monthly Payment", 500.0, 300000.0, float(defaults["avg_monthly_payment"]), 1000.0)
            score_std_last_12m = st.number_input("Score Std Last 12 Months", 0.0, 120.0, float(defaults["score_std_last_12m"]), 0.5)

        with mid2:
            spending_trend = st.number_input("Spending Trend", value=float(defaults["spending_trend"]), step=10.0)
            payment_trend = st.number_input("Payment Trend", value=float(defaults["payment_trend"]), step=10.0)
            score_trend_last_12m = st.number_input("Score Trend Last 12 Months", value=float(defaults["score_trend_last_12m"]), step=0.1)

        with mid3:
            monthly_spend_mean = st.number_input("Monthly Spend Mean", 500.0, 300000.0, float(defaults["monthly_spend_mean"]), 1000.0)
            monthly_payment_mean = st.number_input("Monthly Payment Mean", 500.0, 300000.0, float(defaults["monthly_payment_mean"]), 1000.0)
            monthly_utilization_mean = st.slider("Monthly Utilization Mean", 0.0, 1.2, float(defaults["monthly_utilization_mean"]), 0.01)

        low1, low2 = st.columns(2)
        with low1:
            monthly_spend_trend = st.number_input("Monthly Spend Trend", value=float(defaults["monthly_spend_trend"]), step=10.0)
            monthly_payment_trend = st.number_input("Monthly Payment Trend", value=float(defaults["monthly_payment_trend"]), step=10.0)
            monthly_score_trend = st.number_input("Monthly Score Trend", value=float(defaults["monthly_score_trend"]), step=0.1)

        with low2:
            monthly_utilization_trend = st.number_input("Monthly Utilization Trend", value=float(defaults["monthly_utilization_trend"]), step=0.01, format="%.2f")
            monthly_late_payment_total = st.number_input("Monthly Late Payment Total", 0.0, 24.0, float(defaults["monthly_late_payment_total"]), 1.0)
            monthly_late_payment_trend = st.number_input("Monthly Late Payment Trend", value=float(defaults["monthly_late_payment_trend"]), step=0.01, format="%.2f")

        submitted = st.form_submit_button("Predict Score")

    features = pd.DataFrame(
        [
            {
                "age": age,
                "income": income,
                "occupation": occupation,
                "credit_limit": credit_limit,
                "utility_payment_ratio": utility_payment_ratio,
                "rent_payment_ratio": rent_payment_ratio,
                "existing_loans": existing_loans,
                "avg_monthly_spend": avg_monthly_spend,
                "avg_monthly_payment": avg_monthly_payment,
                "credit_utilization_ratio": credit_utilization_ratio,
                "payment_to_income_ratio": payment_to_income_ratio,
                "late_payment_frequency": late_payment_frequency,
                "repayment_history_score": repayment_history_score,
                "spending_trend": spending_trend,
                "payment_trend": payment_trend,
                "score_trend_last_12m": score_trend_last_12m,
                "avg_score_last_12m": avg_score_last_12m,
                "score_std_last_12m": score_std_last_12m,
                "monthly_spend_mean": monthly_spend_mean,
                "monthly_payment_mean": monthly_payment_mean,
                "monthly_spend_trend": monthly_spend_trend,
                "monthly_payment_trend": monthly_payment_trend,
                "monthly_score_trend": monthly_score_trend,
                "monthly_utilization_mean": monthly_utilization_mean,
                "monthly_utilization_trend": monthly_utilization_trend,
                "monthly_late_payment_total": monthly_late_payment_total,
                "monthly_late_payment_trend": monthly_late_payment_trend,
            }
        ]
    )
    return features, submitted


def render_prediction(results):
    st.subheader("Live Predictor With Explainability")
    default_user = results["modeling_df"].sort_values("cibil_score", ascending=False).iloc[0]
    input_df, submitted = build_prediction_input(default_user)

    if submitted:
        predicted_score = float(results["regression_pipeline"].predict(input_df)[0])
        band = score_band(predicted_score)
        explain_row = pd.concat(
            [
                pd.Series({"user_id": -1, "cibil_score": pd.NA, "future_score_1m": pd.NA}),
                input_df.iloc[0],
            ]
        )
        _, contribution_df = explain_single_user(results["regression_pipeline"], results["modeling_df"], explain_row)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Predicted Score", f"{predicted_score:.2f}")
            st.metric("Score Band", band)
        with col2:
            top_contrib = contribution_df.head(12).copy()
            top_contrib["direction"] = top_contrib["contribution"].apply(lambda x: "Positive" if x >= 0 else "Negative")
            bar = (
                alt.Chart(top_contrib)
                .mark_bar()
                .encode(
                    x=alt.X("contribution:Q", title="Impact on Prediction"),
                    y=alt.Y("feature:N", sort="-x", title="Feature"),
                    color=alt.Color("direction:N", scale=alt.Scale(domain=["Positive", "Negative"], range=["#0f766e", "#b91c1c"])),
                    tooltip=["feature", "contribution", "direction"],
                )
                .properties(height=420, title="Top Prediction Drivers")
            )
            st.altair_chart(bar, use_container_width=True)


def render_explainability(results):
    st.subheader("Global Model Explainability")
    top_features = results["importance_df"].head(20).copy()
    importance_chart = (
        alt.Chart(top_features)
        .mark_bar(color="#0f766e")
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title="Feature"),
            tooltip=["feature", "importance"],
        )
        .properties(height=500, title="Top Global Drivers of CIBIL Score")
    )
    st.altair_chart(importance_chart, use_container_width=True)
    st.dataframe(top_features, use_container_width=True)


def render_risk_monitoring(filtered_df: pd.DataFrame):
    st.subheader("Risk Monitoring and Anomaly Detection")
    risky_users = filtered_df[filtered_df["risk_flag"] == "Risky"].sort_values("anomaly_score")
    st.metric("Flagged Risky Users", f"{len(risky_users):,}")
    st.dataframe(
        risky_users[
            [
                "user_id",
                "cibil_score",
                "income",
                "credit_utilization_ratio",
                "late_payment_frequency",
                "anomaly_score",
                "occupation",
            ]
        ].head(100),
        use_container_width=True,
    )

    risk_chart = (
        alt.Chart(risky_users.head(200))
        .mark_circle(size=70, opacity=0.7)
        .encode(
            x=alt.X("late_payment_frequency:Q", title="Late Payment Frequency"),
            y=alt.Y("cibil_score:Q", title="CIBIL Score"),
            color=alt.Color("anomaly_score:Q", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=["user_id", "cibil_score", "anomaly_score", "income"],
        )
        .interactive()
        .properties(height=360, title="Flagged Users: Score vs Late Payments")
    )
    st.altair_chart(risk_chart, use_container_width=True)


def render_data_explorer(filtered_df: pd.DataFrame, monthly_filtered: pd.DataFrame):
    st.subheader("Data Explorer")
    st.write("User-level modeling dataset")
    st.dataframe(filtered_df.head(200), use_container_width=True)
    st.download_button(
        "Download Filtered User Data",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_user_data.csv",
        mime="text/csv",
    )

    st.write("Monthly time-series dataset")
    st.dataframe(monthly_filtered.head(300), use_container_width=True)
    st.download_button(
        "Download Filtered Monthly Data",
        monthly_filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_monthly_data.csv",
        mime="text/csv",
    )


def main():
    st.title("Credit Card CIBIL Score Prediction Dashboard")
    st.caption("Interactive portfolio simulation, explainable AI, anomaly detection, and live credit score prediction")

    n_users, random_state = build_sidebar()
    with st.spinner(f"Training models on {n_users:,} users..."):
        results = load_pipeline_results(n_users, random_state)

    modeling_df = format_modeling_view(results["modeling_df"]).merge(
        results["anomaly_df"][["user_id", "risk_flag", "anomaly_score"]],
        on="user_id",
        how="left",
    )
    monthly_df = results["monthly_df"]

    st.sidebar.header("Filters")
    occupations = st.sidebar.multiselect(
        "Occupation",
        options=sorted(modeling_df["occupation"].dropna().unique().tolist()),
        default=sorted(modeling_df["occupation"].dropna().unique().tolist()),
    )
    score_range = st.sidebar.slider(
        "CIBIL Score Range",
        min_value=int(modeling_df["cibil_score"].min()),
        max_value=int(modeling_df["cibil_score"].max()),
        value=(int(modeling_df["cibil_score"].min()), int(modeling_df["cibil_score"].max())),
    )
    risk_filter = st.sidebar.multiselect(
        "Risk Flag",
        options=["Normal", "Risky"],
        default=["Normal", "Risky"],
    )

    filtered_df = modeling_df[
        modeling_df["occupation"].isin(occupations)
        & modeling_df["risk_flag"].isin(risk_filter)
        & modeling_df["cibil_score"].between(score_range[0], score_range[1])
    ].copy()
    monthly_filtered = monthly_df[monthly_df["user_id"].isin(filtered_df["user_id"])].copy()

    top_metrics = st.columns(5)
    top_metrics[0].metric("Regression RMSE", f"{results['evaluation']['rmse']:.2f}")
    top_metrics[1].metric("Regression R²", f"{results['evaluation']['r2']:.4f}")
    top_metrics[2].metric("Sequence RMSE", f"{results['sequence_results']['rmse']:.2f}")
    top_metrics[3].metric("Sequence R²", f"{results['sequence_results']['r2']:.4f}")
    top_metrics[4].metric("Dataset Rows", f"{len(modeling_df):,}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Overview", "User Drilldown", "Prediction", "Explainability", "Risk + Data"]
    )

    with tab1:
        render_overview(results, filtered_df, monthly_filtered)

    with tab2:
        render_user_drilldown(filtered_df, monthly_df)

    with tab3:
        render_prediction(results)

    with tab4:
        render_explainability(results)

    with tab5:
        render_risk_monitoring(filtered_df)
        st.divider()
        render_data_explorer(filtered_df, monthly_filtered)


if __name__ == "__main__":
    main()
