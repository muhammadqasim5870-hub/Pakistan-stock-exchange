
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "psx_decision_tree_model.joblib"
FEATURES_PATH = BASE_DIR / "feature_columns.json"
METRICS_PATH = BASE_DIR / "psx_model_metrics.json"

st.set_page_config(
    page_title="PSX Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Pakistan Stock Market Price Predictor")
st.write(
    "Enter the numerical market features to estimate a stock's "
    "closing price using a trained Decision Tree model."
)

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)

    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)

    return model, features, metrics


try:
    model, features, metrics = load_model()
except Exception as exc:
    st.error(f"Could not load the model or its files: {exc}")
    st.info(
        "Check that the model file, feature_columns.json, "
        "and app.py are in the same GitHub repository folder."
    )
    st.stop()

st.subheader("Enter stock-market features")

with st.form("prediction_form"):
    values = {}

    columns = st.columns(2)

    for i, feature in enumerate(features):
        with columns[i % 2]:
            values[feature] = st.number_input(
                feature.replace("_", " ").title(),
                value=0.0,
                step=1.0,
                format="%.4f"
            )

    submitted = st.form_submit_button(
        "Predict Closing Price",
        type="primary",
        use_container_width=True
    )

if submitted:
    input_df = pd.DataFrame([values], columns=features)

    try:
        prediction = float(model.predict(input_df)[0])

        st.success("Prediction completed!")
        st.metric(
            label="Predicted Closing Price",
            value=f"{prediction:,.4f}"
        )
        st.caption(
            "This is a model estimate, not a guaranteed future market price."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.subheader("📊 Model Evaluation")

if metrics:
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "R² Score",
        f"{metrics['R2_score']:.4f}"
        if "R2_score" in metrics else "N/A"
    )
    col2.metric(
        "MAE",
        f"{metrics['MAE']:,.4f}"
        if "MAE" in metrics else "N/A"
    )
    col3.metric(
        "RMSE",
        f"{metrics['RMSE']:,.4f}"
        if "RMSE" in metrics else "N/A"
    )
    col4.metric(
        "Test Records",
        f"{metrics['testing_records']:,}"
        if "testing_records" in metrics else "N/A"
    )

    st.caption(
        "Metrics are from the saved evaluation results, "
        "not recalculated from the user's input."
    )
else:
    st.info("No saved evaluation metrics were found.")

st.divider()
st.caption(
    "Educational machine-learning project | "
    "Pakistan Stock Exchange historical data (2017–2025)"
)
