import os
import pandas as pd
import streamlit as st
from pathlib import Path

import nbformat
from nbclient import NotebookClient

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Copper Prediction Dashboard", layout="wide")

st.title("🟠 Copper Prediction Dashboard")

# ----------------------------
# Paths
# ----------------------------
DATA_DIR = Path("data")

PREDICTIONS_PATH = DATA_DIR / "copper_model_predictions.csv"
LATEST_PATH = DATA_DIR / "copper_latest_prediction.csv"
SUMMARY_PATH = DATA_DIR / "copper_model_summary.csv"

# ----------------------------
# Notebook runner
# ----------------------------
def run_notebook(path):
    # open the notebook file
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # use the kernel metadata already stored in the notebook if possible
    client = NotebookClient(
        nb,
        timeout=600
    )

    client.execute()

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Controls")

fred_api_key = st.sidebar.text_input(
    "FRED API Key",
    type="password",
    help="Paste your FRED API key here."
)

run_button = st.sidebar.button("Refresh Data & Run Model")

# ----------------------------
# Session state
# ----------------------------
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False

# ----------------------------
# Run pipeline
# ----------------------------
if run_button:
    if not fred_api_key:
        st.error("Please enter your FRED API key.")
    else:
        with st.spinner("Running data + model notebooks..."):
            try:
                # pass API key to notebooks
                os.environ["FRED_API_KEY"] = fred_api_key

                # run notebooks
                run_notebook("data_scraper.ipynb")
                run_notebook("predictor.ipynb")

                st.session_state.data_loaded = True

                st.success("Model updated successfully!")

            except Exception as e:
                st.error(f"Error running notebooks: {e}")

# ----------------------------
# Load outputs
# ----------------------------
if not st.session_state.data_loaded:
    st.info("Enter your API key and click 'Refresh Data & Run Model'")
    st.stop()

if not PREDICTIONS_PATH.exists():
    st.error("Prediction file not found. Run the model first.")
    st.stop()

pred_df = pd.read_csv(PREDICTIONS_PATH)

if "Unnamed: 0" in pred_df.columns:
    pred_df = pred_df.rename(columns={"Unnamed: 0": "date"})

if "date" in pred_df.columns:
    pred_df["date"] = pd.to_datetime(pred_df["date"])

latest_df = pd.read_csv(LATEST_PATH)
summary_df = pd.read_csv(SUMMARY_PATH)

latest = latest_df.iloc[0]
summary = summary_df.iloc[0]

# ----------------------------
# Top metrics
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Date", str(pd.to_datetime(latest["date"]).date()))
col2.metric("Copper Price", f"{latest['current_copper_price']:.2f}")
col3.metric("Prob Up", f"{latest['predicted_prob_up']:.1%}")
col4.metric("Direction", latest["predicted_direction"])

st.divider()

# ----------------------------
# Model summary
# ----------------------------
st.subheader("Model Performance")

col1, col2 = st.columns(2)

col1.metric("Accuracy", f"{summary['accuracy']:.2f}")
col2.metric("ROC AUC", f"{summary['roc_auc']:.2f}")

st.divider()

# ----------------------------
# Price chart
# ----------------------------
st.subheader("Actual vs Predicted Next-Month Price")

if "date" in pred_df.columns:
    pred_df = pred_df.sort_values("date")

chart_df = pred_df.set_index("date")[[
    "actual_next_price",
    "pred_next_price_rule"
]]

st.line_chart(chart_df)

# ----------------------------
# Probability chart
# ----------------------------
st.subheader("Probability Copper Goes Up")

prob_df = pred_df.set_index("date")[["pred_prob_up"]]
st.line_chart(prob_df)

# ----------------------------
# Recent predictions table
# ----------------------------
st.subheader("Recent Predictions")

display_cols = [
    "date",
    "copper_price",
    "pred_up",
    "pred_prob_up",
    "pred_next_price_rule",
    "actual_next_price"
]

st.dataframe(
    pred_df[display_cols].sort_values("date", ascending=False).head(20),
    use_container_width=True
)

# ----------------------------
# Latest prediction
# ----------------------------
st.subheader("Latest Prediction")

st.write(f"**Current Price:** {latest['current_copper_price']:.2f}")
st.write(f"**Predicted Price (Next Month):** {latest['predicted_next_month_price']:.2f}")
st.write(f"**Probability Up:** {latest['predicted_prob_up']:.1%}")
st.write(f"**Direction:** {latest['predicted_direction']}")