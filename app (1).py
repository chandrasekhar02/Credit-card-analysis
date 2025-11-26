import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests

st.set_page_config(page_title="Credit Card Delinquency Watch", layout="centered")

MODEL_LOCAL_PATH = "rf_model.joblib"
SCALER_LOCAL_PATH = "scaler.joblib"

@st.cache_resource
def load_model_and_scaler():
    # If model exists in repo, load directly
    if Path(MODEL_LOCAL_PATH).exists() and Path(SCALER_LOCAL_PATH).exists():
        model = joblib.load(MODEL_LOCAL_PATH)
        scaler = joblib.load(SCALER_LOCAL_PATH)
        return model, scaler

    # Else, attempt to download from URL set in secrets or env
    model_url = st.secrets.get("MODEL_URL", None) or os.environ.get("MODEL_URL")
    scaler_url = st.secrets.get("SCALER_URL", None) or os.environ.get("SCALER_URL")

    if model_url:
        # naive HTTP download (works for public URLs); for Google Drive use gdown
        for url, out in [(model_url, MODEL_LOCAL_PATH), (scaler_url, SCALER_LOCAL_PATH)]:
            if url:
                try:
                    r = requests.get(url, stream=True, timeout=30)
                    r.raise_for_status()
                    with open(out, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    st.warning(f"Could not download {out}: {e}")
        # try loading after download
        if Path(MODEL_LOCAL_PATH).exists() and Path(SCALER_LOCAL_PATH).exists():
            model = joblib.load(MODEL_LOCAL_PATH)
            scaler = joblib.load(SCALER_LOCAL_PATH)
            return model, scaler

    # If still not available, return Nones
    return None, None

model, scaler = load_model_and_scaler()

st.title("Credit Card Delinquency — Early Risk Signals")
st.write("Enter customer details and get a risk score + recommended outreach action.")

# Input widgets
util = st.slider("Utilisation (0–1)", 0.0, 1.0, 0.5, step=0.01)
avg_pay = st.slider("Avg Payment Ratio (0–1)", 0.0, 1.0, 0.6, step=0.01)
min_due = st.slider("Min Due Frequency (0–1)", 0.0, 1.0, 0.25, step=0.01)
merchant = st.slider("Merchant Mix Index (0–1)", 0.0, 1.0, 0.5, step=0.01)
cash_pct = st.number_input("Cash Withdrawal % (0–100)", min_value=0, max_value=100, value=10)
spend_chg = st.number_input("Recent Spend Change % (e.g., -20, 15)", value=0)

# Rule-based flag score (same as used in analysis)
flag_score = (
    (util >= 0.85)*3 +
    (min_due >= 0.40)*2 +
    (avg_pay < 0.40)*2 +
    (merchant >= 0.75)*1 +
    (cash_pct >= np.percentile([cash_pct], 75) if False else (cash_pct >= 50))*1 +  # approximate check
    (abs(spend_chg) > 20)*1
)
# (Above: cash threshold simplified; if you have training data, compute exact quartiles once and hardcode)

st.metric("Flag Score", int(flag_score))

# Model score (if model available)
if model is not None and scaler is not None:
    X = pd.DataFrame([[util, avg_pay, min_due, merchant, cash_pct, spend_chg]],
                     columns=["util_pct","avg_payment_ratio","min_due_freq","merchant_mix","cash_withdraw_pct","recent_spend_change"])
    # scaling if scaler exists
    try:
        Xs = scaler.transform(X)
    except Exception:
        Xs = X.values
    score = model.predict_proba(Xs)[0,1]
    st.metric("Model Risk Score", f"{score:.3f}")
else:
    st.info("Model not available in repo. Upload rf_model.joblib and scaler.joblib to repo, or set MODEL_URL secret to download them at startup.")
    score = None

# Recommendation logic
if (flag_score >= 6) or (score is not None and score >= 0.6):
    st.error("Recommendation: HIGH priority outreach — SMS + payment link + agent call")
elif (flag_score >= 3) or (score is not None and score >= 0.35):
    st.warning("Recommendation: MEDIUM priority outreach — SMS reminder")
else:
    st.success("Recommendation: LOW risk — Monitor only")

# Optional: download sample outreach_list
if Path("outreach_list.xlsx").exists():
    with open("outreach_list.xlsx", "rb") as f:
        st.download_button("Download outreach_list.xlsx", f, file_name="outreach_list.xlsx")
