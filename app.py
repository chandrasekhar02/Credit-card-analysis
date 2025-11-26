# app.py - robust version (lazy model load)
import os
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Credit Card Delinquency Watch", layout="centered")

MODEL_LOCAL_PATH = "rf_model.joblib"
SCALER_LOCAL_PATH = "scaler.joblib"

@st.cache_resource
def load_model_if_available():
    # Lazy import inside function to avoid import-time errors if sklearn/numpy absent
    model = None
    scaler = None
    try:
        if Path(MODEL_LOCAL_PATH).exists() and Path(SCALER_LOCAL_PATH).exists():
            import joblib
            model = joblib.load(MODEL_LOCAL_PATH)
            scaler = joblib.load(SCALER_LOCAL_PATH)
    except Exception as e:
        st.warning(f"Model present but failed to load: {e}")
        model = None
        scaler = None
    return model, scaler

model, scaler = load_model_if_available()

st.title("Credit Card Delinquency — Early Risk Signals")

# Input widgets
util = st.slider("Utilisation (0–100%)", 0, 100, 50, step=1)
avg_pay = st.slider("Avg Payment Ratio (0–100%)", 0, 100, 60, step=1)
min_due = st.slider("Min Due Paid Frequency (0–100%)", 0, 100, 25, step=1)
merchant = st.slider("Merchant Mix Index (0–100)", 0, 100, 50, step=1)
cash_pct = st.number_input("Cash Withdrawal % (0–100)", min_value=0, max_value=100, value=10)
spend_chg = st.number_input("Recent Spend Change % (e.g., -20, 15)", value=0)

# Convert percentages to decimals if your logic uses decimals
util_pct = util/100.0
avg_payment_ratio = avg_pay/100.0
min_due_freq = min_due/100.0
merchant_mix = merchant/100.0
recent_spend_change = spend_chg

# Flag logic (same thresholds as analysis)
high_util = int(util_pct >= 0.85)
min_due_flag = int(min_due_freq >= 0.40)
low_payment = int(avg_payment_ratio < 0.40)
merchant_high = int(merchant_mix >= 0.75)
cash_high = int(cash_pct >= 50)  # simplified threshold
spend_volatility = int(abs(recent_spend_change) > 20)

flag_score = 3*high_util + 2*min_due_flag + 2*low_payment + merchant_high + cash_high + spend_volatility
risk_flag = "High" if flag_score >= 6 else ("Medium" if flag_score >= 3 else "Low")

st.metric("Flag Score", int(flag_score))
st.write("Risk flag:", risk_flag)

# If model available, compute model_score. Use lazy import to avoid import errors if sklearn missing
model_score = None
if model is not None and scaler is not None:
    try:
        import numpy as np
        X = [[util_pct, avg_payment_ratio, min_due_freq, merchant_mix, cash_pct, recent_spend_change]]
        Xs = scaler.transform(X)
        model_score = float(model.predict_proba(Xs)[0,1])
        st.metric("Model Risk Score", f"{model_score:.3f}")
    except Exception as e:
        st.warning(f"Model present but failed at predict: {e}")
else:
    st.info("Model files not found in repo or failed to load. Flag-based output still works.")

# Recommendation combining flag & model_score
if (flag_score >= 6) or (model_score is not None and model_score >= 0.6):
    st.error("Recommendation: HIGH priority outreach — SMS + payment link + agent call")
elif (flag_score >= 3) or (model_score is not None and model_score >= 0.35):
    st.warning("Recommendation: MEDIUM priority outreach — SMS reminder")
else:
    st.success("Recommendation: LOW risk — Monitor only")

# Optional: allow downloading outreach list if present
if Path("outreach_list.xlsx").exists():
    with open("outreach_list.xlsx", "rb") as f:
        st.download_button("Download outreach_list.xlsx", f, file_name="outreach_list.xlsx")
