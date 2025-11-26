import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Credit Card Delinquency Risk Predictor")

util = st.slider("Utilisation (0–1)", 0.0, 1.0, 0.5)
avg_pay = st.slider("Avg Payment Ratio (0–1)", 0.0, 1.0, 0.6)
min_due = st.slider("Min Due Frequency (0–1)", 0.0, 1.0, 0.3)
merchant = st.slider("Merchant Mix Index", 0.0, 1.0, 0.5)
cash_pct = st.slider("Cash Withdrawal %", 0, 100, 10)
spend_chg = st.slider("Recent Spend Change %", -50, 50, 0)

flag_score = (
    (util >= 0.85)*3 +
    (min_due >= 0.40)*2 +
    (avg_pay < 0.40)*2 +
    (merchant >= 0.75)*1 +
    (cash_pct >= 50)*1 +
    (abs(spend_chg) > 20)*1
)

st.write("Flag Score:", flag_score)

if flag_score >= 6:
    st.error("High Risk — Immediate Outreach Recommended")
elif flag_score >= 3:
    st.warning("Medium Risk — Send Reminder")
else:
    st.success("Low Risk — Monitor Only")
