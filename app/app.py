# app/app.py

# Streamlit app to visualize grid balance predictions and recommendations based on user input of desired date and hour.

import streamlit as st
import pandas as pd
import json
from pathlib import Path

st.set_page_config(page_title="Alberta Grid Balance Predictor", layout="centered")
st.title("Alberta Grid Balance Predictor & Recommender")
st.caption("Educational prototype. Recommendations are for illustrative purposes only. Times in MST.")
st.caption("By Andrei Tihan, September 2025. https://github.com/andrei-tihan")

PRED_PATH = Path("data/processed/predictions.csv")
META_PATH = Path("models/meta.json")
CLEAN_PATH = Path("data/processed/alberta_hourly_clean.csv")

if not PRED_PATH.exists():
    st.error(
        "Predictions not found. Run the training pipeline first:\n`python -m src.train_pipeline --gen data/raw/alberta_net_generation.csv --load data/raw/alberta_internal_load.csv`")
else:
    df = pd.read_csv(PRED_PATH, parse_dates=["timestamp"])

    # Timezone handling
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert("America/Edmonton")
    except Exception:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize("America/Edmonton", ambiguous='infer',
                                                                         nonexistent='shift_forward')

    meta = {}
    if META_PATH.exists():
        meta = json.load(open(META_PATH))

    # Choose date here
    min_date = df['timestamp'].dt.date.min()
    max_date = df['timestamp'].dt.date.max()
    date_choice = st.date_input("Select date", value=max_date, min_value=min_date, max_value=max_date)

    # Filter for selected date
    day_df = df[df['timestamp'].dt.date == pd.to_datetime(date_choice).date()]

    if day_df.empty:
        st.warning("No data for this date in the dataset.")
    else:
        # Hour selector
        day_df['hour_label'] = day_df['timestamp'].dt.strftime('%I %p')
        hours_for_day = day_df['hour_label'].unique().tolist()
        hour_choice = st.selectbox("Select hour", options=hours_for_day)

        # Map back to timestamp
        sel_row = day_df[day_df['hour_label'] == hour_choice].iloc[0]
        sel_ts = sel_row['timestamp']

        st.metric("Predicted balance (MW)", f"{sel_row['y_pred']:.0f}")
        st.write(f"**Band:** {sel_row['band']}")
        st.write(f"**What it means:** {sel_row['grid_description']}")
        st.write(f"**Grid action:** {sel_row['grid_recommendation']}")
        st.write(f"**Customer action:** {sel_row['customer_recommendation']}")
        st.write(f"Prediction band: [{sel_row['y_lower']:.0f}, {sel_row['y_upper']:.0f}] MW")

        # 72-hour context
        t0 = sel_ts - pd.Timedelta(hours=36)
        t1 = sel_ts + pd.Timedelta(hours=36)
        win = df[(df['timestamp'] >= t0) & (df['timestamp'] <= t1)].copy().set_index('timestamp')
        st.line_chart(win[['balance_mw']])
        st.caption("Predicted balance over 72 hour period. Positive = surplus, negative = deficit.")

# Run with: streamlit run app/app.py