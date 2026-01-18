
import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="EPL Match Outcome Predictor", page_icon="⚽", layout="centered")

st.title("⚽ Premier League Match Outcome Predictor")
st.write("Predict whether the **Home Team will Win** based on match statistics.")

# Load model + feature list
model = joblib.load("best_model.pkl")
feature_cols = joblib.load("model_columns.pkl")

st.divider()
st.subheader("Enter Match Statistics")

# Build UI inputs dynamically from the saved feature list
inputs = {}
for col in feature_cols:
    inputs[col] = st.number_input(f"{col}", min_value=0.0, value=0.0, step=1.0)

input_df = pd.DataFrame([inputs])

st.divider()

if st.button("Predict"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("✅ Prediction: Home Team WILL WIN (Home Win)")
    else:
        st.warning("⚠️ Prediction: Home Team will NOT WIN (Draw or Away Win)")

    # Optional: show probability if supported
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]
        st.info(f"Home Win Probability: {proba*100:.2f}%")
