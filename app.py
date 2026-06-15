import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

st.write("sklearn:", sklearn.__version__)
st.write("imputer type:", type(imputer))
st.write("has _fill_dtype:", hasattr(imputer, "_fill_dtype"))

model_bundle = joblib.load('aussie_rain.joblib')
model    = model_bundle['model']
imputer  = model_bundle['imputer']
scaler   = model_bundle['scaler']
encoder  = model_bundle['encoder']
num_cols = model_bundle['numeric_cols']
cat_cols = model_bundle['categorical_cols']
enc_cols = model_bundle['encoded_cols']

st.title(" Will it Rain Tomorrow in Australia?")

col1, col2 = st.columns(2)
with col1:
    min_temp = st.number_input("Min Temp (°C)", value=12.0)
    max_temp = st.number_input("Max Temp (°C)", value=24.0)
    humidity_3pm = st.slider("Humidity 3pm (%)", 0, 100, 55)
with col2:
    wind_gust = st.number_input("Wind Gust Speed", value=40.0)
    pressure  = st.number_input("Pressure 9am", value=1017.0)
    rain_today = st.selectbox("Rain Today?", ["No", "Yes"])

if st.button("Predict"):

    inp = {c: np.nan for c in num_cols + cat_cols}

    inp.update({
        'MinTemp': min_temp,
        'MaxTemp': max_temp,
        'Humidity3pm': humidity_3pm,
        'WindGustSpeed': wind_gust,
        'Pressure9am': pressure,
        'RainToday': rain_today
    })

    df = pd.DataFrame([inp])

    try:
        # Numeric preprocessing
        num_data = imputer.transform(df[num_cols])
        num_data = scaler.transform(num_data)

        # Categorical preprocessing
        cat_data = encoder.transform(df[cat_cols])

        # Final dataframe
        final_df = pd.DataFrame(
            np.hstack([num_data, cat_data]),
            columns=num_cols + enc_cols
        )

        pred = model.predict(final_df)[0]
        prob = model.predict_proba(final_df)[0]

        st.success(
            f"Prediction: {'🌧 Rain' if pred in [1,'Yes'] else '☀ No Rain'}"
        )

        st.metric(
            "Rain probability",
            f"{prob[1]:.1%}"
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
