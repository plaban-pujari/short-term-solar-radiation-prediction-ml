import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
import os


st.set_page_config(
    page_title="Solar Predictor",
    layout="centered"
)

st.title("☀️ Solar Radiation Prediction System")
st.markdown(
    "### Predict Solar Radiation & Open COMSOL Simulation"
)


rf_r2 = 0.8513
xgb_r2 = 0.8464
ann_r2 = 0.7916
lstm_r2 = 0.7200
hybrid_r2 = 0.8537


prediction_keys = [
    "rf_pred",
    "xgb_pred",
    "ann_pred",
    "lstm_pred",
    "hybrid_pred"
]

for key in prediction_keys:
    if key not in st.session_state:
        st.session_state[key] = None


@st.cache_resource
def load_models():

    rf = joblib.load("rf_model_final.pkl")

    xgb = joblib.load("xgb_model_final.pkl")

    ann = tf.keras.models.load_model(
        "ann_model.keras"
    )

    lstm = tf.keras.models.load_model(
        "lstm_model.keras"
    )

    hybrid_xgb = joblib.load(
        "xgb_model.pkl"
    )

    scaler_X_ann = joblib.load(
        "scaler_X_ann.pkl"
    )

    scaler_y_ann = joblib.load(
        "scaler_y_ann.pkl"
    )

    scaler_X_lstm = joblib.load(
        "scaler_X_lstm.pkl"
    )

    scaler_y_lstm = joblib.load(
        "scaler_y_lstm.pkl"
    )

    scaler_X_hybrid = joblib.load(
        "scaler_X.pkl"
    )

    scaler_y_hybrid = joblib.load(
        "scaler_y.pkl"
    )

    return (
        rf,
        xgb,
        ann,
        lstm,
        hybrid_xgb,
        scaler_X_ann,
        scaler_y_ann,
        scaler_X_lstm,
        scaler_y_lstm,
        scaler_X_hybrid,
        scaler_y_hybrid
    )

(
    rf_model,
    xgb_model,
    ann_model,
    lstm_model,
    hybrid_xgb,
    scaler_X_ann,
    scaler_y_ann,
    scaler_X_lstm,
    scaler_y_lstm,
    scaler_X_hybrid,
    scaler_y_hybrid
) = load_models()


st.subheader("📊 Radiation History")

col1, col2, col3 = st.columns(3)

with col1:
    lag3 = st.number_input(
        "3rd Last Radiation",
        0.0,
        1200.0,
        200.0
    )

with col2:
    lag2 = st.number_input(
        "2nd Last Radiation",
        0.0,
        1200.0,
        250.0
    )

with col3:
    lag1 = st.number_input(
        "Latest Radiation",
        0.0,
        1200.0,
        300.0
    )

rolling_mean_3 = np.mean(
    [lag1, lag2, lag3]
)


st.subheader("🌤 Weather Parameters")

col4, col5, col6 = st.columns(3)

with col4:
    temperature = st.number_input(
        "Temperature (°C)",
        -10.0,
        60.0,
        30.0
    )

with col5:
    humidity = st.number_input(
        "Humidity (%)",
        0.0,
        100.0,
        60.0
    )

with col6:
    wind_speed = st.number_input(
        "Wind Speed (m/s)",
        0.0,
        20.0,
        3.0
    )


now = datetime.now()

hour = now.hour

day_of_year = now.timetuple().tm_yday

hour_sin = np.sin(
    2 * np.pi * hour / 24
)

hour_cos = np.cos(
    2 * np.pi * hour / 24
)


features = np.array([[
    lag1,
    lag2,
    lag3,
    rolling_mean_3,
    hour_sin,
    hour_cos,
    day_of_year,
    temperature,
    humidity,
    wind_speed
]])


if st.button("🚀 Predict Radiation"):

    # RANDOM FOREST
    rf_pred = rf_model.predict(
        features
    )[0]

    # XGBOOST
    xgb_pred = xgb_model.predict(
        features
    )[0]

    # ANN
    X_ann = scaler_X_ann.transform(
        features
    )

    ann_scaled = ann_model.predict(
        X_ann,
        verbose=0
    )

    ann_pred = scaler_y_ann.inverse_transform(
        ann_scaled
    )[0][0]

    # LSTM
    X_lstm = scaler_X_lstm.transform(
        features
    )

    seq = np.repeat(
        X_lstm,
        8,
        axis=0
    ).reshape(1, 8, -1)

    lstm_scaled = lstm_model.predict(
        seq,
        verbose=0
    )

    lstm_pred = scaler_y_lstm.inverse_transform(
        lstm_scaled
    )[0][0]

    # HYBRID
    X_hybrid = scaler_X_hybrid.transform(
        features
    )

    seq_h = np.repeat(
        X_hybrid,
        6,
        axis=0
    ).reshape(1, 6, -1)

    hybrid_lstm_scaled = lstm_model.predict(
        seq_h,
        verbose=0
    )

    hybrid_lstm_pred = scaler_y_hybrid.inverse_transform(
        hybrid_lstm_scaled
    )[0][0]

    xgb_input = pd.DataFrame(
        X_hybrid,
        columns=[
            'lag1',
            'lag2',
            'lag3',
            'rolling_mean_3',
            'hour_sin',
            'hour_cos',
            'day_of_year',
            'temperature',
            'humidity',
            'wind_speed'
        ]
    )

    xgb_input['lstm_pred'] = hybrid_lstm_pred

    residual = hybrid_xgb.predict(
        xgb_input
    )[0]

    hybrid_pred = hybrid_lstm_pred + residual


    st.session_state.rf_pred = rf_pred
    st.session_state.xgb_pred = xgb_pred
    st.session_state.ann_pred = ann_pred
    st.session_state.lstm_pred = lstm_pred
    st.session_state.hybrid_pred = hybrid_pred


    radiation_values = [
        rf_pred,
        xgb_pred,
        ann_pred,
        lstm_pred,
        hybrid_pred
    ]

    values_string = ",".join(
        str(v) for v in radiation_values
    )

    with open(
        "radiation_values.txt",
        "w"
    ) as f:

        f.write(
            f"G {values_string}"
        )

    st.success(
        "✅ Radiation values saved for COMSOL simulation"
    )


if st.session_state.rf_pred is not None:

    st.subheader(
        "📈 Predicted Solar Radiation"
    )

 
    r2_scores = {
        "Random Forest": rf_r2,
        "XGBoost": xgb_r2,
        "ANN": ann_r2,
        "LSTM": lstm_r2,
        "Hybrid": hybrid_r2
    }

    best_model = max(
        r2_scores,
        key=r2_scores.get
    )

    st.success(
        f"🏆 Recommended Model: "
        f"{best_model} "
        f"(Highest R² = "
        f"{r2_scores[best_model]:.4f})"
    )


    st.markdown(
        f"""
        <div style="
            padding:10px;
            border-radius:10px;
            background-color:#1e1e1e;
            margin-bottom:10px;
        ">
        🌲 <b>Random Forest</b><br>
        Prediction:
        <b>{st.session_state.rf_pred:.2f} W/m²</b><br>
        R² Score:
        <b>{rf_r2}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div style="
            padding:10px;
            border-radius:10px;
            background-color:#1e1e1e;
            margin-bottom:10px;
        ">
        ⚡ <b>XGBoost</b><br>
        Prediction:
        <b>{st.session_state.xgb_pred:.2f} W/m²</b><br>
        R² Score:
        <b>{xgb_r2}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div style="
            padding:10px;
            border-radius:10px;
            background-color:#1e1e1e;
            margin-bottom:10px;
        ">
        🧠 <b>ANN</b><br>
        Prediction:
        <b>{st.session_state.ann_pred:.2f} W/m²</b><br>
        R² Score:
        <b>{ann_r2}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div style="
            padding:10px;
            border-radius:10px;
            background-color:#1e1e1e;
            margin-bottom:10px;
        ">
        🔄 <b>LSTM</b><br>
        Prediction:
        <b>{st.session_state.lstm_pred:.2f} W/m²</b><br>
        R² Score:
        <b>{lstm_r2}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        f"""
        <div style="
            padding:15px;
            border-radius:12px;
            background-color:#0f5132;
            border:2px solid #00ff99;
            margin-bottom:10px;
        ">
        🔥 <b>Hybrid (LSTM + XGBoost)</b><br>
        Prediction:
        <b>{st.session_state.hybrid_pred:.2f} W/m²</b><br>
        R² Score:
        <b>{hybrid_r2}</b><br><br>

        ✅ <b>Best Performing Model</b>
        </div>
        """,
        unsafe_allow_html=True
    )


if st.button(
    "⚡ Open COMSOL Simulation"
):

    st.info(
        "Opening COMSOL Model..."
    )

    os.system(
        r'start "" "C:\Users\plaba\Downloads\COMSOL61\Multiphysics\bin\win64\comsol.exe" '
        r'"C:\Users\plaba\OneDrive\Documents\CODE\New folder\streamlit\solar_power_model.mph"'
    )

    st.success(
        "✅ COMSOL Model Opened!"
    )


st.markdown("---")

st.caption(
    "Solar Forecasting & COMSOL Integrated System"
)