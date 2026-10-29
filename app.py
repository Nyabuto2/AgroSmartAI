#%%writefile app.py
"""AgroSmart AI: crop recommendation demo (Streamlit)."""
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AgroSmart AI", page_icon="🌱", layout="centered")

MODEL_PATH = "agrosmart_model.joblib"


@st.cache_resource
def load_bundle(path):
    bundle = joblib.load(path)
    return bundle["model"], bundle["label_encoder"], bundle["features"]


try:
    model, le, FEATURES = load_bundle(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file '{MODEL_PATH}' not found. Keep it in the same folder as app.py.")
    st.stop()

# Slider ranges follow the ranges seen in the training data.
INPUTS = {
    "N":           dict(label="Nitrogen (N)",    lo=0,    hi=140,   default=50,   step=1),
    "P":           dict(label="Phosphorus (P)",  lo=5,    hi=145,   default=53,   step=1),
    "K":           dict(label="Potassium (K)",   lo=5,    hi=205,   default=48,   step=1),
    "ph":          dict(label="Soil pH",         lo=3.5,  hi=10.0,  default=6.5,  step=0.1),
    "temperature": dict(label="Temperature (°C)", lo=8.0, hi=44.0,  default=25.6, step=0.1),
    "humidity":    dict(label="Humidity (%)",    lo=14.0, hi=100.0, default=71.0, step=0.5),
    "rainfall":    dict(label="Rainfall (mm)",   lo=20.0, hi=300.0, default=100.0, step=1.0),
}

# Example conditions, taken from the average values per crop in the training data.
PRESETS = {
    "Custom": None,
    "Rice-like":     dict(N=90,  P=42,  K=43,  ph=6.5, temperature=20.9, humidity=82.0, rainfall=202.9),
    "Apple-like":    dict(N=21,  P=134, K=200, ph=5.9, temperature=22.6, humidity=92.3, rainfall=112.7),
    "Chickpea-like": dict(N=40,  P=68,  K=80,  ph=7.3, temperature=18.9, humidity=16.9, rainfall=80.1),
    "Coconut-like":  dict(N=22,  P=17,  K=31,  ph=6.0, temperature=27.4, humidity=94.8, rainfall=175.7),
    "Coffee-like":   dict(N=101, P=29,  K=30,  ph=6.8, temperature=25.5, humidity=58.9, rainfall=158.1),
}

EMOJI = {
    "apple": "🍎", "banana": "🍌", "coconut": "🥥", "coffee": "☕", "grapes": "🍇",
    "maize": "🌽", "mango": "🥭", "muskmelon": "🍈", "orange": "🍊", "rice": "🌾",
    "watermelon": "🍉",
}
NICE_NAMES = {
    "kidneybeans": "Kidney beans", "mothbeans": "Moth beans", "mungbean": "Mung bean",
    "pigeonpeas": "Pigeon peas", "blackgram": "Black gram",
}


def pretty(crop):
    return NICE_NAMES.get(crop, crop.title())


def icon(crop):
    return EMOJI.get(crop, "🌱")


def apply_preset():
    preset = PRESETS[st.session_state["preset"]]
    if preset:
        for name, value in preset.items():
            st.session_state[name] = value


for name, spec in INPUTS.items():
    st.session_state.setdefault(name, spec["default"])

st.title("🌱 AgroSmart AI")
st.caption("Crop recommendation from soil nutrients and local climate conditions.")

st.selectbox("Quick start: load an example", list(PRESETS), key="preset", on_change=apply_preset)

col_soil, col_climate = st.columns(2)
with col_soil:
    st.subheader("Soil")
    for name in ("N", "P", "K", "ph"):
        s = INPUTS[name]
        st.slider(s["label"], s["lo"], s["hi"], step=s["step"], key=name)
with col_climate:
    st.subheader("Climate")
    for name in ("temperature", "humidity", "rainfall"):
        s = INPUTS[name]
        st.slider(s["label"], s["lo"], s["hi"], step=s["step"], key=name)

# Predict with the same feature order the model was trained on.
row = pd.DataFrame([[st.session_state[f] for f in FEATURES]], columns=FEATURES)
proba = model.predict_proba(row)[0]
crops = le.inverse_transform(model.classes_)
order = np.argsort(proba)[::-1][:3]
top = [(crops[i], float(proba[i])) for i in order]

best, best_p = top[0]
st.divider()
st.subheader("Recommended crop")
left, right = st.columns([3, 1])
left.markdown(f"## {icon(best)} {pretty(best)}")
right.metric("Confidence", f"{best_p:.0%}")

if best_p < 0.6:
    st.warning("These conditions sit between several crops. Treat the top 3 as candidates, not a single answer.")

st.markdown("**Top 3 candidates**")
for crop, p in top:
    st.write(f"{icon(crop)} {pretty(crop)}: {p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))

with st.expander("About this model"):
    st.write(
        "Random Forest (400 trees) trained on the public Kaggle Crop Recommendation dataset: "
        "2,200 samples, 22 crops, 7 inputs. On 440 held-out samples it reached 99.55% accuracy "
        "and 100% top-3 accuracy."
    )
    importance = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    st.caption("Which inputs the model relies on most")
    st.bar_chart(importance)
    st.caption(
        "The data comes from conditions in India and the classes are very cleanly separated, so real-world "
        "accuracy will be lower. This is a decision-support demo, not agronomic advice."
    )