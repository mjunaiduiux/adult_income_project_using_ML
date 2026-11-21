import json
import os

import joblib
import pandas as pd
import requests
import streamlit as st

import features  # noqa: F401

st.set_page_config(page_title="Income Predictor", page_icon="💰", layout="wide")


def get_api_url() -> str:
    try:
        return st.secrets["API_URL"]
    except Exception:
        return os.environ.get("API_URL", "http://localhost:8000")


API_URL = get_api_url().rstrip("/")


@st.cache_resource
def load_local_model():
    return joblib.load("best_model.joblib")


@st.cache_data
def load_info():
    with open("model_info.json") as f:
        return json.load(f)


def predict(payload: dict):
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), "FastAPI"
    except Exception:
        row = pd.DataFrame([payload])
        model = load_local_model()
        label = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None
        return {"prediction": ">50K" if label else "<=50K", "prediction_label": label,
                "probability_over_50k": proba, "model": info["model_name"]}, "local model (API unreachable)"


info = load_info()
schema = info["schema"]

st.title("💰 Adult Income Predictor")
st.caption("Predicts whether annual income exceeds $50K (UCI Adult Income dataset).")

with st.sidebar:
    st.header("Model")
    st.write(f"**{info['model_name']}**")
    st.subheader("Test-set performance")
    m = info["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", m["roc_auc"]); c2.metric("Accuracy", m["accuracy"])
    c1.metric("F1", m["f1"]); c2.metric("Precision", m["precision"])
    c1.metric("Recall", m["recall"])
    st.subheader("Top 5 features")
    top5 = pd.DataFrame(info["top5_features"]).set_index("feature")
    st.bar_chart(top5["importance"])
    st.dataframe(top5, use_container_width=True)
    st.caption("Permutation importance: drop in test ROC-AUC when the feature is shuffled.")

num, cat = schema["numeric"], schema["categorical"]


def pick(col, default=None):
    opts = cat[col]
    return st.selectbox(col.replace("_", " ").title(), opts, index=opts.index(default) if default in opts else 0)


with st.form("applicant"):
    a, b, c = st.columns(3)
    with a:
        age = st.number_input("Age", 17, 100, 39)
        education_num = st.slider("Education (years)", 1, 16, 13)
        hours = st.number_input("Hours per week", 1, 99, 40)
        sex = pick("sex", "Male")
    with b:
        workclass = pick("workclass", "Private")
        occupation = pick("occupation", "Exec-managerial")
        marital = pick("marital_status", "Married-civ-spouse")
        relationship = pick("relationship", "Husband")
    with c:
        race = pick("race", "White")
        country = pick("native_country", "United-States")
        gain = st.number_input("Capital gain", 0, 100000, 0)
        loss = st.number_input("Capital loss", 0, 5000, 0)
    submitted = st.form_submit_button("Predict", type="primary")

if submitted:
    payload = dict(age=int(age), workclass=workclass, education_num=int(education_num),
                   marital_status=marital, occupation=occupation, relationship=relationship,
                   race=race, sex=sex, capital_gain=float(gain), capital_loss=float(loss),
                   hours_per_week=int(hours), native_country=country)
    result, source = predict(payload)
    if result["prediction_label"] == 1:
        st.success(f"Predicted income: **{result['prediction']}**")
    else:
        st.info(f"Predicted income: **{result['prediction']}**")
    if result.get("probability_over_50k") is not None:
        st.progress(result["probability_over_50k"], text=f"P(>50K) = {result['probability_over_50k']:.1%}")
    st.caption(f"Served by: {source}")
    with st.expander("Request payload"):
        st.json(payload)
