import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
* { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #f5f5f0; }
[data-testid="stSidebar"] { background: #1a1a1a; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.result-box { border-radius: 4px; padding: 28px; text-align: center; margin: 16px 0; }
.pass-box { background: #e8f5e9; border: 2px solid #4caf50; }
.fail-box { background: #ffebee; border: 2px solid #f44336; }
.result-main { font-family: 'IBM Plex Mono', monospace; font-size: 40px; font-weight: 600; }
.pass-txt { color: #2e7d32; }
.fail-txt { color: #c62828; }
.result-sub { font-size: 15px; color: #555; margin-top: 8px; }
.section-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; letter-spacing: 2px;
    text-transform: uppercase; color: #888;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px; margin: 24px 0 14px 0;
}
.stButton > button {
    background: #1a1a1a; color: white; border: none;
    border-radius: 4px; font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; padding: 10px 0; width: 100%;
}
.stButton > button:hover { background: #333; }
</style>
""", unsafe_allow_html=True)

# ── Load saved models ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.pkl")
    sel_c  = joblib.load("models/selector_cls.pkl")
    sel_r  = joblib.load("models/selector_reg.pkl")
    lr     = joblib.load("models/logistic.pkl")
    nb     = joblib.load("models/naive_bayes.pkl")
    rid    = joblib.load("models/ridge.pkl")
    las    = joblib.load("models/lasso.pkl")
    return scaler, sel_c, sel_r, lr, nb, rid, las

scaler, sel_c, sel_r, lr, nb, rid, las = load_models()

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("# Student Performance Predictor")
st.markdown("Enter student details below to get a model-based prediction.")
st.markdown("---")

with st.form("predict"):
    student_id = st.number_input("Student ID", min_value=1, max_value=9999, value=1, step=1)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        attendance    = st.slider("Attendance Rate (%)", 40, 100, 75)
        study_hours   = st.slider("Study Hours Per Day", 1.0, 10.0, 5.0, 0.5)
        prev_marks    = st.slider("Previous Marks", 30, 100, 65)
        assignment    = st.slider("Assignment Score", 20, 100, 70)
        quiz          = st.slider("Quiz Score", 20, 100, 65)
    with c2:
        participation = st.slider("Class Participation (1–10)", 1, 10, 6)
        sleep         = st.slider("Sleep Hours Per Day", 4.0, 10.0, 7.0, 0.5)
        stress        = st.slider("Stress Level (1–10)", 1, 10, 5)
        internet      = st.selectbox("Internet Access", [1, 0],
                                     format_func=lambda x: "Yes" if x else "No")
        parental      = st.slider("Parental Support (1–5)", 1, 5, 3)

    submitted = st.form_submit_button("Predict")

# ── Predict ───────────────────────────────────────────────────────────────────
if submitted:
    raw      = np.array([[attendance, study_hours, prev_marks, assignment,
                          quiz, participation, sleep, stress, internet, parental]])
    Xs       = scaler.transform(raw)
    Xc       = sel_c.transform(Xs)
    Xr       = sel_r.transform(Xs)

    lr_pred  = lr.predict(Xc)[0]
    nb_pred  = nb.predict(Xc)[0]
    rid_pred = float(np.clip(rid.predict(Xr)[0], 0, 100))
    las_pred = float(np.clip(las.predict(Xr)[0], 0, 100))
    avg_score = (rid_pred + las_pred) / 2

    final   = (int(lr_pred) + int(nb_pred)) >= 1
    label   = "PASS" if final else "FAIL"
    box_cls = "pass-box" if final else "fail-box"
    txt_cls = "pass-txt" if final else "fail-txt"

    st.markdown(f"""
    <div class='result-box {box_cls}'>
        <div style='font-size:12px; color:#888; letter-spacing:1px; margin-bottom:6px;'>STUDENT ID: {int(student_id)}</div>
        <div class='result-main {txt_cls}'>{label}</div>
        <div class='result-sub'>Predicted Final Score: <b>{avg_score:.1f} / 100</b></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-lbl'>Model Breakdown</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Model":  ["Logistic Regression", "Naive Bayes", "Ridge Regression", "Lasso Regression"],
        "Type":   ["Classification", "Classification", "Regression", "Regression"],
        "Output": ["Pass" if lr_pred else "Fail", "Pass" if nb_pred else "Fail",
                   f"{rid_pred:.1f}", f"{las_pred:.1f}"],
    }), use_container_width=True, hide_index=True)
