import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
* { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #f5f5f0; }
[data-testid="stSidebar"] { background: #1a1a1a; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 14px; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
.card {
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 20px;
    margin-bottom: 16px;
}
.stat-box {
    background: #1a1a1a;
    color: white;
    border-radius: 4px;
    padding: 18px 12px;
    text-align: center;
}
.stat-val { font-family: 'IBM Plex Mono', monospace; font-size: 26px; font-weight: 600; color: #c8f0b0; }
.stat-lbl { font-size: 11px; color: #999; margin-top: 4px; letter-spacing: 1px; text-transform: uppercase; }
.result-box {
    border-radius: 4px;
    padding: 28px;
    text-align: center;
    margin: 16px 0;
}
.pass-box { background: #e8f5e9; border: 2px solid #4caf50; }
.fail-box { background: #ffebee; border: 2px solid #f44336; }
.result-main { font-family: 'IBM Plex Mono', monospace; font-size: 36px; font-weight: 600; }
.pass-txt { color: #2e7d32; }
.fail-txt { color: #c62828; }
.result-sub { font-size: 15px; color: #555; margin-top: 8px; }
.section-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888;
    border-bottom: 1px solid #ddd;
    padding-bottom: 6px;
    margin: 24px 0 14px 0;
}
.stButton > button {
    background: #1a1a1a;
    color: white;
    border: none;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    padding: 10px 0;
    width: 100%;
}
.stButton > button:hover { background: #333; }
</style>
""", unsafe_allow_html=True)

# - Load data & train
@st.cache_data
def load_train():
    df = pd.read_csv("student_performance.csv")

    feats = ["Attendance_Rate(%)", "Study_Hours_Per_Day", "Previous_Marks",
             "Assignment_Score", "Quiz_Score", "Class_Participation(1-10)",
             "Sleep_Hours_Per_Day", "Stress_Level(1-10)",
             "Internet_Access(0/1)", "Parental_Support(1-5)"]

    X = df[feats].values
    y_cls = (df["Result"] == "Pass").astype(int).values
    y_reg = df["Final_Score"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    sel_c = SelectKBest(f_classif, k=8).fit(Xs, y_cls)
    sel_r = SelectKBest(f_regression, k=8).fit(Xs, y_reg)
    Xc = sel_c.transform(Xs)
    Xr = sel_r.transform(Xs)

    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, y_cls, test_size=0.2, random_state=42, stratify=y_cls)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, y_reg, test_size=0.2, random_state=42)

    lr  = LogisticRegression(max_iter=500, random_state=42).fit(Xc_tr, yc_tr)
    nb  = GaussianNB().fit(Xc_tr, yc_tr)
    rid = Ridge(alpha=1.0).fit(Xr_tr, yr_tr)
    las = Lasso(alpha=0.1).fit(Xr_tr, yr_tr)

    metrics = {
        "Logistic Regression": {"type": "cls", "acc": accuracy_score(yc_te, lr.predict(Xc_te))},
        "Naive Bayes":         {"type": "cls", "acc": accuracy_score(yc_te, nb.predict(Xc_te))},
        "Ridge Regression":    {"type": "reg",
                                 "mse": mean_squared_error(yr_te, rid.predict(Xr_te)),
                                 "r2":  r2_score(yr_te, rid.predict(Xr_te))},
        "Lasso Regression":    {"type": "reg",
                                 "mse": mean_squared_error(yr_te, las.predict(Xr_te)),
                                 "r2":  r2_score(yr_te, las.predict(Xr_te))},
    }

    return scaler, sel_c, sel_r, lr, nb, rid, las, metrics, feats

scaler, sel_c, sel_r, lr, nb, rid, las, metrics, feats = load_train()
df = pd.read_csv("student_performance.csv")

# - Sidebar
with st.sidebar:
    st.markdown("### Menu")
    page = st.radio("", ["Dashboard", "Dataset", "Predict"], label_visibility="collapsed")


# DASHBOARD — Overview + Model Metrics
if page == "Dashboard":
    st.markdown("# Student Performance System")
    st.markdown("Supervised ML · Logistic Regression · Naive Bayes · Ridge · Lasso")
    st.markdown("---")

    # - Stat row
    lr_acc  = metrics["Logistic Regression"]["acc"]
    nb_acc  = metrics["Naive Bayes"]["acc"]
    rid_r2  = metrics["Ridge Regression"]["r2"]
    las_r2  = metrics["Lasso Regression"]["r2"]

    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip(
        [c1, c2, c3, c4],
        ["Logistic Acc", "Naive Bayes Acc", "Ridge R²", "Lasso R²"],
        [f"{lr_acc*100:.1f}%", f"{nb_acc*100:.1f}%", f"{rid_r2:.3f}", f"{las_r2:.3f}"]
    ):
        col.markdown(f"""
        <div class='stat-box'>
            <div class='stat-val'>{val}</div>
            <div class='stat-lbl'>{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # - Model comparison charts
    st.markdown("<div class='section-lbl'>Model Performance</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Classification accuracy bar
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")
        names = ["Logistic\nRegression", "Naive\nBayes"]
        vals  = [lr_acc * 100, nb_acc * 100]
        bars  = ax.bar(names, vals, color=["#1a1a1a", "#555"], width=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.1f}%",
                    ha="center", fontsize=10, fontfamily="monospace")
        ax.set_ylim(0, 110)
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.set_title("Classification Accuracy", fontsize=11, fontfamily="monospace")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with col2:
        # Regression R² bar
        fig, ax = plt.subplots(figsize=(5, 3))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")
        names = ["Ridge\nRegression", "Lasso\nRegression"]
        vals  = [rid_r2, las_r2]
        bars  = ax.bar(names, vals, color=["#1a1a1a", "#555"], width=0.4)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                    ha="center", fontsize=10, fontfamily="monospace")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("R² Score", fontsize=9)
        ax.set_title("Regression R² Score", fontsize=11, fontfamily="monospace")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # - Metrics table
    st.markdown("<div class='section-lbl'>Evaluation Summary</div>", unsafe_allow_html=True)
    table = pd.DataFrame({
        "Model":   ["Logistic Regression", "Naive Bayes", "Ridge Regression", "Lasso Regression"],
        "Type":    ["Classification", "Classification", "Regression", "Regression"],
        "Accuracy / R²": [
            f"{lr_acc*100:.2f}%", f"{nb_acc*100:.2f}%",
            f"{rid_r2:.4f}", f"{las_r2:.4f}"
        ],
        "Error (MSE)": [
            f"{(1-lr_acc)*100:.2f}%", f"{(1-nb_acc)*100:.2f}%",
            f"{metrics['Ridge Regression']['mse']:.4f}",
            f"{metrics['Lasso Regression']['mse']:.4f}"
        ],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    # - Best model
    best_cls = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"
    best_cls_val = max(lr_acc, nb_acc) * 100
    best_reg = "Ridge Regression" if metrics["Ridge Regression"]["r2"] >= metrics["Lasso Regression"]["r2"] else "Lasso Regression"
    best_reg_val = max(metrics["Ridge Regression"]["r2"], metrics["Lasso Regression"]["r2"])

    st.markdown("<div class='section-lbl'>Best Models</div>", unsafe_allow_html=True)
    bc1, bc2 = st.columns(2)
    bc1.markdown(f"""
    <div style='background:#1a3a1a; border:1px solid #4caf50; border-radius:4px; padding:16px; text-align:center;'>
        <div style='font-size:11px; color:#aaa; letter-spacing:1px; text-transform:uppercase;'>Best Classifier</div>
        <div style='font-family:IBM Plex Mono,monospace; font-size:20px; color:#c8f0b0; font-weight:600; margin-top:6px;'>{best_cls}</div>
        <div style='font-size:13px; color:#aaa; margin-top:4px;'>Accuracy: {best_cls_val:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
    bc2.markdown(f"""
    <div style='background:#1a3a1a; border:1px solid #4caf50; border-radius:4px; padding:16px; text-align:center;'>
        <div style='font-size:11px; color:#aaa; letter-spacing:1px; text-transform:uppercase;'>Best Regressor</div>
        <div style='font-family:IBM Plex Mono,monospace; font-size:20px; color:#c8f0b0; font-weight:600; margin-top:6px;'>{best_reg}</div>
        <div style='font-size:13px; color:#aaa; margin-top:4px;'>R² Score: {best_reg_val:.4f}</div>
    </div>
    """, unsafe_allow_html=True)

# DATASET
elif page == "Dataset":
    st.markdown("# Dataset Overview")
    st.markdown("Raw data sample and feature correlation.")
    st.markdown("---")

    st.markdown("<div class='section-lbl'>First 20 Rows</div>", unsafe_allow_html=True)
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    st.markdown("<div class='section-lbl'>Correlation Heatmap</div>", unsafe_allow_html=True)
    corr = df[feats + ["Final_Score"]].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.4, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Matrix", fontsize=12, fontfamily="monospace")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    
# PREDICT
elif page == "Predict":
    st.markdown("# Predict Student Outcome")
    st.markdown("Enter student details to get a model-based prediction.")
    st.markdown("---")

    with st.form("predict"):
        student_id = st.number_input("Student ID", min_value=1, max_value=9999, value=1, step=1)
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            attendance  = st.slider("Attendance Rate (%)", 40, 100, 75)
            study_hours = st.slider("Study Hours Per Day", 1.0, 10.0, 5.0, 0.5)
            prev_marks  = st.slider("Previous Marks", 30, 100, 65)
            assignment  = st.slider("Assignment Score", 20, 100, 70)
            quiz        = st.slider("Quiz Score", 20, 100, 65)
        with c2:
            participation = st.slider("Class Participation (1–10)", 1, 10, 6)
            sleep         = st.slider("Sleep Hours Per Day", 4.0, 10.0, 7.0, 0.5)
            stress        = st.slider("Stress Level (1–10)", 1, 10, 5)
            internet      = st.selectbox("Internet Access", [1, 0],
                                         format_func=lambda x: "Yes" if x else "No")
            parental      = st.slider("Parental Support (1–5)", 1, 5, 3)

        submitted = st.form_submit_button("Predict")

    if submitted:
        raw = np.array([[attendance, study_hours, prev_marks, assignment,
                         quiz, participation, sleep, stress, internet, parental]])
        Xs  = scaler.transform(raw)
        Xc  = sel_c.transform(Xs)
        Xr  = sel_r.transform(Xs)

        lr_pred  = lr.predict(Xc)[0]
        nb_pred  = nb.predict(Xc)[0]
        rid_pred = float(np.clip(rid.predict(Xr)[0], 0, 100))
        las_pred = float(np.clip(las.predict(Xr)[0], 0, 100))
        avg_score = (rid_pred + las_pred) / 2

        # Majority vote
        final = (int(lr_pred) + int(nb_pred)) >= 1

        box_cls  = "pass-box" if final else "fail-box"
        txt_cls  = "pass-txt" if final else "fail-txt"
        label    = "PASS" if final else "FAIL"

        st.markdown(f"""
        <div class='result-box {box_cls}'>
            <div style='font-size:12px; color:#888; letter-spacing:1px; margin-bottom:4px;'>STUDENT ID: {student_id if student_id else "—"}</div>
            <div class='result-main {txt_cls}'>{label}</div>
            <div class='result-sub'>Predicted Final Score: <b>{avg_score:.1f} / 100</b></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-lbl'>Model Breakdown</div>", unsafe_allow_html=True)
        bd = pd.DataFrame({
            "Model":      ["Logistic Regression", "Naive Bayes", "Ridge Regression", "Lasso Regression"],
            "Output":     [
                "Pass" if lr_pred else "Fail",
                "Pass" if nb_pred else "Fail",
                f"{rid_pred:.1f}",
                f"{las_pred:.1f}"
            ],
            "Type":       ["Classification", "Classification", "Regression", "Regression"],
        })
        st.dataframe(bd, use_container_width=True, hide_index=True)