# Student Performance Predictor

A web-based ML app that predicts student pass/fail and expected score using 4 supervised learning algorithms.

---

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open browser at **http://localhost:8501**

---

## Files

```
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── student_performance.csv   # Dataset (400 students, 10 features)
└── README.md
```

---

## Pages

| Page | Description |
|------|-------------|
| Dashboard | Model accuracy, R² scores, evaluation table, best model highlight |
| Predict | Enter student data → get Pass/Fail + predicted score from all 4 models |
| Dataset | First 20 rows + correlation heatmap |

---

## Models Used

| Task | Algorithm |
|------|-----------|
| Classification | Logistic Regression, Naive Bayes |
| Regression | Ridge Regression, Lasso Regression |

---

## Features

- Data preprocessing with `StandardScaler`
- Feature selection using `SelectKBest` (top 8 of 10)
- Real-time prediction from user input
- Model comparison with Accuracy, MSE, R² metrics