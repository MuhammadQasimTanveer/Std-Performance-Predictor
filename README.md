# Student Performance Predictor

A web-based ML app that predicts student pass/fail and expected score using 4 supervised learning algorithms.

---

## Features

- Data preprocessing with `StandardScaler`
- Feature selection using `SelectKBest` (top 8 of 10)
- Real-time prediction from user input
- Model comparison with Accuracy, MSE, R² metrics

## Models Used

| Task | Algorithm |
|------|-----------|
| Classification | Logistic Regression, Naive Bayes |
| Regression | Ridge Regression, Lasso Regression |


## Setup

```bash
python -m venv virtualenv
virtualenv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
click & run the analysis.ipynb(run each cell)
streamlit run app.py
```

Open browser at **http://localhost:8501**