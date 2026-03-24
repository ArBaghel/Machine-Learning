# 🎓 Graduate Admission Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://arbaghel-machine-learning-supervised-learninggraduate-admi-xyze4s.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A machine learning web application that predicts whether a graduate school applicant is likely to be **admitted or rejected** based on their academic profile. Built with scikit-learn and deployed via Streamlit.

---

## 🚀 Live Demo

🔗 **[Graduate Admission Predictor · Streamlit](https://arbaghel-machine-learning-supervised-learninggraduate-admi-xyze4s.streamlit.app/)**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [ML Pipeline](#ml-pipeline)
- [Model Selection & GridSearchCV](#model-selection--gridsearchcv)
- [Best Model & Hyperparameters](#best-model--hyperparameters)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [App Screenshots](#app-screenshots)
- [Tech Stack](#tech-stack)

---

## Overview

Graduate school admission is a competitive, multi-dimensional process. This project builds a **binary classification model** that predicts whether a student has a high chance of admission (≥ 75% probability) or not, based on standardized test scores, academic records, and application materials.

The model was trained using an automated **Pipeline + GridSearchCV** approach that evaluated four different classifiers and selected the best-performing one through rigorous cross-validation.

---

## Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — Graduate Admissions Dataset](https://www.kaggle.com/datasets/mohansacharya/graduate-admissions) |
| **Records** | 400 applicants |
| **Features** | 8 input features + 1 target |
| **Missing Values** | None |
| **Duplicates** | None |

### Feature Statistics

| Feature | Min | Mean | Max | Description |
|---|---|---|---|---|
| GRE Score | 290 | 316.8 | 340 | Graduate Record Examination total score |
| TOEFL Score | 92 | 107.4 | 120 | Test of English as a Foreign Language score |
| University Rating | 1 | 3.09 | 5 | Undergraduate university prestige (1–5) |
| SOP | 1.0 | 3.40 | 5.0 | Statement of Purpose strength (1–5) |
| LOR | 1.0 | 3.45 | 5.0 | Letter of Recommendation strength (1–5) |
| CGPA | 6.80 | 8.60 | 9.92 | Undergraduate CGPA (out of 10) |
| Research | 0 | 0.55 | 1 | Research experience (0 = No, 1 = Yes) |

### Target Variable

The original dataset provides a continuous `Chance of Admit` (0.34–0.97). This was **binarized** using a threshold of **0.75**:

```python
df['Admit'] = (df['Chance of Admit'] >= 0.75).astype(int)
# 0 → Rejected  |  1 → Admitted
```

---

## Features

The Streamlit app provides:

- 🎛️ **Interactive input controls** — sliders, select sliders, and radio buttons for all 8 features
- 🤖 **Model info banner** — displays algorithm name, hyperparameters, search strategy, and accuracy
- ✅ **Admission verdict** — clear Admitted / Rejected result card
- 📊 **Dual confidence meters** — visual probability bars for both admission and rejection
- 📈 **Stats panel** — admit probability %, confidence level, model accuracy, algorithm name
- 💡 **Personalised improvement tips** — actionable suggestions when profile is weak

---

## ML Pipeline

The project uses a **scikit-learn Pipeline** to encapsulate the classifier, making the workflow clean, reproducible, and easy to swap algorithms:

```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('classifier', LogisticRegression())   # placeholder — swapped by GridSearchCV
])
```

### Why use a Pipeline?

- Prevents data leakage between training and validation folds
- Makes the entire estimator serialisable as a single `.pkl` object
- Allows GridSearchCV to swap the classifier itself as a hyperparameter

---

## Model Selection & GridSearchCV

Four classifiers were evaluated simultaneously using `GridSearchCV` with **5-fold cross-validation**, optimising for **accuracy**:

```python
Search_space = [
    {
        'classifier': [LogisticRegression(max_iter=500)],
        'classifier__C': [0.1, 1, 10],
        'classifier__solver': ['liblinear', 'lbfgs']
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 10],
        'classifier__kernel': ['linear', 'rbf']
    },
    {
        'classifier': [DecisionTreeClassifier()],
        'classifier__max_depth': [3, 5, None],
        'classifier__criterion': ['gini', 'entropy']
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, None]
    }
]

grid = GridSearchCV(pipeline, Search_space, cv=5, scoring='accuracy')
grid.fit(x_train, y_train)
```

### Search Space Summary

| Algorithm | Hyperparameters Searched | Combinations |
|---|---|---|
| Logistic Regression | C ∈ {0.1, 1, 10}, solver ∈ {liblinear, lbfgs} | 6 |
| Support Vector Classifier | C ∈ {0.1, 1, 10}, kernel ∈ {linear, rbf} | 6 |
| Decision Tree | max_depth ∈ {3, 5, None}, criterion ∈ {gini, entropy} | 6 |
| Random Forest | n_estimators ∈ {50, 100, 200}, max_depth ∈ {3, 5, None} | 9 |

**Total configurations evaluated:** 27 × 5 folds = **135 model fits**

---

## Best Model & Hyperparameters

GridSearchCV selected **Logistic Regression** as the best estimator:

```
Best Algorithm  : LogisticRegression
Best Parameters : C=1, solver=lbfgs, penalty=l2, max_iter=500
CV Accuracy     : 91.25%
```

### Why Logistic Regression won

Logistic Regression outperformed more complex models because:
- The dataset (400 records, 7 features) is relatively small — tree-based models overfit
- The features have roughly linear relationships with the admission outcome
- L2 regularisation (C=1) provided the right bias-variance balance
- It naturally outputs calibrated probability scores, ideal for confidence meters

The best estimator was serialised and saved:

```python
import pickle
with open('admission_model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
```

---

## Model Performance

| Metric | Value |
|---|---|
| **Cross-Validation Accuracy (5-fold)** | **91.25%** |
| **Test Set Accuracy** | ~91% |
| Train/Test Split | 80% / 20% (random_state=42) |
| Evaluation Metric | Accuracy |

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Rejected (0) | High | High | High |
| Admitted (1) | High | High | High |

> The model performs consistently across both classes due to the balanced nature of the binarised target at the 0.75 threshold.

---

## Project Structure

```
Machine-Learning/
├── requirements.txt                          # Python dependencies (root level)
└── Supervised Learning/
    ├── Graduate Admission/
    │   ├── app.py                            # Streamlit web application
    │   ├── admission_model.pkl               # Serialised best estimator
    │   ├── Graduate Admission.ipynb          # Full training notebook
    │   ├── Admission_Predict.csv             # Dataset
    │   └── README.md                         # This file
    └── Iris Flower prediction/
        ├── app.py
        ├── iris_model.pkl
        └── Iris.ipynb
```

---

## Installation & Usage

### Prerequisites

- Python 3.10+
- pip

### Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/ArBaghel/Machine-Learning.git
cd "Machine-Learning/Supervised Learning/Graduate Admission"

# 2. Install dependencies
pip install -r ../../requirements.txt

# 3. Launch the app
streamlit run app.py
```

### requirements.txt

```
streamlit
scikit-learn==1.6.1
numpy
pandas
seaborn
```

> ⚠️ The `scikit-learn==1.6.1` version pin is required — the model was serialised with this version. Using a newer version may cause unpickling warnings or errors.

### Streamlit Cloud Deploy Settings

| Field | Value |
|---|---|
| Repository | `ArBaghel/Machine-Learning` |
| Branch | `main` |
| Main file path | `Supervised Learning/Graduate Admission/app.py` |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **pandas** | Data loading and preprocessing |
| **scikit-learn** | ML pipeline, GridSearchCV, classifiers |
| **pickle** | Model serialisation |
| **Streamlit** | Interactive web UI |
| **missingno** | Missing value visualisation (training only) |

---

## Author

**ArBaghel**
🔗 [GitHub Profile](https://github.com/ArBaghel)

---

*Built as part of a supervised learning portfolio — exploring automated model selection via Pipeline + GridSearchCV.*
