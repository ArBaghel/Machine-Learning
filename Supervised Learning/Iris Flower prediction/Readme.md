# Iris Flower Predictor

A machine learning web application that classifies Iris flower species based on sepal and petal measurements. Built with Streamlit and powered by a Support Vector Machine (SVM) model trained using scikit-learn.

---

## Live Demo

[View Deployed Application](https://machine-learning-iris-flower-prediction.streamlit.app)

---

## Repository

[ArBaghel/Machine-Learning — Iris Flower Prediction](https://github.com/ArBaghel/Machine-Learning/tree/main/Supervised%20Learning/Iris%20Flower%20prediction)

---

## Project Overview

This project demonstrates a complete machine learning pipeline - from data preprocessing and model selection to deployment as an interactive web application. The model was selected through a GridSearchCV comparison across four algorithms, with SVM achieving the highest cross-validation accuracy.

---

## Features

- Real-time species prediction based on four input measurements
- Confidence score derived from the SVM decision function
- Visual confidence breakdown for all three Iris classes
- Model information panel displaying algorithm, accuracy, and best parameters
- Input summary with live values
- Fully responsive two-column layout

---

## Dataset

- **Source:** UCI Machine Learning Repository - Iris Dataset
- **Samples:** 150
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Features:** Sepal Length, Sepal Width, Petal Length, Petal Width

---

## Model

| Property         | Detail                          |
|------------------|---------------------------------|
| Algorithm        | Support Vector Classifier (SVC) |
| Kernel           | Linear                          |
| Regularization C | 1                               |
| Selection Method | GridSearchCV (5-fold CV)        |
| Test Accuracy    | 98%                             |

### Algorithms Compared

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Random Forest Classifier

GridSearchCV evaluated all algorithms and hyperparameter combinations. SVC with a linear kernel and C=1 was selected as the best estimator.

---

## Project Structure

```
machine-learning/
├── requirements.txt               
└── Supervised Learning/
    └── Iris Flower prediction/
        ├── app.py
        └── iris_model.pkl      
```

---

## Installation and Usage

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:

```bash
git clone https://github.com/ArBaghel/Machine-Learning.git
cd "Machine-Learning/Supervised Learning/Iris Flower prediction"
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

---

## Dependencies

| Package       | Purpose                            |
|---------------|------------------------------------|
| streamlit     | Web application framework          |
| scikit-learn  | Machine learning and Iris dataset  |
| numpy         | Numerical computation              |
| pandas        | Data manipulation (training)       |
| seaborn       | Data visualization (training)      |

---

## Training Pipeline

The model was trained in `Iris.ipynb` using the following steps:

1. Load the Iris dataset via Seaborn
2. Encode target labels using LabelEncoder
3. Split data into training and test sets (67% / 33%)
4. Define a Pipeline with a classifier placeholder
5. Run GridSearchCV across four algorithms and multiple hyperparameters
6. Evaluate the best estimator on the test set
7. Serialize the best pipeline using pickle

---

## Deployment

The application is deployed on **Streamlit Cloud**.

Key notes for deployment:
- `requirements.txt` must be placed at the **root of the repository**
- The pickle file path uses `os.path.dirname(__file__)` to ensure correct resolution on the cloud server
- The Iris dataset is loaded directly from `sklearn.datasets` — no external data file required

---

## Author

**Ar Baghel**
GitHub: [ArBaghel](https://github.com/ArBaghel)

---

