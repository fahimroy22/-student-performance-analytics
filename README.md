# Student Performance Analytics

Student Performance Analytics is an interactive machine-learning web application built with Streamlit. The project analyzes student academic data and uses Linear Regression and Balanced Logistic Regression to estimate student performance.

The application presents the complete machine-learning workflow, including exploratory data analysis, preprocessing, model evaluation, comparison, and interactive prediction.

---

## Project Objective

The main objective of this project is to investigate whether pre-exam academic and study-related information can be used to predict student performance.

The application performs two prediction tasks:

1. **Linear Regression** predicts the student's expected numerical exam score.
2. **Balanced Logistic Regression** predicts whether the student is likely to Pass or Fail and provides a pass probability.

---

## Dataset

The project uses the **Student Exam Performance & Success Dataset**.

The working dataset contains:

- **100,000 student records**
- **44 variables**
- Academic, behavioral, demographic, and study-related information
- Numerical exam scores
- Pass/Fail outcomes

The dataset contains some missing values but no duplicate records.

---

## Selected Prediction Features

Six pre-exam variables are used by both machine-learning models:

1. Previous Exam Score
2. Previous GPA
3. Attendance Percentage
4. Assignment Completion Rate
5. Study Hours per Day
6. Practice Tests Completed

These variables are available before the final exam and therefore allow the models to estimate future exam performance without directly using the target exam score as an input.

---

## Prediction Targets

### Linear Regression

**Target:** `exam_score`

The model predicts a numerical exam score on a 0–100 scale.

### Balanced Logistic Regression

**Target:** `pass_status`

The model predicts:

- Pass
- Fail
- Pass probability

---

## Data Preprocessing

The preprocessing workflow includes:

**Raw Data → Feature Selection → Missing-Value Imputation → Train/Test Split → Standardization → Model Training**

### Missing Values

Missing numerical values are handled using **median imputation**.

### Train/Test Split

The dataset is divided into:

- **80% Training Data**
- **20% Testing Data**

A random state of `42` is used for reproducibility.

For classification, the split is stratified to preserve the Pass/Fail class distribution.

### Feature Standardization

Numerical predictors are standardized before model training using `StandardScaler`.

The preprocessing steps are included inside Scikit-learn pipelines to reduce data leakage and ensure that the same transformations are applied during prediction.

---

## Linear Regression

Linear Regression is used to predict the student's numerical exam score.

### Model Performance

| Metric | Result |
|---|---:|
| MAE | 9.38 |
| RMSE | 11.76 |
| R² | 0.444 |

The model has an average absolute prediction error of approximately **9.38 exam-score points**.

An R² of **0.444** indicates that approximately **44.4% of the variation in exam scores** is explained by the six selected predictors.

The model provides a useful estimate of student performance, although a substantial amount of variation remains unexplained.

---

## Logistic Regression

The classification component predicts whether a student is likely to Pass or Fail.

Because the dataset contains substantially more passing students than failing students, class imbalance is an important consideration.

Three classification approaches are compared:

- Dummy Baseline
- Standard Logistic Regression
- Balanced Logistic Regression

### Final Balanced Logistic Regression Performance

| Metric | Result |
|---|---:|
| Accuracy | 72.5% |
| Precision | 90.6% |
| Recall | 71.9% |
| F1 Score | 80.1% |
| Balanced Accuracy | 73.2% |
| ROC AUC | 0.814 |

---

## Classification Model Comparison

| Model | Accuracy | Balanced Accuracy |
|---|---:|---:|
| Dummy Baseline | 77.4% | 50.0% |
| Standard Logistic Regression | 80.9% | 64.8% |
| Balanced Logistic Regression | 72.5% | 73.2% |

Although the Dummy Baseline achieves relatively high ordinary accuracy because most students pass, its balanced accuracy is only **50%**.

The **Balanced Logistic Regression** model is selected for the final application because it provides the strongest balanced accuracy and performs more evenly across the Pass and Fail classes.

---

## Interactive Prediction

The prediction page allows users to enter:

- Previous Exam Score
- Previous GPA
- Attendance Percentage
- Assignment Completion Rate
- Study Hours per Day
- Practice Tests Completed

The application then returns:

- **Predicted Exam Score**
- **Predicted Pass/Fail Status**
- **Pass Probability**
- **Approximate MAE-based Score Range**

The displayed score range is based on the Linear Regression model's MAE and should not be interpreted as a statistical confidence interval.

---

## Application Pages

The Streamlit application contains seven main pages:

### Dashboard
Provides a summary of the dataset, important relationships, model performance, and the prediction workflow.

### Dataset & EDA
Explores dataset characteristics, distributions, relationships, correlations, missing values, and Pass/Fail class distribution.

### Preprocessing
Explains feature selection, missing-value handling, train/test splitting, standardization, and model pipelines.

### Linear Regression
Presents regression performance, actual-versus-predicted scores, residual analysis, feature coefficients, and model interpretation.

### Logistic Regression
Presents classification metrics, class distribution, confusion matrix, ROC curve, baseline comparison, and balanced-model evaluation.

### Model Comparison
Compares the roles and performance of Linear Regression and Balanced Logistic Regression.

### Predict Performance
Provides an interactive interface for entering student information and generating predictions from the trained models.

---

## Project Structure

```text
student-performance-app/
│
├── Dashboard.py
│
├── requirements.txt
├── README.md
│
├── data/
│   └── student_performance.csv
│
├── models/
│   ├── linear_pipeline.pkl
│   └── logistic_pipeline.pkl
│
├── pages/
│   ├── 01_Dataset_EDA.py
│   ├── 02_Preprocessing.py
│   ├── 03_Linear_Regression.py
│   ├── 04_Logistic_Regression.py
│   ├── 05_Model_Comparison.py
│   └── 06_Predict_Performance.py
│
├── components/
│   ├── styles.py
│   ├── icons.py
│   └── footer.py
│
└── core/
    ├── model_loader.py
    └── prediction.py