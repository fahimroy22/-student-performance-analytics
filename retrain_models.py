import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/student_performance.csv")

features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed",
]

X = df[features]


# ==================================================
# 2. LINEAR REGRESSION
# ==================================================

y_reg = df["exam_score"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X,
    y_reg,
    test_size=0.20,
    random_state=42
)

linear_pipeline = Pipeline([
    (
        "preprocessing",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    ),
    ("model", LinearRegression())
])

linear_pipeline.fit(X_train_reg, y_train_reg)

reg_predictions = linear_pipeline.predict(X_test_reg)

mae = mean_absolute_error(y_test_reg, reg_predictions)
rmse = mean_squared_error(y_test_reg, reg_predictions) ** 0.5
r2 = r2_score(y_test_reg, reg_predictions)

joblib.dump(
    linear_pipeline,
    "models/linear_pipeline.pkl"
)


# ==================================================
# 3. BALANCED LOGISTIC REGRESSION
# ==================================================

y_cls = df["pass_status"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X,
    y_cls,
    test_size=0.20,
    random_state=42,
    stratify=y_cls
)

logistic_pipeline = Pipeline([
    (
        "preprocessing",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
    ),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )
])

logistic_pipeline.fit(X_train_cls, y_train_cls)

cls_predictions = logistic_pipeline.predict(X_test_cls)
cls_probabilities = logistic_pipeline.predict_proba(X_test_cls)

# Find the probability column corresponding to "Pass"
pass_index = list(
    logistic_pipeline.named_steps["model"].classes_
).index("Pass")

pass_probabilities = cls_probabilities[:, pass_index]

accuracy = accuracy_score(y_test_cls, cls_predictions)
precision = precision_score(
    y_test_cls,
    cls_predictions,
    pos_label="Pass"
)
recall = recall_score(
    y_test_cls,
    cls_predictions,
    pos_label="Pass"
)
f1 = f1_score(
    y_test_cls,
    cls_predictions,
    pos_label="Pass"
)
balanced_accuracy = balanced_accuracy_score(
    y_test_cls,
    cls_predictions
)

y_binary = (y_test_cls == "Pass").astype(int)

auc = roc_auc_score(
    y_binary,
    pass_probabilities
)

joblib.dump(
    logistic_pipeline,
    "models/logistic_pipeline.pkl"
)


# ==================================================
# 4. Display results
# ==================================================

print("\nModels retrained successfully.")
print("\nLinear Regression")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")

print("\nBalanced Logistic Regression")
print(f"Accuracy:          {accuracy:.3%}")
print(f"Precision:         {precision:.3%}")
print(f"Recall:            {recall:.3%}")
print(f"F1 Score:          {f1:.3%}")
print(f"Balanced Accuracy: {balanced_accuracy:.3%}")
print(f"AUC:               {auc:.3f}")

print(
    "\nscikit-learn compatible model files saved to models/"
)