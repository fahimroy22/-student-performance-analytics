# ============================================================
# PREDICTION SERVICE
# ============================================================

import pandas as pd

from core.model_loader import (
    load_linear_model,
    load_logistic_model
)


FEATURES = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed"
]


def predict_student_performance(input_data):
    """
    Predict numerical exam score and Pass/Fail status.

    input_data should be a dictionary containing the
    six model features.
    """

    # Convert user input into a one-row DataFrame
    student_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )

    # Load trained pipelines
    linear_model = load_linear_model()
    logistic_model = load_logistic_model()

    # Linear Regression prediction
    predicted_score = linear_model.predict(
        student_df
    )[0]

    # Keep displayed score within a sensible 0–100 range
    predicted_score = max(
        0,
        min(100, predicted_score)
    )

    # Logistic Regression prediction
    predicted_class = logistic_model.predict(
        student_df
    )[0]

    pass_probability = logistic_model.predict_proba(
        student_df
    )[0, 1]

    predicted_status = (
        "Pass" if predicted_class == 1 else "Fail"
    )

    return {
        "predicted_score": float(predicted_score),
        "predicted_status": predicted_status,
        "pass_probability": float(pass_probability)
    }