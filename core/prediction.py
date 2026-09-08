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

    # --------------------------------------------------------
    # PREPARE USER INPUT
    # --------------------------------------------------------

    student_df = pd.DataFrame(
        [input_data],
        columns=FEATURES
    )


    # --------------------------------------------------------
    # LOAD TRAINED MODELS
    # --------------------------------------------------------

    linear_model = load_linear_model()
    logistic_model = load_logistic_model()


    # --------------------------------------------------------
    # LINEAR REGRESSION PREDICTION
    # --------------------------------------------------------

    predicted_score = linear_model.predict(
        student_df
    )[0]

    # Keep displayed score within 0–100
    predicted_score = max(
        0,
        min(
            100,
            float(predicted_score)
        )
    )


    # --------------------------------------------------------
    # LOGISTIC REGRESSION PREDICTION
    # --------------------------------------------------------

    predicted_class = logistic_model.predict(
        student_df
    )[0]

    # Model classes are stored as:
    # ["Fail", "Pass"]
    #
    # Find the probability column corresponding
    # specifically to "Pass" rather than assuming
    # it is always column index 1.

    model_classes = list(
        logistic_model.named_steps[
            "model"
        ].classes_
    )

    pass_index = model_classes.index(
        "Pass"
    )

    probability_values = (
        logistic_model.predict_proba(
            student_df
        )[0]
    )

    pass_probability = (
        probability_values[
            pass_index
        ]
    )


    # --------------------------------------------------------
    # PREDICTED STATUS
    # --------------------------------------------------------

    # The trained model already returns
    # "Pass" or "Fail".

    predicted_status = str(
        predicted_class
    )


    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {
        "predicted_score":
            float(predicted_score),

        "predicted_status":
            predicted_status,

        "pass_probability":
            float(pass_probability)
    }