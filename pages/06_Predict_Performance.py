# ============================================================
# PREDICT PERFORMANCE PAGE
# ============================================================

import random
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from components.styles import apply_global_styles
from components.footer import show_footer
from components.icons import (
    page_title,
    section_title,
    icon_card
)

from core.prediction import predict_student_performance


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Predict Performance",
    page_icon="🎓",
    layout="wide"
)

apply_global_styles()


# ------------------------------------------------------------
# SESSION STATE DEFAULTS
# ------------------------------------------------------------

DEFAULT_VALUES = {
    "previous_exam_score": 70.0,
    "previous_gpa": 3.0,
    "attendance_percentage": 85.0,
    "assignment_completion_rate": 75.0,
    "study_hours_per_day": 3.0,
    "practice_tests_completed": 6,
}


for key, value in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ------------------------------------------------------------
# SAMPLE / RESET FUNCTIONS
# ------------------------------------------------------------

def generate_sample():

    st.session_state.previous_exam_score = float(
        random.randint(40, 95)
    )

    st.session_state.previous_gpa = round(
        random.uniform(1.5, 4.0),
        1
    )

    st.session_state.attendance_percentage = float(
        random.randint(50, 100)
    )

    st.session_state.assignment_completion_rate = float(
        random.randint(40, 100)
    )

    st.session_state.study_hours_per_day = round(
        random.uniform(0.5, 8.0),
        1
    )

    st.session_state.practice_tests_completed = (
        random.randint(0, 17)
    )


def reset_inputs():

    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

page_title(
    "sparkles",
    "Student Performance Predictor",
    "Enter six student academic variables to estimate exam score, "
    "Pass / Fail status, and pass probability."
)


# ------------------------------------------------------------
# QUICK ACTIONS
# ------------------------------------------------------------

section_title(
    "sliders",
    "Quick Actions"
)


button_col1, button_col2, _ = st.columns(
    [1, 1, 3]
)


with button_col1:

    st.button(
        "Generate Sample",
        on_click=generate_sample,
        use_container_width=True
    )


with button_col2:

    st.button(
        "Reset Inputs",
        on_click=reset_inputs,
        use_container_width=True
    )


# ------------------------------------------------------------
# STUDENT INFORMATION
# ------------------------------------------------------------

section_title(
    "filter",
    "Student Academic Information"
)


col1, col2 = st.columns(2)


# ------------------------------------------------------------
# LEFT COLUMN
# ------------------------------------------------------------

with col1:

    previous_exam_score = st.number_input(
        "Previous Exam Score · 0–100",
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        key="previous_exam_score"
    )

    st.caption(
        f"{previous_exam_score:.0f} / 100"
    )


    previous_gpa = st.number_input(
        "Previous GPA · 0–4.0",
        min_value=0.0,
        max_value=4.0,
        step=0.1,
        key="previous_gpa"
    )

    st.caption(
        f"{previous_gpa:.2f} / 4.00"
    )


    attendance_percentage = st.number_input(
        "Attendance · 0–100%",
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        key="attendance_percentage"
    )

    st.caption(
        f"{attendance_percentage:.0f} / 100"
    )


# ------------------------------------------------------------
# RIGHT COLUMN
# ------------------------------------------------------------

with col2:

    assignment_completion_rate = st.number_input(
        "Assignment Completion · 0–100%",
        min_value=0.0,
        max_value=100.0,
        step=1.0,
        key="assignment_completion_rate"
    )

    st.caption(
        f"{assignment_completion_rate:.0f} / 100"
    )


    study_hours_per_day = st.number_input(
        "Study Hours per Day · 0–12",
        min_value=0.0,
        max_value=12.0,
        step=0.5,
        key="study_hours_per_day"
    )

    st.caption(
        f"{study_hours_per_day:.1f} / 12 hours"
    )


    practice_tests_completed = st.number_input(
        "Practice Tests Completed · 0–17",
        min_value=0,
        max_value=17,
        step=1,
        key="practice_tests_completed"
    )

    st.caption(
        f"{practice_tests_completed} / 17"
    )


# ------------------------------------------------------------
# STUDENT INPUT PROFILE
# ------------------------------------------------------------

section_title(
    "chart",
    "Student Input Profile"
)


profile_df = pd.DataFrame({

    "Feature": [
        "Previous Exam",
        "GPA",
        "Attendance",
        "Assignments",
        "Study Hours",
        "Practice Tests"
    ],

    "Percentage": [
        previous_exam_score,
        (previous_gpa / 4.0) * 100,
        attendance_percentage,
        assignment_completion_rate,
        (study_hours_per_day / 12.0) * 100,
        (practice_tests_completed / 17.0) * 100
    ]
})


fig_profile = px.bar(
    profile_df,
    x="Percentage",
    y="Feature",
    orientation="h",
    text="Percentage"
)


fig_profile.update_traces(
    texttemplate="%{text:.0f}%",
    textposition="outside"
)


fig_profile.update_layout(
    height=330,
    margin=dict(
        l=10,
        r=40,
        t=10,
        b=20
    ),
    xaxis=dict(
        range=[0, 105],
        title="Normalized Input Level (%)"
    ),
    yaxis_title="",
    showlegend=False
)


st.plotly_chart(
    fig_profile,
    use_container_width=True,
    config={
        "displayModeBar": False
    }
)


# ------------------------------------------------------------
# PREDICTION ACTION
# ------------------------------------------------------------

section_title(
    "sparkles",
    "Generate Prediction"
)


predict_button = st.button(
    "Predict Performance",
    type="primary",
    use_container_width=True
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

if predict_button:

    input_data = {

        "previous_exam_score":
            previous_exam_score,

        "previous_gpa":
            previous_gpa,

        "attendance_percentage":
            attendance_percentage,

        "assignment_completion_rate":
            assignment_completion_rate,

        "study_hours_per_day":
            study_hours_per_day,

        "practice_tests_completed":
            practice_tests_completed
    }


    with st.spinner(
        "Analyzing student performance..."
    ):

        result = predict_student_performance(
            input_data
        )


    predicted_score = float(
        result["predicted_score"]
    )

    predicted_status = result[
        "predicted_status"
    ]

    pass_probability = float(
        result["pass_probability"]
    )

    probability_percent = (
        pass_probability * 100
    )


    # --------------------------------------------------------
    # RESULT SUMMARY
    # --------------------------------------------------------

    section_title(
        "target",
        "Prediction Results"
    )


    result_col1, result_col2, result_col3 = (
        st.columns(3)
    )


    with result_col1:

        with st.container(
            border=True
        ):

            st.metric(
                "Predicted Exam Score",
                f"{predicted_score:.1f} / 100"
            )


    with result_col2:

        with st.container(
            border=True
        ):

            st.metric(
                "Predicted Status",
                predicted_status
            )


    with result_col3:

        with st.container(
            border=True
        ):

            st.metric(
                "Pass Probability",
                f"{probability_percent:.1f}%"
            )


    # --------------------------------------------------------
    # VISUAL RESULT CARDS
    # --------------------------------------------------------

    summary1, summary2, summary3 = (
        st.columns(3)
    )


    with summary1:

        icon_card(
            "trending",
            "Exam Score",
            f"Estimated score: {predicted_score:.1f} out of 100."
        )


    with summary2:

        icon_card(
            "target",
            "Predicted Outcome",
            f"The classification model predicts {predicted_status}."
        )


    with summary3:

        icon_card(
            "scale",
            "Pass Probability",
            f"Estimated probability of passing: "
            f"{probability_percent:.1f}%."
        )


    # --------------------------------------------------------
    # GAUGES
    # --------------------------------------------------------

    section_title(
        "chart",
        "Prediction Gauges"
    )


    gauge_col1, gauge_col2 = st.columns(2)


    # --------------------------------------------------------
    # SCORE GAUGE
    # --------------------------------------------------------

    with gauge_col1:

        fig_score = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=predicted_score,

                number={
                    "font": {
                        "size": 42
                    }
                },

                title={
                    "text":
                        "Predicted Exam Score"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#355C7D"
                    },

                    "bgcolor":
                        "#F3F4F6"
                }
            )
        )


        fig_score.update_layout(

            height=280,

            margin=dict(
                l=30,
                r=30,
                t=60,
                b=10
            )
        )


        st.plotly_chart(
            fig_score,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # --------------------------------------------------------
    # PROBABILITY GAUGE
    # --------------------------------------------------------

    with gauge_col2:

        fig_probability = go.Figure(

            go.Indicator(

                mode="gauge+number",

                value=probability_percent,

                number={
                    "suffix": "%",
                    "font": {
                        "size": 42
                    }
                },

                title={
                    "text":
                        "Pass Probability"
                },

                gauge={

                    "axis": {
                        "range": [0, 100]
                    },

                    "bar": {
                        "color": "#355C7D"
                    },

                    "bgcolor":
                        "#F3F4F6"
                }
            )
        )


        fig_probability.update_layout(

            height=280,

            margin=dict(
                l=30,
                r=30,
                t=60,
                b=10
            )
        )


        st.plotly_chart(
            fig_probability,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    # --------------------------------------------------------
    # OUTCOME MESSAGE
    # --------------------------------------------------------

    if predicted_status.lower() == "pass":

        st.success(
            f"Predicted outcome: PASS "
            f"({probability_percent:.1f}% probability)"
        )

    else:

        st.error(
            f"Predicted outcome: FAIL "
            f"({probability_percent:.1f}% pass probability)"
        )


    # --------------------------------------------------------
    # EXPECTED SCORE RANGE
    # --------------------------------------------------------

    section_title(
        "trending",
        "Expected Score Range"
    )


    MODEL_MAE = 9.38


    lower_score = max(
        0,
        predicted_score - MODEL_MAE
    )

    upper_score = min(
        100,
        predicted_score + MODEL_MAE
    )


    fig_range = go.Figure()


    # --------------------------------------------------------
    # ERROR RANGE
    # --------------------------------------------------------

    fig_range.add_trace(

        go.Scatter(

            x=[
                lower_score,
                upper_score
            ],

            y=[1, 1],

            mode="lines",

            line=dict(
                width=14,
                color="#355C7D"
            ),

            hovertemplate=(
                f"Approximate range: "
                f"{lower_score:.1f}–"
                f"{upper_score:.1f}"
                "<extra></extra>"
            ),

            showlegend=False
        )
    )


    # --------------------------------------------------------
    # PREDICTED SCORE POINT
    # --------------------------------------------------------

    fig_range.add_trace(

        go.Scatter(

            x=[predicted_score],

            y=[1],

            mode="markers+text",

            marker=dict(
                size=18,
                color="#7DB9E8"
            ),

            text=[
                f"{predicted_score:.1f}"
            ],

            textposition="top center",

            hovertemplate=(
                f"Predicted Score: "
                f"{predicted_score:.1f}"
                "<extra></extra>"
            ),

            showlegend=False
        )
    )


    fig_range.update_layout(

        height=220,

        margin=dict(
            l=20,
            r=20,
            t=30,
            b=30
        ),

        xaxis=dict(
            range=[0, 100],
            title="Exam Score"
        ),

        yaxis=dict(
            visible=False,
            range=[0.7, 1.3]
        ),

        showlegend=False
    )


    st.plotly_chart(
        fig_range,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    st.caption(
        f"Approximate MAE-based range: "
        f"{lower_score:.1f}–{upper_score:.1f}. "
        f"This is not a statistical confidence interval."
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

show_footer()