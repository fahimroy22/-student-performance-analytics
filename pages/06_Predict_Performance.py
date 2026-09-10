# ============================================================
# PREDICT PERFORMANCE PAGE
# ============================================================

import random
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from components.styles import (
    apply_global_styles,
    get_plotly_template,
    get_theme_colors,
)

from components.footer import show_footer

from components.icons import (
    page_title,
    section_title,
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
# THEME SETTINGS
# ------------------------------------------------------------

plotly_template = get_plotly_template()
theme = get_theme_colors()

background = theme["background"]
surface = theme["surface"]
text_color = theme["text"]
muted = theme["muted"]
border = theme["border"]
accent = theme["accent"]


# ------------------------------------------------------------
# CUSTOM PAGE STYLES
# ------------------------------------------------------------

st.markdown(
    f"""
    <style>

    .prediction-hero {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 125px;
        margin-bottom: 6px;
    }}

    .prediction-hero-label {{
        color: {muted};
        font-size: 0.78rem;
        margin-bottom: 6px;
    }}

    .prediction-hero-value {{
        color: {text_color};
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.15;
    }}

    .prediction-hero-note {{
        color: {muted};
        font-size: 0.80rem;
        margin-top: 8px;
    }}

    .result-pass {{
        color: #5FA879 !important;
    }}

    .result-fail {{
        color: #C86B6B !important;
    }}

    .interpretation-box {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 16px 18px;
        margin-top: 6px;
    }}

    .interpretation-title {{
        color: {text_color};
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 7px;
    }}

    .interpretation-text {{
        color: {muted};
        font-size: 0.88rem;
        line-height: 1.55;
    }}

    .threshold-card {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 15px 18px;
        margin-top: 5px;
        margin-bottom: 8px;
    }}

    .threshold-label {{
        color: {muted};
        font-size: 0.78rem;
        margin-bottom: 4px;
    }}

    .threshold-value {{
        color: {text_color};
        font-size: 1.1rem;
        font-weight: 650;
    }}

    .threshold-note {{
        color: {muted};
        font-size: 0.82rem;
        margin-top: 6px;
        line-height: 1.45;
    }}

    .scenario-card {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 15px 18px;
        min-height: 116px;
    }}

    .scenario-label {{
        color: {muted};
        font-size: 0.78rem;
        margin-bottom: 6px;
    }}

    .scenario-value {{
        color: {text_color};
        font-size: 1.45rem;
        font-weight: 700;
    }}

    .scenario-note {{
        color: {muted};
        font-size: 0.80rem;
        margin-top: 6px;
    }}

    .positive-change {{
        color: #5FA879;
        font-weight: 600;
    }}

    .negative-change {{
        color: #C86B6B;
        font-weight: 600;
    }}

    .neutral-change {{
        color: {muted};
        font-weight: 600;
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------

MODEL_MAE = 9.38
CLASSIFICATION_BALANCED_ACCURACY = 73.2
CLASSIFICATION_AUC = 0.814


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


if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


if "prediction_inputs" not in st.session_state:
    st.session_state.prediction_inputs = None


if "scenario_result" not in st.session_state:
    st.session_state.scenario_result = None


# ------------------------------------------------------------
# RESET STORED RESULTS
# ------------------------------------------------------------

def clear_results():

    st.session_state.prediction_result = None
    st.session_state.prediction_inputs = None
    st.session_state.scenario_result = None

    scenario_keys = [
        "sim_attendance",
        "sim_assignments",
        "sim_study_hours",
        "sim_practice_tests",
    ]

    for key in scenario_keys:
        st.session_state.pop(
            key,
            None
        )


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

    clear_results()


def generate_at_risk_sample():

    st.session_state.previous_exam_score = 42.0
    st.session_state.previous_gpa = 1.8
    st.session_state.attendance_percentage = 58.0
    st.session_state.assignment_completion_rate = 48.0
    st.session_state.study_hours_per_day = 1.0
    st.session_state.practice_tests_completed = 1

    clear_results()


def reset_inputs():

    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value

    clear_results()


# ------------------------------------------------------------
# PLOTLY STYLE HELPER
# ------------------------------------------------------------

def style_chart(fig):

    fig.update_layout(
        template=plotly_template,
        paper_bgcolor=background,
        plot_bgcolor=background,
        font=dict(
            color=text_color
        )
    )

    fig.update_xaxes(
        color=text_color,
        gridcolor=border,
        zerolinecolor=border
    )

    fig.update_yaxes(
        color=text_color,
        gridcolor=border,
        zerolinecolor=border
    )

    return fig


# ------------------------------------------------------------
# FORMAT CHANGE
# ------------------------------------------------------------

def format_change(
    value,
    suffix=""
):

    if value > 0:
        css_class = "positive-change"
        sign = "+"

    elif value < 0:
        css_class = "negative-change"
        sign = ""

    else:
        css_class = "neutral-change"
        sign = ""

    return (
        css_class,
        f"{sign}{value:.1f}{suffix}"
    )


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


action1, action2, action3, _ = st.columns(
    [1, 1, 1.15, 2.5]
)


with action1:

    st.button(
        "Generate Sample",
        on_click=generate_sample,
        width="stretch"
    )


with action2:

    st.button(
        "Reset Inputs",
        on_click=reset_inputs,
        width="stretch"
    )


with action3:

    st.button(
        "At-Risk Example",
        on_click=generate_at_risk_sample,
        width="stretch"
    )


# ------------------------------------------------------------
# STUDENT INPUTS
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
# INPUT PROFILE
# ------------------------------------------------------------

section_title(
    "chart",
    "Student Input Profile"
)


profile_df = pd.DataFrame(
    {
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
    }
)


fig_profile = px.bar(
    profile_df,
    x="Percentage",
    y="Feature",
    orientation="h",
    text="Percentage"
)


fig_profile.update_traces(
    marker_color=accent,
    texttemplate="%{text:.0f}%",
    textposition="outside"
)


fig_profile.update_layout(
    height=245,

    margin=dict(
        l=10,
        r=45,
        t=5,
        b=20
    ),

    xaxis=dict(
        range=[0, 105],
        title="Normalized Input Level (%)"
    ),

    yaxis_title="",

    showlegend=False
)


style_chart(
    fig_profile
)


st.plotly_chart(
    fig_profile,
    width="stretch",
    theme=None,
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
    width="stretch"
)


# ------------------------------------------------------------
# GENERATE AND STORE PREDICTION
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


    st.session_state.prediction_inputs = (
        input_data.copy()
    )

    st.session_state.prediction_result = (
        result
    )

    st.session_state.scenario_result = None


    # Reset scenario values to current inputs whenever
    # a completely new main prediction is generated.

    st.session_state.sim_attendance = float(
        attendance_percentage
    )

    st.session_state.sim_assignments = float(
        assignment_completion_rate
    )

    st.session_state.sim_study_hours = float(
        study_hours_per_day
    )

    st.session_state.sim_practice_tests = int(
        practice_tests_completed
    )


# ------------------------------------------------------------
# DISPLAY STORED PREDICTION
# ------------------------------------------------------------

if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    baseline_inputs = (
        st.session_state.prediction_inputs
    )


    predicted_score = float(
        result["predicted_score"]
    )

    predicted_status = str(
        result["predicted_status"]
    )

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


    status_class = (
        "result-pass"
        if predicted_status.lower() == "pass"
        else "result-fail"
    )


    result1, result2, result3 = st.columns(3)


    score_card = (
        f'<div class="prediction-hero">'
        f'<div class="prediction-hero-label">'
        f'Predicted Exam Score'
        f'</div>'
        f'<div class="prediction-hero-value">'
        f'{predicted_score:.1f} / 100'
        f'</div>'
        f'<div class="prediction-hero-note">'
        f'Linear Regression output'
        f'</div>'
        f'</div>'
    )


    outcome_card = (
        f'<div class="prediction-hero">'
        f'<div class="prediction-hero-label">'
        f'Predicted Outcome'
        f'</div>'
        f'<div class="prediction-hero-value {status_class}">'
        f'{predicted_status.upper()}'
        f'</div>'
        f'<div class="prediction-hero-note">'
        f'Balanced Logistic Regression'
        f'</div>'
        f'</div>'
    )


    probability_card = (
        f'<div class="prediction-hero">'
        f'<div class="prediction-hero-label">'
        f'Pass Probability'
        f'</div>'
        f'<div class="prediction-hero-value">'
        f'{probability_percent:.1f}%'
        f'</div>'
        f'<div class="prediction-hero-note">'
        f'Estimated classification probability'
        f'</div>'
        f'</div>'
    )


    with result1:
        st.html(score_card)


    with result2:
        st.html(outcome_card)


    with result3:
        st.html(probability_card)


    # --------------------------------------------------------
    # PREDICTION OVERVIEW
    # --------------------------------------------------------

    section_title(
        "chart",
        "Prediction Overview"
    )


    visual_col1, visual_col2 = st.columns(2)


    # --------------------------------------------------------
    # SCORE BAR
    # --------------------------------------------------------

    with visual_col1:

        fig_score = go.Figure()


        fig_score.add_trace(
            go.Bar(
                x=[predicted_score],
                y=["Exam Score"],
                orientation="h",

                marker=dict(
                    color=accent
                ),

                text=[
                    f"{predicted_score:.1f}"
                ],

                textposition="inside",

                hovertemplate=(
                    f"Predicted exam score: "
                    f"{predicted_score:.1f}"
                    "<extra></extra>"
                )
            )
        )


        fig_score.update_layout(
            height=175,

            margin=dict(
                l=10,
                r=20,
                t=35,
                b=25
            ),

            title=dict(
                text="Predicted Exam Score",
                font=dict(
                    size=15
                )
            ),

            xaxis=dict(
                range=[0, 100],
                title=""
            ),

            yaxis_title="",

            showlegend=False
        )


        style_chart(
            fig_score
        )


        st.plotly_chart(
            fig_score,
            width="stretch",
            theme=None,
            config={
                "displayModeBar": False
            }
        )


    # --------------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------------

    with visual_col2:

        fig_probability = go.Figure()


        fig_probability.add_trace(
            go.Bar(
                x=[probability_percent],
                y=["Pass Probability"],
                orientation="h",

                marker=dict(
                    color=accent
                ),

                text=[
                    f"{probability_percent:.1f}%"
                ],

                textposition="inside",

                hovertemplate=(
                    f"Pass probability: "
                    f"{probability_percent:.1f}%"
                    "<extra></extra>"
                )
            )
        )


        fig_probability.add_vline(
            x=50,
            line_dash="dash",
            line_color=muted
        )


        fig_probability.update_layout(
            height=175,

            margin=dict(
                l=10,
                r=20,
                t=35,
                b=25
            ),

            title=dict(
                text="Pass Probability",
                font=dict(
                    size=15
                )
            ),

            xaxis=dict(
                range=[0, 100],
                title="",
                ticksuffix="%"
            ),

            yaxis_title="",

            showlegend=False
        )


        style_chart(
            fig_probability
        )


        st.plotly_chart(
            fig_probability,
            width="stretch",
            theme=None,
            config={
                "displayModeBar": False
            }
        )


    # --------------------------------------------------------
    # OUTCOME MESSAGE
    # --------------------------------------------------------

    if predicted_status.lower() == "pass":

        st.success(
            f"Predicted outcome: PASS · "
            f"{probability_percent:.1f}% estimated pass probability."
        )

    else:

        st.error(
            f"Predicted outcome: FAIL · "
            f"{probability_percent:.1f}% estimated pass probability."
        )


    # --------------------------------------------------------
    # DECISION THRESHOLD PROXIMITY
    # --------------------------------------------------------

    section_title(
        "scale",
        "Decision Threshold"
    )


    distance_from_threshold = abs(
        probability_percent - 50
    )


    if distance_from_threshold < 10:

        threshold_label = (
            "Close to the Decision Boundary"
        )

        threshold_explanation = (
            "The predicted probability is relatively close to "
            "the 50% classification threshold, so the predicted "
            "class could change with comparatively small changes "
            "in the model inputs."
        )

    elif distance_from_threshold < 20:

        threshold_label = (
            "Moderately Far from the Decision Boundary"
        )

        threshold_explanation = (
            "The predicted probability has some separation from "
            "the model's 50% decision threshold, although it is "
            "not extremely far from the boundary."
        )

    else:

        threshold_label = (
            "Farther from the Decision Boundary"
        )

        threshold_explanation = (
            "The predicted probability is farther from the model's "
            "50% decision threshold, indicating stronger separation "
            "for this particular classification."
        )


    threshold_html = (
        f'<div class="threshold-card">'
        f'<div class="threshold-label">'
        f'Distance from 50% threshold'
        f'</div>'
        f'<div class="threshold-value">'
        f'{distance_from_threshold:.1f} percentage points · '
        f'{threshold_label}'
        f'</div>'
        f'<div class="threshold-note">'
        f'{threshold_explanation}'
        f'</div>'
        f'</div>'
    )


    st.html(
        threshold_html
    )


    st.caption(
        "Distance from the classification threshold is not the same "
        "as a statistical confidence interval."
    )


    # --------------------------------------------------------
    # EXPECTED SCORE RANGE
    # --------------------------------------------------------

    section_title(
        "trending",
        "Expected Score Range"
    )


    lower_score = max(
        0,
        predicted_score - MODEL_MAE
    )

    upper_score = min(
        100,
        predicted_score + MODEL_MAE
    )


    fig_range = go.Figure()


    fig_range.add_trace(
        go.Scatter(

            x=[
                lower_score,
                upper_score
            ],

            y=[1, 1],

            mode="lines",

            line=dict(
                width=10,
                color=accent
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


    fig_range.add_trace(
        go.Scatter(

            x=[predicted_score],

            y=[1],

            mode="markers+text",

            marker=dict(
                size=15,
                color=accent,

                line=dict(
                    width=2,
                    color=background
                )
            ),

            text=[
                f"{predicted_score:.1f}"
            ],

            textposition="top center",

            textfont=dict(
                color=text_color
            ),

            hovertemplate=(
                f"Predicted score: "
                f"{predicted_score:.1f}"
                "<extra></extra>"
            ),

            showlegend=False
        )
    )


    fig_range.update_layout(
        height=135,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=30
        ),

        xaxis=dict(
            range=[0, 100],
            title="Exam Score"
        ),

        yaxis=dict(
            visible=False,
            range=[0.82, 1.18]
        ),

        showlegend=False
    )


    style_chart(
        fig_range
    )


    st.plotly_chart(
        fig_range,
        width="stretch",
        theme=None,

        config={
            "displayModeBar": False
        }
    )


    st.caption(
        f"Approximate MAE-based range: "
        f"{lower_score:.1f}–{upper_score:.1f}. "
        "This is not a statistical confidence interval."
    )


    # --------------------------------------------------------
    # INTERPRETATION
    # --------------------------------------------------------

    section_title(
        "brain",
        "What Does This Result Mean?"
    )


    interpretation_html = (
        f'<div class="interpretation-box">'
        f'<div class="interpretation-title">'
        f'Prediction Interpretation'
        f'</div>'
        f'<div class="interpretation-text">'
        f'The Linear Regression model estimates an exam score of '
        f'<b>{predicted_score:.1f}</b> out of 100.'
        f'<br><br>'
        f'The Balanced Logistic Regression model separately predicts '
        f'<b>{predicted_status}</b> with an estimated '
        f'<b>{probability_percent:.1f}% probability of passing</b>.'
        f'<br><br>'
        f'These are two independent model outputs produced from the '
        f'same six student input variables.'
        f'</div>'
        f'</div>'
    )


    st.html(
        interpretation_html
    )


    # --------------------------------------------------------
    # IMPROVEMENT SCENARIO SIMULATOR
    # --------------------------------------------------------

    section_title(
        "sliders",
        "Improvement Scenario Simulator"
    )


    st.caption(
        "Adjust controllable study-related variables and compare the "
        "model's new prediction with the current result. This is a "
        "model-based scenario, not a guaranteed or causal outcome."
    )


    # Make sure scenario defaults exist if prediction
    # was restored through session state.

    if "sim_attendance" not in st.session_state:

        st.session_state.sim_attendance = float(
            baseline_inputs[
                "attendance_percentage"
            ]
        )


    if "sim_assignments" not in st.session_state:

        st.session_state.sim_assignments = float(
            baseline_inputs[
                "assignment_completion_rate"
            ]
        )


    if "sim_study_hours" not in st.session_state:

        st.session_state.sim_study_hours = float(
            baseline_inputs[
                "study_hours_per_day"
            ]
        )


    if "sim_practice_tests" not in st.session_state:

        st.session_state.sim_practice_tests = int(
            baseline_inputs[
                "practice_tests_completed"
            ]
        )


    # --------------------------------------------------------
    # SCENARIO FORM
    # --------------------------------------------------------

    with st.form(
        "improvement_scenario_form"
    ):

        sim_col1, sim_col2 = st.columns(2)


        with sim_col1:

            sim_attendance = st.number_input(
                "Scenario Attendance · 0–100%",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="sim_attendance"
            )


            sim_assignments = st.number_input(
                "Scenario Assignment Completion · 0–100%",
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                key="sim_assignments"
            )


        with sim_col2:

            sim_study_hours = st.number_input(
                "Scenario Study Hours per Day · 0–12",
                min_value=0.0,
                max_value=12.0,
                step=0.5,
                key="sim_study_hours"
            )


            sim_practice_tests = st.number_input(
                "Scenario Practice Tests · 0–17",
                min_value=0,
                max_value=17,
                step=1,
                key="sim_practice_tests"
            )


        simulate_button = st.form_submit_button(
            "Simulate Scenario",
            type="primary",
            width="stretch"
        )


    # --------------------------------------------------------
    # RUN SCENARIO MODEL
    # --------------------------------------------------------

    if simulate_button:

        scenario_inputs = {

            # Previous academic performance remains unchanged.

            "previous_exam_score":
                baseline_inputs[
                    "previous_exam_score"
                ],

            "previous_gpa":
                baseline_inputs[
                    "previous_gpa"
                ],

            # User can change these four variables.

            "attendance_percentage":
                sim_attendance,

            "assignment_completion_rate":
                sim_assignments,

            "study_hours_per_day":
                sim_study_hours,

            "practice_tests_completed":
                sim_practice_tests
        }


        with st.spinner(
            "Simulating scenario..."
        ):

            scenario_prediction = (
                predict_student_performance(
                    scenario_inputs
                )
            )


        st.session_state.scenario_result = {
            "prediction":
                scenario_prediction,

            "inputs":
                scenario_inputs
        }


    # --------------------------------------------------------
    # SCENARIO RESULTS
    # --------------------------------------------------------

    if st.session_state.scenario_result is not None:

        scenario_data = (
            st.session_state.scenario_result
        )

        scenario_prediction = (
            scenario_data[
                "prediction"
            ]
        )


        scenario_score = float(
            scenario_prediction[
                "predicted_score"
            ]
        )


        scenario_status = str(
            scenario_prediction[
                "predicted_status"
            ]
        )


        scenario_probability = float(
            scenario_prediction[
                "pass_probability"
            ]
        ) * 100


        score_change = (
            scenario_score
            - predicted_score
        )


        probability_change = (
            scenario_probability
            - probability_percent
        )


        score_css, score_change_text = (
            format_change(
                score_change,
                " points"
            )
        )


        probability_css, probability_change_text = (
            format_change(
                probability_change,
                " pp"
            )
        )


        st.markdown(
            "#### Scenario Comparison"
        )


        compare1, compare2, compare3 = (
            st.columns(3)
        )


        current_card = (
            f'<div class="scenario-card">'
            f'<div class="scenario-label">'
            f'Current Prediction'
            f'</div>'
            f'<div class="scenario-value">'
            f'{predicted_score:.1f} / 100'
            f'</div>'
            f'<div class="scenario-note">'
            f'{predicted_status} · '
            f'{probability_percent:.1f}% pass probability'
            f'</div>'
            f'</div>'
        )


        scenario_card = (
            f'<div class="scenario-card">'
            f'<div class="scenario-label">'
            f'Scenario Prediction'
            f'</div>'
            f'<div class="scenario-value">'
            f'{scenario_score:.1f} / 100'
            f'</div>'
            f'<div class="scenario-note">'
            f'{scenario_status} · '
            f'{scenario_probability:.1f}% pass probability'
            f'</div>'
            f'</div>'
        )


        change_card = (
            f'<div class="scenario-card">'
            f'<div class="scenario-label">'
            f'Model-Predicted Change'
            f'</div>'
            f'<div class="{score_css}">'
            f'Score: {score_change_text}'
            f'</div>'
            f'<div class="{probability_css}" '
            f'style="margin-top:8px;">'
            f'Pass probability: {probability_change_text}'
            f'</div>'
            f'</div>'
        )


        with compare1:
            st.html(
                current_card
            )


        with compare2:
            st.html(
                scenario_card
            )


        with compare3:
            st.html(
                change_card
            )


        st.caption(
            "pp = percentage points. Differences shown here are changes "
            "in model predictions under the selected scenario; they do "
            "not establish that changing a variable will cause the "
            "predicted improvement."
        )


    # --------------------------------------------------------
    # HOW THE PREDICTION WORKS
    # --------------------------------------------------------

    section_title(
        "database",
        "How This Prediction Works"
    )


    with st.expander(
        "View model and evaluation details"
    ):

        st.markdown(
            """
            **Exam-score prediction**

            The numerical exam score is generated by the project's
            **Linear Regression** model using six pre-exam variables.

            **Pass / Fail prediction**

            Pass / Fail status and pass probability are generated
            separately by the **Balanced Logistic Regression** model.

            **Inputs used by both models**

            - Previous Exam Score
            - Previous GPA
            - Attendance Percentage
            - Assignment Completion Rate
            - Study Hours per Day
            - Practice Tests Completed
            """
        )


        detail1, detail2, detail3 = (
            st.columns(3)
        )


        detail1.metric(
            "Linear Model MAE",
            "9.38 points"
        )


        detail2.metric(
            "Classification Balanced Accuracy",
            "73.2%"
        )


        detail3.metric(
            "Classification AUC",
            "0.814"
        )


        st.info(
            "The two predictions are produced by separate models. "
            "Therefore, the predicted exam score does not directly "
            "determine the Pass / Fail probability."
        )


        st.caption(
            "Predictions are estimates based on patterns in the training "
            "data and should not be interpreted as guaranteed student outcomes."
        )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

show_footer()