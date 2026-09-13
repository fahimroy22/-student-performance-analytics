# ============================================================
# PREDICT PERFORMANCE PAGE
# Compact card-based interactive version
# ============================================================

import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

from core.model_loader import (
    load_linear_model,
    load_logistic_model,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predict Performance",
    page_icon="🎓",
    layout="wide",
)

apply_global_styles()


# ============================================================
# THEME
# ============================================================

plotly_template = get_plotly_template()
theme = get_theme_colors()

background = theme["background"]
surface = theme["surface"]
text_color = theme["text"]
muted = theme["muted"]
border = theme["border"]
accent = theme["accent"]

positive_color = "#5FA879"
negative_color = "#C86B6B"


# ============================================================
# CONSTANTS
# ============================================================

MODEL_MAE = 9.38
CLASSIFICATION_BALANCED_ACCURACY = 73.2
CLASSIFICATION_AUC = 0.814

FEATURES = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed",
]

FEATURE_LABELS = {
    "previous_exam_score": "Previous Exam",
    "previous_gpa": "Previous GPA",
    "attendance_percentage": "Attendance",
    "assignment_completion_rate": "Assignments",
    "study_hours_per_day": "Study Hours",
    "practice_tests_completed": "Practice Tests",
}

DEFAULT_VALUES = {
    "previous_exam_score": 70.0,
    "previous_gpa": 3.0,
    "attendance_percentage": 85.0,
    "assignment_completion_rate": 75.0,
    "study_hours_per_day": 3.0,
    "practice_tests_completed": 6,
}


# ============================================================
# PAGE CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .block-container {{
        padding-top: 1.55rem;
        padding-bottom: 2rem;
    }}

    /* --------------------------------------------------------
       COMPACT INPUT CARDS
    -------------------------------------------------------- */

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {surface};
        border-color: {border} !important;
        border-radius: 14px;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        padding-top: 0.15rem;
    }}

    div[data-testid="stNumberInput"] {{
        margin-top: -0.10rem;
        margin-bottom: -0.15rem;
    }}

    div[data-testid="stNumberInput"] label {{
        display: none;
    }}

    div[data-testid="stNumberInput"] input {{
        font-weight: 650;
        font-size: 1rem;
    }}

    .input-category {{
        color: {accent};
        font-size: 0.66rem;
        font-weight: 750;
        letter-spacing: 0.06rem;
        text-transform: uppercase;
        margin-bottom: 2px;
    }}

    .input-card-title {{
        color: {text_color};
        font-size: 0.93rem;
        font-weight: 650;
        margin-bottom: 2px;
    }}

    .input-card-range {{
        color: {muted};
        font-size: 0.70rem;
        margin-bottom: 6px;
    }}

    /* --------------------------------------------------------
       RESULT CARDS
    -------------------------------------------------------- */

    .prediction-hero {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 15px 18px;
        min-height: 112px;
    }}

    .prediction-hero-label {{
        color: {muted};
        font-size: 0.76rem;
        margin-bottom: 5px;
    }}

    .prediction-hero-value {{
        color: {text_color};
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
    }}

    .prediction-hero-note {{
        color: {muted};
        font-size: 0.78rem;
        margin-top: 7px;
    }}

    .result-pass {{
        color: {positive_color} !important;
    }}

    .result-fail {{
        color: {negative_color} !important;
    }}

    /* --------------------------------------------------------
       SUMMARY TABLE
    -------------------------------------------------------- */

    .compact-model-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        border: 1px solid {border};
        border-radius: 10px;
        overflow: hidden;
        background: {surface};
        font-size: 0.82rem;
    }}

    .compact-model-table th {{
        color: {muted};
        background: {surface};
        padding: 10px 12px;
        text-align: left;
        border-bottom: 1px solid {border};
    }}

    .compact-model-table td {{
        color: {text_color};
        padding: 10px 12px;
        border-bottom: 1px solid {border};
    }}

    .compact-model-table tr:last-child td {{
        border-bottom: none;
    }}

    /* --------------------------------------------------------
       SCENARIO
    -------------------------------------------------------- */

    .scenario-card {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 13px 16px;
        min-height: 103px;
    }}

    .scenario-label {{
        color: {muted};
        font-size: 0.76rem;
        margin-bottom: 5px;
    }}

    .scenario-value {{
        color: {text_color};
        font-size: 1.35rem;
        font-weight: 700;
    }}

    .scenario-note {{
        color: {muted};
        font-size: 0.78rem;
        margin-top: 5px;
    }}

    .positive-change {{
        color: {positive_color};
        font-weight: 650;
    }}

    .negative-change {{
        color: {negative_color};
        font-weight: 650;
    }}

    .neutral-change {{
        color: {muted};
        font-weight: 650;
    }}

    .no-change-box {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 14px 16px;
        margin-top: 12px;
        color: {muted};
        font-size: 0.82rem;
    }}

    .scroll-anchor {{
        width: 1px;
        height: 1px;
        visibility: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

for key, value in DEFAULT_VALUES.items():

    if key not in st.session_state:
        st.session_state[key] = value


for key in [
    "prediction_result",
    "prediction_inputs",
    "scenario_result",
    "scroll_target",
]:

    if key not in st.session_state:
        st.session_state[key] = None


# ============================================================
# MODELS FOR PREDICTION EXPLANATION
# ============================================================

@st.cache_resource
def get_explanation_models():

    return (
        load_linear_model(),
        load_logistic_model(),
    )


linear_model, logistic_model = (
    get_explanation_models()
)


# ============================================================
# CHART STYLE
# ============================================================

def style_chart(fig):

    fig.update_layout(
        template=plotly_template,
        paper_bgcolor=background,
        plot_bgcolor=background,
        font=dict(
            color=text_color
        ),
    )

    fig.update_xaxes(
        color=text_color,
        gridcolor=border,
        zerolinecolor=border,
    )

    fig.update_yaxes(
        color=text_color,
        gridcolor=border,
        zerolinecolor=border,
    )

    return fig


# ============================================================
# CLEAR RESULTS
# ============================================================

def clear_results():

    st.session_state.prediction_result = None
    st.session_state.prediction_inputs = None
    st.session_state.scenario_result = None
    st.session_state.scroll_target = None

    for key in [
        "sim_attendance",
        "sim_assignments",
        "sim_study_hours",
        "sim_practice_tests",
    ]:

        st.session_state.pop(
            key,
            None,
        )


# ============================================================
# QUICK ACTIONS
# ============================================================

def generate_sample():

    st.session_state.previous_exam_score = float(
        random.randint(40, 95)
    )

    st.session_state.previous_gpa = round(
        random.uniform(1.5, 4.0),
        1,
    )

    st.session_state.attendance_percentage = float(
        random.randint(50, 100)
    )

    st.session_state.assignment_completion_rate = float(
        random.randint(40, 100)
    )

    st.session_state.study_hours_per_day = round(
        random.uniform(0.5, 8.0),
        1,
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


# ============================================================
# AUTO SCROLL
# ============================================================

def scroll_to_anchor(
    anchor_id
):

    scroll_html = f"""
    <html>

    <body
        style="
            margin:0;
            padding:0;
            background:transparent;
        "
    >

    <script>

    (function() {{

        function performScroll() {{

            try {{

                const doc =
                    window.parent.document;

                const target =
                    doc.getElementById(
                        "{anchor_id}"
                    );

                if (target) {{

                    target.scrollIntoView({{
                        behavior: "smooth",
                        block: "start"
                    }});

                }}

            }}

            catch (error) {{

            }}

        }}

        setTimeout(
            performScroll,
            120
        );

    }})();

    </script>

    </body>

    </html>
    """

    st.iframe(
        scroll_html,
        height=1,
        width="stretch",
        tab_index=-1,
    )


# ============================================================
# CHANGE FORMAT
# ============================================================

def format_change(
    value,
    suffix="",
):

    if value > 0:

        return (
            "positive-change",
            f"+{value:.1f}{suffix}",
        )

    if value < 0:

        return (
            "negative-change",
            f"{value:.1f}{suffix}",
        )

    return (
        "neutral-change",
        f"{value:.1f}{suffix}",
    )


# ============================================================
# MODEL CONTRIBUTIONS
# ============================================================

def model_contributions(
    pipeline,
    input_data,
    logistic=False,
):

    input_df = pd.DataFrame(
        [input_data],
        columns=FEATURES,
    )

    transformed = (
        pipeline[:-1]
        .transform(
            input_df
        )
    )

    if hasattr(
        transformed,
        "toarray",
    ):

        transformed = (
            transformed.toarray()
        )

    transformed = np.asarray(
        transformed
    )

    coefficients = np.asarray(
        pipeline
        .named_steps["model"]
        .coef_
    )


    # --------------------------------------------------------
    # LOGISTIC
    # --------------------------------------------------------

    if logistic:

        classes = list(
            pipeline
            .named_steps["model"]
            .classes_
        )

        coefficients = (
            coefficients[0]
        )

        # sklearn binary logistic coefficients point
        # toward classes_[1].
        # Positive values should always mean "toward Pass".

        if classes[1] != "Pass":

            coefficients = (
                -coefficients
            )


    # --------------------------------------------------------
    # LINEAR
    # --------------------------------------------------------

    else:

        if coefficients.ndim > 1:

            coefficients = (
                coefficients[0]
            )


    contributions = (
        transformed[0]
        * coefficients
    )


    contribution_df = pd.DataFrame(
        {
            "Feature": [
                FEATURE_LABELS[
                    feature
                ]
                for feature in FEATURES
            ],

            "Contribution":
                contributions,
        }
    )

    return contribution_df


# ============================================================
# INPUT CARD
# ============================================================

def input_card(
    category,
    title,
    range_text,
    key,
    min_value,
    max_value,
    step,
    display_format=None,
):

    with st.container(
        border=True
    ):

        st.html(
            f"""
            <div class="input-category">
                {category}
            </div>

            <div class="input-card-title">
                {title}
            </div>

            <div class="input-card-range">
                {range_text}
            </div>
            """
        )

        kwargs = {
            "label":
                title,

            "min_value":
                min_value,

            "max_value":
                max_value,

            "step":
                step,

            "key":
                key,

            "label_visibility":
                "collapsed",
        }

        if display_format is not None:

            kwargs["format"] = (
                display_format
            )


        value = st.number_input(
            **kwargs
        )


        normalized = (
            (
                float(value)
                - float(min_value)
            )
            /
            (
                float(max_value)
                - float(min_value)
            )
        )


        normalized = max(
            0.0,
            min(
                1.0,
                normalized,
            ),
        )


        st.progress(
            normalized
        )


        return value


# ============================================================
# HEADER
# ============================================================

page_title(
    "sparkles",
    "Student Performance Predictor",
    "Enter six academic variables to estimate exam score, "
    "Pass / Fail status, and pass probability.",
)


# ============================================================
# QUICK ACTIONS
# ============================================================

section_title(
    "sliders",
    "Quick Actions",
)


qa1, qa2, qa3, _ = st.columns(
    [
        1,
        1,
        1.1,
        4,
    ]
)


with qa1:

    st.button(
        "Generate Sample",
        on_click=generate_sample,
        width="stretch",
    )


with qa2:

    st.button(
        "Reset Inputs",
        on_click=reset_inputs,
        width="stretch",
    )


with qa3:

    st.button(
        "At-Risk Example",
        on_click=generate_at_risk_sample,
        width="stretch",
    )


# ============================================================
# TOP INPUT DASHBOARD
# ============================================================

input_panel, profile_panel = (
    st.columns(
        [
            2,
            0.95,
        ],
        gap="large",
    )
)


# ============================================================
# INPUT CARDS
# ============================================================

with input_panel:

    section_title(
        "filter",
        "Student Academic Information",
    )


    # --------------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------------

    card1, card2, card3 = (
        st.columns(3)
    )


    with card1:

        previous_exam_score = (
            input_card(
                category=(
                    "Academic History"
                ),

                title=(
                    "Previous Exam"
                ),

                range_text=(
                    "Score · 0–100"
                ),

                key=(
                    "previous_exam_score"
                ),

                min_value=0.0,
                max_value=100.0,
                step=1.0,

                display_format="%.0f",
            )
        )


    with card2:

        previous_gpa = (
            input_card(
                category=(
                    "Academic History"
                ),

                title=(
                    "Previous GPA"
                ),

                range_text=(
                    "GPA · 0–4.0"
                ),

                key=(
                    "previous_gpa"
                ),

                min_value=0.0,
                max_value=4.0,
                step=0.1,

                display_format="%.1f",
            )
        )


    with card3:

        attendance_percentage = (
            input_card(
                category=(
                    "Engagement"
                ),

                title=(
                    "Attendance"
                ),

                range_text=(
                    "Percent · 0–100"
                ),

                key=(
                    "attendance_percentage"
                ),

                min_value=0.0,
                max_value=100.0,
                step=1.0,

                display_format="%.0f",
            )
        )


    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    card4, card5, card6 = (
        st.columns(3)
    )


    with card4:

        assignment_completion_rate = (
            input_card(
                category=(
                    "Engagement"
                ),

                title=(
                    "Assignments"
                ),

                range_text=(
                    "Completion · 0–100%"
                ),

                key=(
                    "assignment_completion_rate"
                ),

                min_value=0.0,
                max_value=100.0,
                step=1.0,

                display_format="%.0f",
            )
        )


    with card5:

        study_hours_per_day = (
            input_card(
                category=(
                    "Preparation"
                ),

                title=(
                    "Study Hours"
                ),

                range_text=(
                    "Hours/day · 0–12"
                ),

                key=(
                    "study_hours_per_day"
                ),

                min_value=0.0,
                max_value=12.0,
                step=0.5,

                display_format="%.1f",
            )
        )


    with card6:

        practice_tests_completed = (
            input_card(
                category=(
                    "Preparation"
                ),

                title=(
                    "Practice Tests"
                ),

                range_text=(
                    "Completed · 0–17"
                ),

                key=(
                    "practice_tests_completed"
                ),

                min_value=0,
                max_value=17,
                step=1,
            )
        )


    predict_button = st.button(
        "Predict Performance",
        type="primary",
        width="stretch",
    )


# ============================================================
# LIVE RADAR PROFILE
# ============================================================

with profile_panel:

    section_title(
        "chart",
        "Student Profile",
    )


    radar_labels = [
        "Previous Exam",
        "GPA",
        "Attendance",
        "Assignments",
        "Study Hours",
        "Practice Tests",
    ]


    radar_values = [
        previous_exam_score,

        (
            previous_gpa
            / 4.0
        ) * 100,

        attendance_percentage,

        assignment_completion_rate,

        (
            study_hours_per_day
            / 12.0
        ) * 100,

        (
            practice_tests_completed
            / 17.0
        ) * 100,
    ]


    radar_labels_closed = (
        radar_labels
        + [
            radar_labels[0]
        ]
    )


    radar_values_closed = (
        radar_values
        + [
            radar_values[0]
        ]
    )


    fig_profile = go.Figure()


    fig_profile.add_trace(
        go.Scatterpolar(
            r=radar_values_closed,

            theta=(
                radar_labels_closed
            ),

            fill="toself",

            line=dict(
                color=accent,
                width=2,
            ),

            marker=dict(
                color=accent,
                size=6,
            ),

            hovertemplate=(
                "%{theta}: "
                "%{r:.0f}%"
                "<extra></extra>"
            ),
        )
    )


    fig_profile.update_layout(
        template=plotly_template,

        height=395,

        margin=dict(
            l=35,
            r=35,
            t=20,
            b=25,
        ),

        polar=dict(
            bgcolor=background,

            radialaxis=dict(
                visible=True,

                range=[
                    0,
                    100,
                ],

                tickvals=[
                    25,
                    50,
                    75,
                    100,
                ],

                ticksuffix="%",

                gridcolor=border,

                color=muted,
            ),

            angularaxis=dict(
                gridcolor=border,
                color=text_color,
            ),
        ),

        showlegend=False,

        paper_bgcolor=background,

        font=dict(
            color=text_color
        ),
    )


    st.plotly_chart(
        fig_profile,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


    st.caption(
        "Each input is normalized to a 0–100 scale "
        "for profile visualization only."
    )


# ============================================================
# CREATE PREDICTION
# ============================================================

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
            practice_tests_completed,
    }


    with st.spinner(
        "Analyzing student performance..."
    ):

        result = (
            predict_student_performance(
                input_data
            )
        )


    st.session_state.prediction_inputs = (
        input_data.copy()
    )

    st.session_state.prediction_result = (
        result
    )

    st.session_state.scenario_result = (
        None
    )


    st.session_state.sim_attendance = (
        float(
            attendance_percentage
        )
    )

    st.session_state.sim_assignments = (
        float(
            assignment_completion_rate
        )
    )

    st.session_state.sim_study_hours = (
        float(
            study_hours_per_day
        )
    )

    st.session_state.sim_practice_tests = (
        int(
            practice_tests_completed
        )
    )


    st.session_state.scroll_target = (
        "prediction-results-anchor"
    )


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.prediction_result
    is not None
):

    result = (
        st.session_state.prediction_result
    )

    baseline_inputs = (
        st.session_state.prediction_inputs
    )


    predicted_score = float(
        result[
            "predicted_score"
        ]
    )


    predicted_status = str(
        result[
            "predicted_status"
        ]
    )


    pass_probability = float(
        result[
            "pass_probability"
        ]
    )


    probability_percent = (
        pass_probability
        * 100
    )


    # ========================================================
    # RESULT ANCHOR
    # ========================================================

    st.html(
        """
        <div
            id="prediction-results-anchor"
            class="scroll-anchor"
        ></div>
        """
    )


    if (
        st.session_state.scroll_target
        == "prediction-results-anchor"
    ):

        scroll_to_anchor(
            "prediction-results-anchor"
        )

        st.session_state.scroll_target = (
            None
        )


    # ========================================================
    # RESULT CARDS
    # ========================================================

    section_title(
        "target",
        "Prediction Results",
    )


    status_class = (
        "result-pass"
        if predicted_status.lower()
        == "pass"
        else "result-fail"
    )


    result1, result2, result3 = (
        st.columns(3)
    )


    with result1:

        st.html(
            f"""
            <div class="prediction-hero">

                <div class="prediction-hero-label">
                    Predicted Exam Score
                </div>

                <div class="prediction-hero-value">
                    {predicted_score:.1f} / 100
                </div>

                <div class="prediction-hero-note">
                    Linear Regression
                </div>

            </div>
            """
        )


    with result2:

        st.html(
            f"""
            <div class="prediction-hero">

                <div class="prediction-hero-label">
                    Predicted Outcome
                </div>

                <div class="prediction-hero-value {status_class}">
                    {predicted_status.upper()}
                </div>

                <div class="prediction-hero-note">
                    Balanced Logistic Regression
                </div>

            </div>
            """
        )


    with result3:

        st.html(
            f"""
            <div class="prediction-hero">

                <div class="prediction-hero-label">
                    Pass Probability
                </div>

                <div class="prediction-hero-value">
                    {probability_percent:.1f}%
                </div>

                <div class="prediction-hero-note">
                    Model probability estimate
                </div>

            </div>
            """
        )


    # ========================================================
    # PREDICTION DASHBOARD
    # ========================================================

    section_title(
        "chart",
        "Prediction Dashboard",
    )


    gauge1, gauge2 = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # SCORE GAUGE
    # --------------------------------------------------------

    with gauge1:

        fig_score_gauge = (
            go.Figure(
                go.Indicator(
                    mode="gauge+number",

                    value=predicted_score,

                    number={
                        "suffix":
                            " / 100",

                        "font": {
                            "size":
                                36
                        },
                    },

                    title={
                        "text":
                            "Predicted Exam Score"
                    },

                    gauge={
                        "axis": {
                            "range":
                                [
                                    0,
                                    100,
                                ]
                        },

                        "bar": {
                            "color":
                                accent,

                            "thickness":
                                0.22,
                        },

                        "bgcolor":
                            surface,

                        "bordercolor":
                            border,
                    },
                )
            )
        )


        fig_score_gauge.update_layout(
            height=215,

            margin=dict(
                l=30,
                r=30,
                t=35,
                b=5,
            ),

            paper_bgcolor=(
                background
            ),

            font=dict(
                color=text_color
            ),
        )


        st.plotly_chart(
            fig_score_gauge,
            width="stretch",
            theme=None,

            config={
                "displayModeBar":
                    False
            },
        )


    # --------------------------------------------------------
    # PASS PROBABILITY GAUGE
    # --------------------------------------------------------

    with gauge2:

        fig_probability_gauge = (
            go.Figure(
                go.Indicator(
                    mode="gauge+number",

                    value=(
                        probability_percent
                    ),

                    number={
                        "suffix":
                            "%",

                        "font": {
                            "size":
                                36
                        },
                    },

                    title={
                        "text":
                            "Pass Probability"
                    },

                    gauge={
                        "axis": {
                            "range":
                                [
                                    0,
                                    100,
                                ]
                        },

                        "bar": {
                            "color":
                                accent,

                            "thickness":
                                0.22,
                        },

                        "bgcolor":
                            surface,

                        "bordercolor":
                            border,

                        "threshold": {
                            "line": {
                                "color":
                                    muted,

                                "width":
                                    3,
                            },

                            "thickness":
                                0.70,

                            "value":
                                50,
                        },
                    },
                )
            )
        )


        fig_probability_gauge.update_layout(
            height=215,

            margin=dict(
                l=30,
                r=30,
                t=35,
                b=5,
            ),

            paper_bgcolor=(
                background
            ),

            font=dict(
                color=text_color
            ),
        )


        st.plotly_chart(
            fig_probability_gauge,
            width="stretch",
            theme=None,

            config={
                "displayModeBar":
                    False
            },
        )


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    summary_html = f"""
    <table class="compact-model-table">

        <thead>

            <tr>
                <th>Output</th>
                <th>Value</th>
                <th>Model</th>
            </tr>

        </thead>

        <tbody>

            <tr>
                <td>Exam Score</td>
                <td>{predicted_score:.1f} / 100</td>
                <td>Linear Regression</td>
            </tr>

            <tr>
                <td>Classification</td>
                <td>{predicted_status}</td>
                <td>Balanced Logistic Regression</td>
            </tr>

            <tr>
                <td>Pass Probability</td>
                <td>{probability_percent:.1f}%</td>
                <td>Balanced Logistic Regression</td>
            </tr>

        </tbody>

    </table>
    """


    st.html(
        summary_html
    )


    if (
        predicted_status.lower()
        == "pass"
    ):

        st.success(
            f"PASS predicted · "
            f"{probability_percent:.1f}% "
            f"estimated pass probability."
        )

    else:

        st.error(
            f"FAIL predicted · "
            f"{probability_percent:.1f}% "
            f"estimated pass probability."
        )


    # ========================================================
    # THRESHOLD + SCORE RANGE
    # ========================================================

    threshold_col, range_col = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # DECISION THRESHOLD
    # --------------------------------------------------------

    with threshold_col:

        section_title(
            "scale",
            "Decision Threshold",
        )


        distance_from_threshold = (
            abs(
                probability_percent
                - 50
            )
        )


        if (
            distance_from_threshold
            < 10
        ):

            threshold_label = (
                "Close to Boundary"
            )


        elif (
            distance_from_threshold
            < 20
        ):

            threshold_label = (
                "Moderate Separation"
            )


        else:

            threshold_label = (
                "Stronger Separation"
            )


        fig_threshold = (
            go.Figure()
        )


        fig_threshold.add_trace(
            go.Bar(
                x=[
                    probability_percent
                ],

                y=[
                    "Probability"
                ],

                orientation="h",

                marker_color=accent,

                text=[
                    f"{probability_percent:.1f}%"
                ],

                textposition="inside",
            )
        )


        fig_threshold.add_vline(
            x=50,

            line_dash="dash",

            line_color=muted,

            annotation_text=(
                "50% threshold"
            ),

            annotation_position=(
                "top"
            ),
        )


        fig_threshold.update_layout(
            height=180,

            margin=dict(
                l=10,
                r=20,
                t=40,
                b=25,
            ),

            xaxis=dict(
                range=[
                    0,
                    100,
                ],

                ticksuffix="%",

                title="",
            ),

            yaxis_title="",

            showlegend=False,
        )


        style_chart(
            fig_threshold
        )


        st.plotly_chart(
            fig_threshold,
            width="stretch",
            theme=None,

            config={
                "displayModeBar":
                    False
            },
        )


        st.caption(
            f"{distance_from_threshold:.1f} percentage points "
            f"from 50% · {threshold_label}"
        )


    # --------------------------------------------------------
    # SCORE RANGE
    # --------------------------------------------------------

    with range_col:

        section_title(
            "trending",
            "Expected Score Range",
        )


        lower_score = max(
            0,
            predicted_score
            - MODEL_MAE,
        )


        upper_score = min(
            100,
            predicted_score
            + MODEL_MAE,
        )


        fig_range = (
            go.Figure()
        )


        fig_range.add_trace(
            go.Scatter(
                x=[
                    lower_score,
                    upper_score,
                ],

                y=[
                    1,
                    1,
                ],

                mode="lines",

                line=dict(
                    width=12,
                    color=accent,
                ),

                showlegend=False,
            )
        )


        fig_range.add_trace(
            go.Scatter(
                x=[
                    predicted_score
                ],

                y=[
                    1
                ],

                mode=(
                    "markers+text"
                ),

                marker=dict(
                    size=15,
                    color=accent,
                ),

                text=[
                    f"{predicted_score:.1f}"
                ],

                textposition=(
                    "top center"
                ),

                showlegend=False,
            )
        )


        fig_range.update_layout(
            height=180,

            margin=dict(
                l=20,
                r=20,
                t=35,
                b=25,
            ),

            xaxis=dict(
                range=[
                    0,
                    100,
                ],

                title=(
                    "Exam Score"
                ),
            ),

            yaxis=dict(
                visible=False,

                range=[
                    0.78,
                    1.22,
                ],
            ),

            showlegend=False,
        )


        style_chart(
            fig_range
        )


        st.plotly_chart(
            fig_range,
            width="stretch",
            theme=None,

            config={
                "displayModeBar":
                    False
            },
        )


        st.caption(
            f"MAE-based interval: "
            f"{lower_score:.1f}–{upper_score:.1f} · "
            f"not a statistical confidence interval."
        )


    # ========================================================
    # LOCAL PREDICTION EXPLANATION
    # ========================================================

    section_title(
        "brain",
        "What Influenced This Prediction?",
    )


    linear_contributions = (
        model_contributions(
            linear_model,
            baseline_inputs,
            logistic=False,
        )
    )


    logistic_contributions = (
        model_contributions(
            logistic_model,
            baseline_inputs,
            logistic=True,
        )
    )


    influence_left, influence_right = (
        st.columns(2)
    )


    # --------------------------------------------------------
    # SCORE CONTRIBUTIONS
    # --------------------------------------------------------

    with influence_left:

        st.markdown(
            "#### Score Prediction"
        )


        linear_plot = (
            linear_contributions
            .sort_values(
                "Contribution"
            )
        )


        linear_colors = [
            (
                positive_color
                if value >= 0
                else negative_color
            )
            for value in (
                linear_plot[
                    "Contribution"
                ]
            )
        ]


        fig_linear_influence = (
            go.Figure(
                go.Bar(
                    x=linear_plot[
                        "Contribution"
                    ],

                    y=linear_plot[
                        "Feature"
                    ],

                    orientation="h",

                    marker_color=(
                        linear_colors
                    ),

                    text=[
                        f"{value:+.2f}"
                        for value in (
                            linear_plot[
                                "Contribution"
                            ]
                        )
                    ],

                    textposition=(
                        "outside"
                    ),
                )
            )
        )


        fig_linear_influence.add_vline(
            x=0,

            line_color=muted,

            line_dash="dash",
        )


        fig_linear_influence.update_layout(
            height=335,

            margin=dict(
                l=15,
                r=55,
                t=5,
                b=35,
            ),

            xaxis_title=(
                "Contribution to Predicted Score"
            ),

            yaxis_title="",

            showlegend=False,
        )


        style_chart(
            fig_linear_influence
        )


        st.plotly_chart(
            fig_linear_influence,
            width="stretch",
            theme=None,

            config={
                "displayModeBar":
                    False
            },
        )


        st.caption(
            "Positive values raise the score estimate relative "
            "to the model baseline; negative values lower it."
        )


    # --------------------------------------------------------
    # CLASSIFICATION CONTRIBUTIONS
    # --------------------------------------------------------

    with influence_right:

        st.markdown(
            "#### Pass / Fail Prediction"
        )


        logistic_plot = (
            logistic_contributions
            .sort_values(
                "Contribution"
            )
        )


        logistic_colors = [
            (
                positive_color
                if value >= 0
                else negative_color
            )
            for value in (
                logistic_plot[
                    "Contribution"
                ]
            )
        ]


        fig_logistic_influence = (
            go.Figure(
                go.Bar(
                    x=logistic_plot[
                        "Contribution"
                    ],

                    y=logistic_plot[
                        "Feature"
                    ],

                    orientation="h",

                    marker_color=(
                        logistic_colors
                    ),

                    text=[
                        f"{value:+.2f}"
                        for value in (
                            logistic_plot[
                                "Contribution"
                            ]
                        )
                    ],

                    textposition=(
                        "outside"
                    ),
                )
            )
        )


        fig_logistic_influence.add_vline(
            x=0,

            line_color=muted,

            line_dash="dash",
        )


        fig_logistic_influence.update_layout(
            height=335,

            margin=dict(
                l=15,
                r=55,
                t=5,
                b=35,
            ),

            xaxis_title=(
                "Contribution to Pass Log-Odds"
            ),

            yaxis_title="",

            showlegend=False,
        )


        style_chart(
            fig_logistic_influence
        )


        st.plotly_chart(
            fig_logistic_influence,
            width="stretch",
            theme=None,

            config={
                "displayModeBar":
                    False
            },
        )


        st.caption(
            "Positive values push the classifier toward Pass; "
            "negative values push it toward Fail."
        )


    st.info(
        "These charts explain how the trained models used this "
        "specific student profile. They describe model behavior "
        "and associations, not causal effects."
    )


    # ========================================================
    # IMPROVEMENT SCENARIO SIMULATOR
    # ========================================================

    section_title(
        "sliders",
        "Improvement Scenario Simulator",
    )


    st.caption(
        "Change four controllable study-related inputs and compare "
        "the model's prediction with the current result."
    )


    # --------------------------------------------------------
    # DEFAULT SCENARIO VALUES
    # --------------------------------------------------------

    if (
        "sim_attendance"
        not in st.session_state
    ):

        st.session_state.sim_attendance = (
            float(
                baseline_inputs[
                    "attendance_percentage"
                ]
            )
        )


    if (
        "sim_assignments"
        not in st.session_state
    ):

        st.session_state.sim_assignments = (
            float(
                baseline_inputs[
                    "assignment_completion_rate"
                ]
            )
        )


    if (
        "sim_study_hours"
        not in st.session_state
    ):

        st.session_state.sim_study_hours = (
            float(
                baseline_inputs[
                    "study_hours_per_day"
                ]
            )
        )


    if (
        "sim_practice_tests"
        not in st.session_state
    ):

        st.session_state.sim_practice_tests = (
            int(
                baseline_inputs[
                    "practice_tests_completed"
                ]
            )
        )


    # ========================================================
    # SCENARIO FORM
    # ========================================================

    with st.form(
        "improvement_scenario_form"
    ):

        (
            sim_col1,
            sim_col2,
            sim_col3,
            sim_col4,
        ) = st.columns(4)


        with sim_col1:

            sim_attendance = (
                st.number_input(
                    "Attendance · 0–100%",

                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,

                    key=(
                        "sim_attendance"
                    ),
                )
            )


        with sim_col2:

            sim_assignments = (
                st.number_input(
                    "Assignments · 0–100%",

                    min_value=0.0,
                    max_value=100.0,
                    step=1.0,

                    key=(
                        "sim_assignments"
                    ),
                )
            )


        with sim_col3:

            sim_study_hours = (
                st.number_input(
                    "Study Hours · 0–12",

                    min_value=0.0,
                    max_value=12.0,
                    step=0.5,

                    key=(
                        "sim_study_hours"
                    ),
                )
            )


        with sim_col4:

            sim_practice_tests = (
                st.number_input(
                    "Practice Tests · 0–17",

                    min_value=0,
                    max_value=17,
                    step=1,

                    key=(
                        "sim_practice_tests"
                    ),
                )
            )


        simulate_button = (
            st.form_submit_button(
                "Simulate Scenario",

                type="primary",

                width="stretch",
            )
        )


    # ========================================================
    # RUN SCENARIO
    # ========================================================

    if simulate_button:

        scenario_inputs = {

            "previous_exam_score":
                baseline_inputs[
                    "previous_exam_score"
                ],

            "previous_gpa":
                baseline_inputs[
                    "previous_gpa"
                ],

            "attendance_percentage":
                sim_attendance,

            "assignment_completion_rate":
                sim_assignments,

            "study_hours_per_day":
                sim_study_hours,

            "practice_tests_completed":
                sim_practice_tests,
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
                scenario_inputs,
        }


        st.session_state.scroll_target = (
            "scenario-results-anchor"
        )


    # ========================================================
    # SCENARIO RESULT
    # ========================================================

    if (
        st.session_state.scenario_result
        is not None
    ):

        st.html(
            """
            <div
                id="scenario-results-anchor"
                class="scroll-anchor"
            ></div>
            """
        )


        if (
            st.session_state.scroll_target
            == "scenario-results-anchor"
        ):

            scroll_to_anchor(
                "scenario-results-anchor"
            )

            st.session_state.scroll_target = (
                None
            )


        scenario_data = (
            st.session_state.scenario_result
        )


        scenario_prediction = (
            scenario_data[
                "prediction"
            ]
        )


        scenario_inputs = (
            scenario_data[
                "inputs"
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


        scenario_probability = (
            float(
                scenario_prediction[
                    "pass_probability"
                ]
            )
            * 100
        )


        score_change = (
            scenario_score
            - predicted_score
        )


        probability_change = (
            scenario_probability
            - probability_percent
        )


        (
            score_css,
            score_change_text,
        ) = format_change(
            score_change,
            " points",
        )


        (
            probability_css,
            probability_change_text,
        ) = format_change(
            probability_change,
            " pp",
        )


        # ====================================================
        # DETECT INPUT CHANGES
        # ====================================================

        scenario_changed = any(
            [
                abs(
                    scenario_inputs[
                        "attendance_percentage"
                    ]
                    - baseline_inputs[
                        "attendance_percentage"
                    ]
                ) > 1e-9,

                abs(
                    scenario_inputs[
                        "assignment_completion_rate"
                    ]
                    - baseline_inputs[
                        "assignment_completion_rate"
                    ]
                ) > 1e-9,

                abs(
                    scenario_inputs[
                        "study_hours_per_day"
                    ]
                    - baseline_inputs[
                        "study_hours_per_day"
                    ]
                ) > 1e-9,

                int(
                    scenario_inputs[
                        "practice_tests_completed"
                    ]
                )
                != int(
                    baseline_inputs[
                        "practice_tests_completed"
                    ]
                ),
            ]
        )


        # ====================================================
        # SCENARIO CARDS
        # ====================================================

        st.markdown(
            "#### Scenario Comparison"
        )


        compare1, compare2, compare3 = (
            st.columns(3)
        )


        with compare1:

            st.html(
                f"""
                <div class="scenario-card">

                    <div class="scenario-label">
                        Current Prediction
                    </div>

                    <div class="scenario-value">
                        {predicted_score:.1f} / 100
                    </div>

                    <div class="scenario-note">
                        {predicted_status} ·
                        {probability_percent:.1f}% probability
                    </div>

                </div>
                """
            )


        with compare2:

            st.html(
                f"""
                <div class="scenario-card">

                    <div class="scenario-label">
                        Scenario Prediction
                    </div>

                    <div class="scenario-value">
                        {scenario_score:.1f} / 100
                    </div>

                    <div class="scenario-note">
                        {scenario_status} ·
                        {scenario_probability:.1f}% probability
                    </div>

                </div>
                """
            )


        with compare3:

            st.html(
                f"""
                <div class="scenario-card">

                    <div class="scenario-label">
                        Model-Predicted Change
                    </div>

                    <div class="{score_css}">
                        Score:
                        {score_change_text}
                    </div>

                    <div
                        class="{probability_css}"
                        style="margin-top:8px;"
                    >
                        Pass probability:
                        {probability_change_text}
                    </div>

                </div>
                """
            )


        # ====================================================
        # NO CHANGE MESSAGE
        # ====================================================

        if not scenario_changed:

            st.html(
                """
                <div class="no-change-box">

                    <b>
                        No scenario changes detected
                    </b>

                    <br>

                    The scenario inputs are identical to the
                    current student profile. Adjust one or more
                    values above and run the simulator again.

                </div>
                """
            )


        # ====================================================
        # SCENARIO VISUALS
        # ====================================================

        else:

            (
                scenario_visual1,
                scenario_visual2,
            ) = st.columns(2)


            # ------------------------------------------------
            # OUTPUT CHANGE
            # ------------------------------------------------

            with scenario_visual1:

                scenario_output_df = (
                    pd.DataFrame(
                        {
                            "State": [
                                "Current",
                                "Scenario",
                            ],

                            "Score": [
                                predicted_score,
                                scenario_score,
                            ],

                            "Pass Probability": [
                                probability_percent,
                                scenario_probability,
                            ],
                        }
                    )
                )


                scenario_output_long = (
                    scenario_output_df
                    .melt(
                        id_vars=(
                            "State"
                        ),

                        value_vars=[
                            "Score",
                            "Pass Probability",
                        ],

                        var_name=(
                            "Metric"
                        ),

                        value_name=(
                            "Value"
                        ),
                    )
                )


                fig_scenario_output = (
                    px.bar(
                        scenario_output_long,

                        x="Metric",
                        y="Value",

                        color="State",

                        barmode=(
                            "group"
                        ),

                        text="Value",
                    )
                )


                fig_scenario_output.update_traces(
                    texttemplate=(
                        "%{text:.1f}"
                    ),

                    textposition=(
                        "outside"
                    ),
                )


                fig_scenario_output.update_layout(
                    height=320,

                    margin=dict(
                        l=20,
                        r=20,
                        t=35,
                        b=25,
                    ),

                    title=dict(
                        text=(
                            "Prediction Before vs Scenario"
                        ),

                        font=dict(
                            size=14
                        ),
                    ),

                    yaxis=dict(
                        range=[
                            0,
                            100,
                        ],

                        title="Value",
                    ),

                    xaxis_title="",

                    legend_title="",
                )


                style_chart(
                    fig_scenario_output
                )


                st.plotly_chart(
                    fig_scenario_output,
                    width="stretch",
                    theme=None,

                    config={
                        "displayModeBar":
                            False
                    },
                )


            # ------------------------------------------------
            # INPUT CHANGE
            # ------------------------------------------------

            with scenario_visual2:

                input_comparison_df = (
                    pd.DataFrame(
                        {
                            "Feature": [
                                "Attendance",
                                "Assignments",
                                "Study Hours",
                                "Practice Tests",
                            ],

                            "Current": [
                                baseline_inputs[
                                    "attendance_percentage"
                                ],

                                baseline_inputs[
                                    "assignment_completion_rate"
                                ],

                                (
                                    baseline_inputs[
                                        "study_hours_per_day"
                                    ]
                                    / 12
                                )
                                * 100,

                                (
                                    baseline_inputs[
                                        "practice_tests_completed"
                                    ]
                                    / 17
                                )
                                * 100,
                            ],

                            "Scenario": [
                                scenario_inputs[
                                    "attendance_percentage"
                                ],

                                scenario_inputs[
                                    "assignment_completion_rate"
                                ],

                                (
                                    scenario_inputs[
                                        "study_hours_per_day"
                                    ]
                                    / 12
                                )
                                * 100,

                                (
                                    scenario_inputs[
                                        "practice_tests_completed"
                                    ]
                                    / 17
                                )
                                * 100,
                            ],
                        }
                    )
                )


                input_long = (
                    input_comparison_df
                    .melt(
                        id_vars=(
                            "Feature"
                        ),

                        value_vars=[
                            "Current",
                            "Scenario",
                        ],

                        var_name=(
                            "State"
                        ),

                        value_name=(
                            "Level"
                        ),
                    )
                )


                fig_input_change = (
                    px.bar(
                        input_long,

                        x="Feature",
                        y="Level",

                        color="State",

                        barmode=(
                            "group"
                        ),
                    )
                )


                fig_input_change.update_layout(
                    height=320,

                    margin=dict(
                        l=20,
                        r=20,
                        t=35,
                        b=25,
                    ),

                    title=dict(
                        text=(
                            "Input Changes"
                        ),

                        font=dict(
                            size=14
                        ),
                    ),

                    yaxis=dict(
                        range=[
                            0,
                            100,
                        ],

                        title=(
                            "Normalized Level (%)"
                        ),

                        ticksuffix="%",
                    ),

                    xaxis_title="",

                    legend_title="",
                )


                style_chart(
                    fig_input_change
                )


                st.plotly_chart(
                    fig_input_change,
                    width="stretch",
                    theme=None,

                    config={
                        "displayModeBar":
                            False
                    },
                )


            st.caption(
                "Scenario differences are changes in model predictions. "
                "They do not establish that changing these variables will "
                "cause the same real-world improvement."
            )


    # ========================================================
    # MODEL DETAILS
    # ========================================================

    section_title(
        "database",
        "How This Prediction Works",
    )


    with st.expander(
        "View model and evaluation details"
    ):

        detail1, detail2, detail3 = (
            st.columns(3)
        )


        detail1.metric(
            "Linear Model MAE",
            "9.38 points",
        )


        detail2.metric(
            "Balanced Accuracy",
            "73.2%",
        )


        detail3.metric(
            "Classification AUC",
            "0.814",
        )


        st.markdown(
            """
            **Six inputs → two independent prediction models**

            **Linear Regression →** predicted exam score

            **Balanced Logistic Regression →** Pass / Fail +
            pass probability
            """
        )


        st.info(
            "The predicted exam score does not directly determine "
            "the Pass / Fail probability because the outputs come "
            "from two separately trained models."
        )


# ============================================================
# FOOTER
# ============================================================

show_footer()