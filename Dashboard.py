# ============================================================
# STUDENT PERFORMANCE ANALYTICS - DASHBOARD
# Compact visual-first final version
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split

from components.footer import show_footer

from components.styles import (
    apply_global_styles,
    get_plotly_template,
    get_theme_colors,
)

from components.icons import (
    page_title,
    section_title,
    icon_card,
)

from components.ml_pipeline_story import (
    render_ml_pipeline_story,
)

from core.model_loader import (
    load_linear_model,
    load_logistic_model,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
)

apply_global_styles()


# ============================================================
# DATA LOADING
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "student_performance.csv"
)


@st.cache_data
def load_data():

    return pd.read_csv(
        DATA_PATH
    )


df = load_data()


# ============================================================
# THEME
# ============================================================

plotly_template = get_plotly_template()
theme_colors = get_theme_colors()

chart_background = theme_colors["background"]
surface = theme_colors["surface"]
chart_text = theme_colors["text"]
chart_muted = theme_colors["muted"]
chart_border = theme_colors["border"]
accent = theme_colors["accent"]

positive_color = "#5FA879"
negative_color = "#C86B6B"
warning_color = "#C89C55"


# ============================================================
# PAGE CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .block-container {{
        padding-top: 1.7rem;
        padding-bottom: 2rem;
    }}

    /* --------------------------------------------------------
       SMALL SUPPORT TEXT
    -------------------------------------------------------- */

    .dashboard-note {{
        color: {chart_muted};
        font-size: 0.78rem;
        line-height: 1.45;
    }}


    /* --------------------------------------------------------
       NAVIGATION CARDS
    -------------------------------------------------------- */

    .nav-card {{
        background: {surface};
        border: 1px solid {chart_border};
        border-radius: 14px;
        padding: 17px 17px 13px 17px;
        min-height: 116px;
        margin-bottom: 8px;
    }}

    .nav-label {{
        color: {accent};
        font-size: 0.66rem;
        font-weight: 750;
        letter-spacing: 0.06rem;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}

    .nav-title {{
        color: {chart_text};
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 5px;
    }}

    .nav-description {{
        color: {chart_muted};
        font-size: 0.77rem;
        line-height: 1.45;
    }}


    /* --------------------------------------------------------
       MODEL PERFORMANCE CARDS
    -------------------------------------------------------- */

    .model-panel {{
        background: {surface};
        border: 1px solid {chart_border};
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 135px;
    }}

    .model-panel-label {{
        color: {accent};
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.05rem;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    .model-panel-title {{
        color: {chart_text};
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 4px;
    }}

    .model-panel-text {{
        color: {chart_muted};
        font-size: 0.77rem;
        line-height: 1.45;
    }}


    /* --------------------------------------------------------
       AUDIT CARDS
    -------------------------------------------------------- */

    .audit-card {{
        background: {surface};
        border: 1px solid {chart_border};
        border-radius: 14px;
        padding: 17px 18px;
        min-height: 135px;
    }}

    .audit-label {{
        color: {accent};
        font-size: 0.67rem;
        font-weight: 750;
        letter-spacing: 0.05rem;
        text-transform: uppercase;
        margin-bottom: 5px;
    }}

    .audit-title {{
        color: {chart_text};
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 6px;
    }}

    .audit-text {{
        color: {chart_muted};
        font-size: 0.78rem;
        line-height: 1.5;
    }}


    /* --------------------------------------------------------
       DATA SUMMARY
    -------------------------------------------------------- */

    .dataset-health {{
        background: {surface};
        border: 1px solid {chart_border};
        border-radius: 11px;
        padding: 10px 14px;
        color: {chart_muted};
        font-size: 0.76rem;
        line-height: 1.45;
        margin-top: 4px;
        margin-bottom: 10px;
    }}

    .dataset-health b {{
        color: {chart_text};
        font-weight: 650;
    }}


    /* --------------------------------------------------------
       RESPONSIVE
    -------------------------------------------------------- */

    @media(max-width: 900px) {{

        .nav-card,
        .model-panel,
        .audit-card {{
            min-height: auto;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# COMMON PLOTLY STYLE
# ============================================================

def style_chart(fig):

    fig.update_layout(
        template=plotly_template,
        paper_bgcolor=chart_background,
        plot_bgcolor=chart_background,

        font=dict(
            color=chart_text
        ),
    )

    fig.update_xaxes(
        color=chart_text,
        gridcolor=chart_border,
        zerolinecolor=chart_border,
    )

    fig.update_yaxes(
        color=chart_text,
        gridcolor=chart_border,
        zerolinecolor=chart_border,
    )

    return fig


# ============================================================
# MODEL CONFIGURATION
# ============================================================

features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed",
]

regression_target = (
    "exam_score"
)

classification_target = (
    "pass_status"
)


# ============================================================
# MODEL RESOURCES
# ============================================================

@st.cache_resource
def get_linear_model():

    return load_linear_model()


@st.cache_resource
def get_logistic_model():

    return load_logistic_model()


# ============================================================
# REAL MODEL METRICS
# ============================================================

@st.cache_data
def calculate_dashboard_metrics():

    X = df[
        features
    ].copy()


    # --------------------------------------------------------
    # LINEAR REGRESSION
    # --------------------------------------------------------

    y_reg = df[
        regression_target
    ].copy()


    (
        _,
        X_test_reg,
        _,
        y_test_reg,
    ) = train_test_split(
        X,
        y_reg,
        test_size=0.20,
        random_state=42,
    )


    linear_model = (
        get_linear_model()
    )


    y_pred_reg = (
        linear_model.predict(
            X_test_reg
        )
    )


    linear_mae = (
        mean_absolute_error(
            y_test_reg,
            y_pred_reg,
        )
    )


    linear_rmse = (
        np.sqrt(
            mean_squared_error(
                y_test_reg,
                y_pred_reg,
            )
        )
    )


    linear_r2 = (
        r2_score(
            y_test_reg,
            y_pred_reg,
        )
    )


    # --------------------------------------------------------
    # BALANCED LOGISTIC REGRESSION
    # --------------------------------------------------------

    y_cls = (
        df[
            classification_target
        ]
        .astype(str)
    )


    (
        _,
        X_test_cls,
        _,
        y_test_cls,
    ) = train_test_split(
        X,
        y_cls,
        test_size=0.20,
        random_state=42,
        stratify=y_cls,
    )


    logistic_model = (
        get_logistic_model()
    )


    y_pred_cls = (
        logistic_model.predict(
            X_test_cls
        )
    )


    logistic_accuracy = (
        accuracy_score(
            y_test_cls,
            y_pred_cls,
        )
    )


    logistic_balanced_accuracy = (
        balanced_accuracy_score(
            y_test_cls,
            y_pred_cls,
        )
    )


    logistic_f1 = (
        f1_score(
            y_test_cls,
            y_pred_cls,
            pos_label="Pass",
            zero_division=0,
        )
    )


    return {

        "linear_mae":
            linear_mae,

        "linear_rmse":
            linear_rmse,

        "linear_r2":
            linear_r2,

        "logistic_accuracy":
            logistic_accuracy,

        "logistic_balanced_accuracy":
            logistic_balanced_accuracy,

        "logistic_f1":
            logistic_f1,
    }


model_metrics = (
    calculate_dashboard_metrics()
)


# ============================================================
# DATASET SUMMARY VALUES
# ============================================================

records = len(
    df
)

columns = (
    df.shape[1]
)

missing_values = int(
    df.isnull()
    .sum()
    .sum()
)

duplicates = int(
    df.duplicated()
    .sum()
)


# ============================================================
# HEADER
# ============================================================

page_title(
    "dashboard",
    "Student Performance Analytics",
    "Explore the complete machine-learning workflow, understand the "
    "models, and generate interactive student performance predictions.",
)


# ============================================================
# PROJECT AT A GLANCE
# ============================================================

section_title(
    "sparkles",
    "Project at a Glance",
)


glance1, glance2, glance3, glance4 = (
    st.columns(4)
)


glance1.metric(
    "Student Records",
    f"{records:,}",
)


glance2.metric(
    "Model Predictors",
    "6",
)


glance3.metric(
    "ML Models",
    "2",
)


glance4.metric(
    "Prediction Outputs",
    "3",
)


st.html(
    f"""
    <div class="dataset-health">

        Dataset health ·

        <b>{columns}</b> variables ·

        <b>{missing_values:,}</b> missing cells ·

        <b>{duplicates:,}</b> duplicate rows

        &nbsp;&nbsp;|&nbsp;&nbsp;

        Outputs:
        <b>Exam Score</b> ·
        <b>Pass / Fail</b> ·
        <b>Pass Probability</b>

    </div>
    """
)


# ============================================================
# FROZEN ML STORY
# ============================================================

section_title(
    "sparkles",
    "How the Machine-Learning System Works",
)


st.caption(
    "Follow the complete path from raw student data to an "
    "interactive prediction."
)


# IMPORTANT:
# Keep the existing pipeline component unchanged.
render_ml_pipeline_story(
    theme_colors
)


# ============================================================
# EXPLORE PROJECT
# ============================================================

section_title(
    "dashboard",
    "Explore the Project",
)


nav1, nav2, nav3, nav4 = (
    st.columns(4)
)


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------

with nav1:

    st.html(
        """
        <div class="nav-card">

            <div class="nav-label">
                Step 1
            </div>

            <div class="nav-title">
                Explore Data
            </div>

            <div class="nav-description">
                Inspect the dataset, distributions, missing values,
                variables, and exploratory analysis.
            </div>

        </div>
        """
    )


    if st.button(
        "Dataset & EDA",
        key="nav_eda",
        width="stretch",
    ):

        st.switch_page(
            "pages/01_Dataset_EDA.py"
        )


# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

with nav2:

    st.html(
        """
        <div class="nav-card">

            <div class="nav-label">
                Step 2
            </div>

            <div class="nav-title">
                Prepare Data
            </div>

            <div class="nav-description">
                Explore train/test splitting, median imputation,
                standardization, and leakage protection.
            </div>

        </div>
        """
    )


    if st.button(
        "Preprocessing",
        key="nav_preprocessing",
        width="stretch",
    ):

        st.switch_page(
            "pages/02_Preprocessing.py"
        )


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

with nav3:

    st.html(
        """
        <div class="nav-card">

            <div class="nav-label">
                Step 3
            </div>

            <div class="nav-title">
                Analyze Models
            </div>

            <div class="nav-description">
                Study regression, classification, model comparison,
                and evaluation performance.
            </div>

        </div>
        """
    )


    if st.button(
        "Model Comparison",
        key="nav_models",
        width="stretch",
    ):

        st.switch_page(
            "pages/05_Model_Comparison.py"
        )


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

with nav4:

    st.html(
        """
        <div class="nav-card">

            <div class="nav-label">
                Step 4
            </div>

            <div class="nav-title">
                Make a Prediction
            </div>

            <div class="nav-description">
                Enter six student variables and generate score,
                Pass / Fail, and probability predictions.
            </div>

        </div>
        """
    )


    if st.button(
        "Predict Performance",
        key="nav_predict",
        type="primary",
        width="stretch",
    ):

        st.switch_page(
            "pages/06_Predict_Performance.py"
        )


# ============================================================
# DATASET INSIGHTS
# ============================================================

section_title(
    "database",
    "Dataset Insights",
)


chart1, chart2 = (
    st.columns(
        [
            1.55,
            1,
        ]
    )
)


# ------------------------------------------------------------
# EXAM SCORE DISTRIBUTION
# ------------------------------------------------------------

with chart1:

    st.markdown(
        "#### Exam Score Distribution"
    )


    fig_score = px.histogram(
        df,
        x="exam_score",
        nbins=25,
    )


    fig_score.update_traces(
        marker_color=accent
    )


    fig_score.update_layout(
        height=315,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),

        xaxis_title=(
            "Exam Score"
        ),

        yaxis_title=(
            "Students"
        ),

        showlegend=False,
    )


    style_chart(
        fig_score
    )


    st.plotly_chart(
        fig_score,
        width="stretch",
        theme=None,

        config={
            "displayModeBar":
                False
        },
    )


# ------------------------------------------------------------
# PASS / FAIL
# ------------------------------------------------------------

with chart2:

    st.markdown(
        "#### Pass / Fail Distribution"
    )


    class_counts = (
        df[
            "pass_status"
        ]
        .value_counts()
    )


    pass_count = int(
        class_counts.get(
            "Pass",
            0,
        )
    )


    fail_count = int(
        class_counts.get(
            "Fail",
            0,
        )
    )


    class_df = pd.DataFrame(
        {
            "Status": [
                "Pass",
                "Fail",
            ],

            "Students": [
                pass_count,
                fail_count,
            ],
        }
    )


    fig_class = go.Figure(
        data=[
            go.Pie(
                labels=class_df[
                    "Status"
                ],

                values=class_df[
                    "Students"
                ],

                hole=0.66,

                marker=dict(
                    colors=[
                        positive_color,
                        negative_color,
                    ],

                    line=dict(
                        color=chart_background,
                        width=2,
                    ),
                ),

                textinfo=(
                    "label+percent"
                ),

                textposition=(
                    "inside"
                ),

                hovertemplate=(
                    "%{label}"
                    "<br>"
                    "%{value:,} students"
                    "<br>"
                    "%{percent}"
                    "<extra></extra>"
                ),
            )
        ]
    )


    fig_class.add_annotation(
        x=0.5,
        y=0.56,

        text=(
            f"<b>{records:,}</b>"
        ),

        showarrow=False,

        font=dict(
            size=19,
            color=chart_text,
        ),
    )


    fig_class.add_annotation(
        x=0.5,
        y=0.43,

        text=(
            "Students"
        ),

        showarrow=False,

        font=dict(
            size=10,
            color=chart_muted,
        ),
    )


    fig_class.update_layout(
        template=plotly_template,

        height=315,

        margin=dict(
            l=5,
            r=5,
            t=5,
            b=5,
        ),

        showlegend=False,

        paper_bgcolor=(
            chart_background
        ),

        plot_bgcolor=(
            chart_background
        ),

        font=dict(
            color=chart_text
        ),
    )


    st.plotly_chart(
        fig_class,
        width="stretch",
        theme=None,

        config={
            "displayModeBar":
                False
        },
    )


    st.caption(
        f"Pass: {pass_count:,} ({pass_count / records * 100:.1f}%) · "
        f"Fail: {fail_count:,} ({fail_count / records * 100:.1f}%)"
    )


# ============================================================
# FEATURE CORRELATION
# ============================================================

selected_features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed",
]


feature_labels = {

    "previous_exam_score":
        "Previous Exam",

    "previous_gpa":
        "Previous GPA",

    "attendance_percentage":
        "Attendance",

    "assignment_completion_rate":
        "Assignments",

    "study_hours_per_day":
        "Study Hours",

    "practice_tests_completed":
        "Practice Tests",
}


available_features = [
    feature
    for feature
    in selected_features
    if feature in df.columns
]


if (
    "exam_score" in df.columns
    and available_features
):

    section_title(
        "chart",
        "Which Inputs Relate Most to Exam Score?",
    )


    st.caption(
        "Pearson correlation between each of the six selected predictors "
        "and the final exam score."
    )


    correlation_data = (
        df[
            available_features
            + [
                "exam_score"
            ]
        ]
        .corr(
            numeric_only=True
        )[
            "exam_score"
        ]
        .drop(
            "exam_score"
        )
        .sort_values()
        .reset_index()
    )


    correlation_data.columns = [
        "Feature",
        "Correlation",
    ]


    correlation_data[
        "Feature"
    ] = (
        correlation_data[
            "Feature"
        ]
        .map(
            feature_labels
        )
    )


    correlation_colors = [
        (
            positive_color
            if value >= 0
            else negative_color
        )

        for value in (
            correlation_data[
                "Correlation"
            ]
        )
    ]


    fig_corr = go.Figure()


    fig_corr.add_trace(
        go.Bar(
            x=correlation_data[
                "Correlation"
            ],

            y=correlation_data[
                "Feature"
            ],

            orientation="h",

            marker_color=(
                correlation_colors
            ),

            text=correlation_data[
                "Correlation"
            ],

            texttemplate=(
                "%{text:.2f}"
            ),

            textposition=(
                "outside"
            ),

            hovertemplate=(
                "%{y}"
                "<br>"
                "Correlation: %{x:.3f}"
                "<extra></extra>"
            ),
        )
    )


    fig_corr.add_vline(
        x=0,
        line_dash="dash",
        line_color=chart_muted,
    )


    fig_corr.update_layout(
        height=315,

        margin=dict(
            l=20,
            r=65,
            t=10,
            b=20,
        ),

        xaxis_title=(
            "Correlation with Exam Score"
        ),

        yaxis_title="",

        showlegend=False,
    )


    style_chart(
        fig_corr
    )


    st.plotly_chart(
        fig_corr,
        width="stretch",
        theme=None,

        config={
            "displayModeBar":
                False
        },
    )


    st.caption(
        "Positive values indicate that higher feature values tend to "
        "occur with higher exam scores. Correlation does not imply causation."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

section_title(
    "brain",
    "Model Performance",
)


model_left, model_right = (
    st.columns(2)
)


# ============================================================
# LINEAR REGRESSION PANEL
# ============================================================

with model_left:

    st.html(
        f"""
        <div class="model-panel">

            <div class="model-panel-label">
                Regression Model
            </div>

            <div class="model-panel-title">
                Linear Regression
            </div>

            <div class="model-panel-text">
                Predicts a continuous exam score from the six
                pre-exam student variables.
            </div>

        </div>
        """
    )


    lr1, lr2, lr3 = (
        st.columns(3)
    )


    lr1.metric(
        "MAE",
        f"{model_metrics['linear_mae']:.2f}",
    )


    lr2.metric(
        "RMSE",
        f"{model_metrics['linear_rmse']:.2f}",
    )


    lr3.metric(
        "R²",
        f"{model_metrics['linear_r2']:.3f}",
    )


    r2_percent = (
        max(
            0,
            model_metrics[
                "linear_r2"
            ]
        )
        * 100
    )


    fig_r2 = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=r2_percent,

            number={
                "suffix":
                    "%",

                "font": {
                    "size":
                        28
                },
            },

            title={
                "text":
                    "Variance Explained"
            },

            gauge={
                "axis": {
                    "range": [
                        0,
                        100,
                    ]
                },

                "bar": {
                    "color":
                        accent,

                    "thickness":
                        0.24,
                },

                "bgcolor":
                    surface,

                "bordercolor":
                    chart_border,
            },
        )
    )


    fig_r2.update_layout(
        height=205,

        margin=dict(
            l=25,
            r=25,
            t=35,
            b=5,
        ),

        paper_bgcolor=(
            chart_background
        ),

        font=dict(
            color=chart_text
        ),
    )


    st.plotly_chart(
        fig_r2,
        width="stretch",
        theme=None,

        config={
            "displayModeBar":
                False
        },
    )


# ============================================================
# LOGISTIC REGRESSION PANEL
# ============================================================

with model_right:

    st.html(
        f"""
        <div class="model-panel">

            <div class="model-panel-label">
                Classification Model
            </div>

            <div class="model-panel-title">
                Balanced Logistic Regression
            </div>

            <div class="model-panel-text">
                Predicts Pass / Fail and estimates the probability
                of passing while accounting for class imbalance.
            </div>

        </div>
        """
    )


    cls1, cls2, cls3 = (
        st.columns(3)
    )


    cls1.metric(
        "Accuracy",
        (
            f"{model_metrics['logistic_accuracy'] * 100:.1f}%"
        ),
    )


    cls2.metric(
        "Balanced Acc.",
        (
            f"{model_metrics['logistic_balanced_accuracy'] * 100:.1f}%"
        ),
    )


    cls3.metric(
        "F1",
        (
            f"{model_metrics['logistic_f1'] * 100:.1f}%"
        ),
    )


    balanced_accuracy_percent = (
        model_metrics[
            "logistic_balanced_accuracy"
        ]
        * 100
    )


    fig_ba = go.Figure(
        go.Indicator(
            mode="gauge+number",

            value=(
                balanced_accuracy_percent
            ),

            number={
                "suffix":
                    "%",

                "font": {
                    "size":
                        28
                },
            },

            title={
                "text":
                    "Balanced Accuracy"
            },

            gauge={
                "axis": {
                    "range": [
                        0,
                        100,
                    ]
                },

                "bar": {
                    "color":
                        positive_color,

                    "thickness":
                        0.24,
                },

                "bgcolor":
                    surface,

                "bordercolor":
                    chart_border,
            },
        )
    )


    fig_ba.update_layout(
        height=205,

        margin=dict(
            l=25,
            r=25,
            t=35,
            b=5,
        ),

        paper_bgcolor=(
            chart_background
        ),

        font=dict(
            color=chart_text
        ),
    )


    st.plotly_chart(
        fig_ba,
        width="stretch",
        theme=None,

        config={
            "displayModeBar":
                False
        },
    )


st.caption(
    "Performance values are calculated directly from the saved models "
    "using the same held-out evaluation splits used throughout the app."
)


# ============================================================
# MODEL AUDIT & ADVANCED TOOLS
# ============================================================

section_title(
    "scale",
    "Model Audit & Advanced Tools",
)


audit1, audit2 = (
    st.columns(2)
)


# ------------------------------------------------------------
# FAIRNESS / SUBGROUP AUDIT
# ------------------------------------------------------------

with audit1:

    st.html(
        """
        <div class="audit-card">

            <div class="audit-label">
                Model Audit
            </div>

            <div class="audit-title">
                Fairness & Subgroup Analysis
            </div>

            <div class="audit-text">
                Compare regression error, Balanced Accuracy,
                Pass Recall, Fail Recall, and subgroup size across
                gender, education level, school type, family income,
                and urban/rural groups.
            </div>

        </div>
        """
    )


    if st.button(
        "Open Fairness Analysis",
        key="open_fairness",
        width="stretch",
    ):

        st.switch_page(
            "pages/05_Model_Comparison.py"
        )


# ------------------------------------------------------------
# THRESHOLD EXPLORER
# ------------------------------------------------------------

with audit2:

    st.html(
        """
        <div class="audit-card">

            <div class="audit-label">
                Interactive Evaluation
            </div>

            <div class="audit-title">
                Classification Threshold Explorer
            </div>

            <div class="audit-text">
                Move the decision threshold and observe how
                Balanced Accuracy, precision, recall, false positives,
                and false negatives change on the held-out test set.
            </div>

        </div>
        """
    )


    if st.button(
        "Explore Classification Threshold",
        key="open_threshold",
        width="stretch",
    ):

        st.switch_page(
            "pages/07_Threshold_Explorer.py"
        )


# ============================================================
# FINAL CTA
# ============================================================

section_title(
    "sparkles",
    "Try the Prediction System",
)


cta_left, cta_center, cta_right = (
    st.columns(
        [
            1,
            2,
            1,
        ]
    )
)


with cta_center:

    st.caption(
        "Enter your own student profile to generate an exam-score "
        "estimate, Pass / Fail classification, pass probability, "
        "feature-level prediction explanation, and improvement scenario."
    )


    if st.button(
        "Try Student Performance Prediction",
        key="final_prediction_cta",
        type="primary",
        width="stretch",
    ):

        st.switch_page(
            "pages/06_Predict_Performance.py"
        )


# ============================================================
# FOOTER
# ============================================================

show_footer()