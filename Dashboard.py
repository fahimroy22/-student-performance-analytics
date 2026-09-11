# ============================================================
# STUDENT PERFORMANCE ANALYTICS - DASHBOARD
# Final visual-first version
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
    render_system_status,
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
# THEME SETTINGS
# ============================================================

plotly_template = get_plotly_template()
theme_colors = get_theme_colors()

chart_background = theme_colors["background"]
surface = theme_colors["surface"]
chart_text = theme_colors["text"]
chart_muted = theme_colors["muted"]
chart_border = theme_colors["border"]
accent = theme_colors["accent"]


# ============================================================
# PAGE CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }}

    .dashboard-note {{
        color:{chart_muted};
        font-size:11px;
        line-height:1.45;
    }}

    .class-summary {{
        background:{surface};
        border:1px solid {chart_border};
        border-radius:12px;
        padding:14px 15px;
        min-height:105px;
    }}

    .class-summary-label {{
        color:{chart_muted};
        font-size:11px;
        margin-bottom:6px;
    }}

    .class-summary-value {{
        color:{chart_text};
        font-size:21px;
        font-weight:700;
        margin-bottom:4px;
    }}

    .class-summary-note {{
        color:{chart_muted};
        font-size:11px;
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
    # LOGISTIC REGRESSION
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
# HEADER
# ============================================================

page_title(
    "dashboard",
    "Student Performance Analytics",
    "Explore the complete machine-learning workflow, model performance, "
    "and interactive student predictions.",
)


# ============================================================
# DATASET SNAPSHOT
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


col1, col2, col3, col4 = (
    st.columns(4)
)


col1.metric(
    "Student Records",
    f"{records:,}",
)

col2.metric(
    "Variables",
    columns,
)

col3.metric(
    "Missing Values",
    f"{missing_values:,}",
)

col4.metric(
    "Duplicates",
    duplicates,
)


# ============================================================
# FROZEN ML STORY
# ============================================================

section_title(
    "sparkles",
    "How the Machine-Learning System Works",
)


st.caption(
    "Follow the full path from raw student data to an interactive prediction."
)


# Keep this component unchanged.
render_ml_pipeline_story(
    theme_colors
)


# ============================================================
# SYSTEM STATUS
# ============================================================

section_title(
    "settings",
    "System Status",
)


render_system_status(
    theme_colors=theme_colors,
    records=records,
    variables=columns,
    missing_values=missing_values,
)


# ============================================================
# QUICK NAVIGATION
# ============================================================

section_title(
    "dashboard",
    "Explore the Project",
)


nav1, nav2, nav3 = (
    st.columns(3)
)


with nav1:

    if st.button(
        "Dataset & EDA",
        width="stretch",
    ):

        st.switch_page(
            "pages/01_Dataset_EDA.py"
        )


with nav2:

    if st.button(
        "Preprocessing",
        width="stretch",
    ):

        st.switch_page(
            "pages/02_Preprocessing.py"
        )


with nav3:

    if st.button(
        "Linear Regression",
        width="stretch",
    ):

        st.switch_page(
            "pages/03_Linear_Regression.py"
        )


nav4, nav5, nav6 = (
    st.columns(3)
)


with nav4:

    if st.button(
        "Logistic Regression",
        width="stretch",
    ):

        st.switch_page(
            "pages/04_Logistic_Regression.py"
        )


with nav5:

    if st.button(
        "Model Comparison",
        width="stretch",
    ):

        st.switch_page(
            "pages/05_Model_Comparison.py"
        )


with nav6:

    if st.button(
        "Predict Performance",
        type="primary",
        width="stretch",
    ):

        st.switch_page(
            "pages/06_Predict_Performance.py"
        )


threshold_col1, threshold_col2, threshold_col3 = (
    st.columns(
        [
            1,
            1.5,
            1,
        ]
    )
)


with threshold_col2:

    if st.button(
        "Explore Classification Threshold",
        width="stretch",
    ):

        st.switch_page(
            "pages/07_Threshold_Explorer.py"
        )


# ============================================================
# DATASET OVERVIEW
# ============================================================

section_title(
    "database",
    "Dataset Overview",
)


chart1, chart2 = (
    st.columns(
        [
            1.5,
            1,
        ]
    )
)


# ------------------------------------------------------------
# EXAM SCORE DISTRIBUTION
# ------------------------------------------------------------

with chart1:

    if (
        "exam_score"
        in df.columns
    ):

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
            height=320,

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
# PASS / FAIL DISTRIBUTION
# ------------------------------------------------------------

with chart2:

    if (
        "pass_status"
        in df.columns
    ):

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


        fig_class = px.pie(
            class_df,
            names="Status",
            values="Students",
            hole=0.62,
        )


        fig_class.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )


        fig_class.update_layout(
            template=plotly_template,

            height=245,

            margin=dict(
                l=10,
                r=10,
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


        pass_col, fail_col = (
            st.columns(2)
        )


        with pass_col:

            st.html(
                f"""
                <div class="class-summary">

                    <div class="class-summary-label">
                        PASS
                    </div>

                    <div class="class-summary-value">
                        {pass_count:,}
                    </div>

                    <div class="class-summary-note">
                        {pass_count / len(df) * 100:.1f}% of students
                    </div>

                </div>
                """
            )


        with fail_col:

            st.html(
                f"""
                <div class="class-summary">

                    <div class="class-summary-label">
                        FAIL
                    </div>

                    <div class="class-summary-value">
                        {fail_count:,}
                    </div>

                    <div class="class-summary-note">
                        {fail_count / len(df) * 100:.1f}% of students
                    </div>

                </div>
                """
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


available_features = [
    feature
    for feature
    in selected_features
    if feature in df.columns
]


feature_labels = {
    "previous_exam_score":
        "Previous Exam Score",

    "previous_gpa":
        "Previous GPA",

    "attendance_percentage":
        "Attendance",

    "assignment_completion_rate":
        "Assignment Completion",

    "study_hours_per_day":
        "Study Hours",

    "practice_tests_completed":
        "Practice Tests",
}


if (
    "exam_score"
    in df.columns
    and available_features
):

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


    correlation_data[
        "Direction"
    ] = np.where(
        correlation_data[
            "Correlation"
        ] >= 0,
        "Positive",
        "Negative",
    )


    section_title(
        "chart",
        "Selected Features vs. Exam Score",
    )


    st.caption(
        "Pearson correlation between each selected predictor "
        "and the final exam score."
    )


    positive_color = (
        "#5FA879"
    )

    negative_color = (
        "#C86B6B"
    )


    correlation_colors = [
        (
            positive_color
            if value >= 0
            else negative_color
        )
        for value
        in correlation_data[
            "Correlation"
        ]
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
        height=320,

        margin=dict(
            l=20,
            r=60,
            t=10,
            b=20,
        ),

        xaxis_title=(
            "Correlation"
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
        "Green indicates a positive relationship with exam score; "
        "red indicates a negative relationship. Correlation does not imply causation."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

section_title(
    "brain",
    "Model Performance",
)


model1, model2 = (
    st.columns(2)
)


# ------------------------------------------------------------
# LINEAR REGRESSION
# ------------------------------------------------------------

with model1:

    with st.container(
        border=True
    ):

        icon_card(
            "trending",
            "Linear Regression",
            "Predicts the student's numerical exam score.",
        )


        metric1, metric2, metric3 = (
            st.columns(3)
        )


        metric1.metric(
            "MAE",
            (
                f"{model_metrics['linear_mae']:.2f}"
            ),
        )


        metric2.metric(
            "RMSE",
            (
                f"{model_metrics['linear_rmse']:.2f}"
            ),
        )


        metric3.metric(
            "R²",
            (
                f"{model_metrics['linear_r2']:.3f}"
            ),
        )


# ------------------------------------------------------------
# LOGISTIC REGRESSION
# ------------------------------------------------------------

with model2:

    with st.container(
        border=True
    ):

        icon_card(
            "target",
            "Balanced Logistic Regression",
            "Predicts Pass / Fail and pass probability.",
        )


        metric1, metric2, metric3 = (
            st.columns(3)
        )


        metric1.metric(
            "Accuracy",
            (
                f"{model_metrics['logistic_accuracy'] * 100:.1f}%"
            ),
        )


        metric2.metric(
            "Balanced Acc.",
            (
                f"{model_metrics['logistic_balanced_accuracy'] * 100:.1f}%"
            ),
        )


        metric3.metric(
            "F1",
            (
                f"{model_metrics['logistic_f1'] * 100:.1f}%"
            ),
        )


st.caption(
    "Dashboard metrics are calculated directly from the saved models "
    "using the same held-out evaluation splits used throughout the application."
)


# ============================================================
# HOW PREDICTION WORKS
# ============================================================

section_title(
    "sparkles",
    "How Prediction Works",
)


flow1, arrow1, flow2, arrow2, flow3 = (
    st.columns(
        [
            2.2,
            0.4,
            2.2,
            0.4,
            2.2,
        ]
    )
)


with flow1:

    icon_card(
        "sliders",
        "6 Student Inputs",
        "Previous performance, GPA, attendance, assignments, "
        "study hours, and practice tests.",
    )


with arrow1:

    st.html(
        f"""
        <div style="
            text-align:center;
            font-size:30px;
            padding-top:42px;
            color:{chart_muted};
        ">
            →
        </div>
        """
    )


with flow2:

    icon_card(
        "brain",
        "Trained Models",
        "Linear Regression and Balanced Logistic Regression.",
    )


with arrow2:

    st.html(
        f"""
        <div style="
            text-align:center;
            font-size:30px;
            padding-top:42px;
            color:{chart_muted};
        ">
            →
        </div>
        """
    )


with flow3:

    icon_card(
        "graduation",
        "Prediction",
        "Exam Score, Pass / Fail status, and Pass Probability.",
    )


# ============================================================
# INTERACTIVE ML LEARNING
# ============================================================

section_title(
    "sparkles",
    "Interactive ML Learning",
)


learn1, learn2, learn3 = (
    st.columns(3)
)


with learn1:

    icon_card(
        "scale",
        "Threshold Explorer",
        "Adjust the classification threshold and observe "
        "changes in precision, recall, false positives, "
        "and false negatives.",
    )


with learn2:

    icon_card(
        "chart",
        "Model Transparency",
        "Explore class imbalance, evaluation metrics, "
        "feature influence, and prediction uncertainty.",
    )


with learn3:

    icon_card(
        "sliders",
        "Scenario Simulation",
        "Compare alternative student profiles using model estimates "
        "without presenting the changes as causal guarantees.",
    )


# ============================================================
# FINAL CTA
# ============================================================

section_title(
    "sparkles",
    "Try the Prediction System",
)


cta1, cta2, cta3 = (
    st.columns(
        [
            1,
            2,
            1,
        ]
    )
)


with cta2:

    if st.button(
        "Try Student Performance Prediction",
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