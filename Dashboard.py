# ============================================================
# STUDENT PERFORMANCE ANALYTICS - DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

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


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="🎓",
    layout="wide",
)

apply_global_styles()


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "student_performance.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# ------------------------------------------------------------
# THEME SETTINGS
# ------------------------------------------------------------

plotly_template = get_plotly_template()

theme_colors = get_theme_colors()

chart_background = theme_colors["background"]
chart_text = theme_colors["text"]
chart_muted = theme_colors["muted"]
chart_border = theme_colors["border"]
accent = theme_colors["accent"]


# ------------------------------------------------------------
# COMMON PLOTLY THEME
# ------------------------------------------------------------

def style_chart(fig):

    fig.update_layout(
        template=plotly_template,

        paper_bgcolor=chart_background,
        plot_bgcolor=chart_background,

        font=dict(
            color=chart_text
        ),

        xaxis=dict(
            color=chart_text,
            gridcolor=chart_border,
            zerolinecolor=chart_border,
        ),

        yaxis=dict(
            color=chart_text,
            gridcolor=chart_border,
            zerolinecolor=chart_border,
        ),
    )

    return fig


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

page_title(
    "dashboard",
    "Student Performance Analytics",
    "Explore the dataset, machine-learning workflow, "
    "model performance, and student predictions.",
)


# ------------------------------------------------------------
# DATASET SNAPSHOT
# ------------------------------------------------------------

records = len(df)
columns = df.shape[1]

missing_values = int(
    df.isnull().sum().sum()
)

duplicates = int(
    df.duplicated().sum()
)


col1, col2, col3, col4 = st.columns(4)


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


# ------------------------------------------------------------
# QUICK NAVIGATION
# ------------------------------------------------------------

section_title(
    "dashboard",
    "Explore the Project",
)


nav1, nav2, nav3 = st.columns(3)


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


nav4, nav5, nav6 = st.columns(3)


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


# ------------------------------------------------------------
# DATASET OVERVIEW
# ------------------------------------------------------------

section_title(
    "database",
    "Dataset Overview",
)


chart1, chart2 = st.columns(
    [1.5, 1]
)


# ------------------------------------------------------------
# EXAM SCORE DISTRIBUTION
# ------------------------------------------------------------

with chart1:

    if "exam_score" in df.columns:

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

            xaxis_title="Exam Score",
            yaxis_title="Students",
            showlegend=False,
        )


        style_chart(
            fig_score
        )


        st.markdown(
            "#### Exam Score Distribution"
        )


        st.plotly_chart(
            fig_score,
            width="stretch",
            theme=None,

            config={
                "displayModeBar": False
            },
        )


# ------------------------------------------------------------
# PASS / FAIL DISTRIBUTION
# ------------------------------------------------------------

with chart2:

    if "pass_status" in df.columns:

        class_counts = (
            df["pass_status"]
            .value_counts()
            .reset_index()
        )


        class_counts.columns = [
            "Status",
            "Students",
        ]


        fig_class = px.pie(
            class_counts,
            names="Status",
            values="Students",
            hole=0.60,
        )


        fig_class.update_traces(
            textposition="inside",
            textinfo="percent+label",
        )


        fig_class.update_layout(
            template=plotly_template,

            height=320,

            margin=dict(
                l=20,
                r=20,
                t=10,
                b=20,
            ),

            showlegend=False,

            paper_bgcolor=chart_background,
            plot_bgcolor=chart_background,

            font=dict(
                color=chart_text
            ),
        )


        st.markdown(
            "#### Pass / Fail Distribution"
        )


        st.plotly_chart(
            fig_class,
            width="stretch",
            theme=None,

            config={
                "displayModeBar": False
            },
        )


# ------------------------------------------------------------
# FEATURE CORRELATION
# ------------------------------------------------------------

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
    for feature in selected_features
    if feature in df.columns
]


if (
    "exam_score" in df.columns
    and available_features
):

    correlation_data = (
        df[
            available_features
            + ["exam_score"]
        ]

        .corr(
            numeric_only=True
        )["exam_score"]

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


    correlation_data["Feature"] = (
        correlation_data["Feature"]
        .map(
            feature_labels
        )
    )


    section_title(
        "chart",
        "Selected Features vs. Exam Score",
    )


    st.caption(
        "Pearson correlation with the final exam score."
    )


    fig_corr = px.bar(
        correlation_data,
        x="Correlation",
        y="Feature",
        orientation="h",
        text="Correlation",
    )


    fig_corr.update_traces(
        marker_color=accent,

        texttemplate="%{text:.2f}",
        textposition="outside",
    )


    fig_corr.update_layout(
        height=320,

        margin=dict(
            l=20,
            r=60,
            t=10,
            b=20,
        ),

        xaxis_title="Correlation",
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
            "displayModeBar": False
        },
    )


# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------

section_title(
    "brain",
    "Model Performance",
)


model1, model2 = st.columns(2)


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
            "9.38",
        )

        metric2.metric(
            "RMSE",
            "11.76",
        )

        metric3.metric(
            "R²",
            "0.444",
        )


# ------------------------------------------------------------
# BALANCED LOGISTIC REGRESSION
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
            "72.5%",
        )

        metric2.metric(
            "Balanced Acc.",
            "73.2%",
        )

        metric3.metric(
            "F1",
            "80.1%",
        )


# ------------------------------------------------------------
# HOW PREDICTION WORKS
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# INPUTS
# ------------------------------------------------------------

with flow1:

    icon_card(
        "sliders",
        "6 Student Inputs",
        "Previous performance, GPA, attendance, assignments, "
        "study hours, and practice tests.",
    )


# ------------------------------------------------------------
# ARROW 1
# ------------------------------------------------------------

with arrow1:

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:30px;
            padding-top:42px;
            color:{chart_muted};
        ">
            →
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# MODELS
# ------------------------------------------------------------

with flow2:

    icon_card(
        "brain",
        "Trained Models",
        "Linear Regression and Balanced Logistic Regression.",
    )


# ------------------------------------------------------------
# ARROW 2
# ------------------------------------------------------------

with arrow2:

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:30px;
            padding-top:42px;
            color:{chart_muted};
        ">
            →
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

with flow3:

    icon_card(
        "graduation",
        "Prediction",
        "Exam Score, Pass / Fail status, and Pass Probability.",
    )


# ------------------------------------------------------------
# FINAL CALL TO ACTION
# ------------------------------------------------------------

section_title(
    "sparkles",
    "Try the Prediction System",
)


cta1, cta2, cta3 = st.columns(
    [1, 2, 1]
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


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

show_footer()