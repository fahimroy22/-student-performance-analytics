# ============================================================
# LINEAR REGRESSION PAGE
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from components.styles import (
    apply_global_styles,
    get_plotly_template,
    get_theme_colors,
)

from components.footer import show_footer

from components.icons import (
    page_title,
    section_title,
    icon_card
)

from core.model_loader import load_linear_model


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Linear Regression",
    page_icon="📈",
    layout="wide"
)

apply_global_styles()


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

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
surface = theme_colors["surface"]
chart_text = theme_colors["text"]
chart_muted = theme_colors["muted"]
chart_border = theme_colors["border"]
accent = theme_colors["accent"]


# ------------------------------------------------------------
# COMMON PLOTLY STYLE
# ------------------------------------------------------------

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
        zerolinecolor=chart_border
    )

    fig.update_yaxes(
        color=chart_text,
        gridcolor=chart_border,
        zerolinecolor=chart_border
    )

    return fig


# ------------------------------------------------------------
# MODEL VARIABLES
# ------------------------------------------------------------

features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed"
]

target = "exam_score"


# ------------------------------------------------------------
# PREPARE TEST DATA AND EVALUATE MODEL
# ------------------------------------------------------------

@st.cache_data
def prepare_regression_results():

    X = df[features].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = load_linear_model()

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results_df = pd.DataFrame(
        {
            "Actual Score":
                y_test.values,

            "Predicted Score":
                predictions
        }
    )

    results_df["Residual"] = (
        results_df["Actual Score"]
        - results_df["Predicted Score"]
    )

    return (
        results_df,
        mae,
        rmse,
        r2,
        len(X_train),
        len(X_test)
    )


results_df, mae, rmse, r2, train_n, test_n = (
    prepare_regression_results()
)


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

page_title(
    "trending",
    "Linear Regression",
    "Predict numerical exam scores using six pre-exam "
    "academic and behavioral variables."
)


# ------------------------------------------------------------
# MODEL OVERVIEW
# ------------------------------------------------------------

section_title(
    "database",
    "Model Overview"
)


overview1, overview2, overview3, overview4 = (
    st.columns(4)
)


overview1.metric(
    "Training Records",
    f"{train_n:,}"
)

overview2.metric(
    "Testing Records",
    f"{test_n:,}"
)

overview3.metric(
    "Input Features",
    len(features)
)

overview4.metric(
    "Target",
    "Exam Score"
)


# ------------------------------------------------------------
# MODEL PERFORMANCE
# ------------------------------------------------------------

section_title(
    "chart",
    "Model Performance"
)


m1, m2, m3 = st.columns(3)


with m1:

    with st.container(
        border=True
    ):

        st.metric(
            "MAE",
            f"{mae:.2f}"
        )

        st.caption(
            "Average absolute prediction error."
        )


with m2:

    with st.container(
        border=True
    ):

        st.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        st.caption(
            "Gives more weight to larger errors."
        )


with m3:

    with st.container(
        border=True
    ):

        st.metric(
            "R²",
            f"{r2:.3f}"
        )

        st.caption(
            "Variation explained by the model."
        )


st.info(
    f"Average prediction error is about {mae:.2f} points, "
    f"while the model explains approximately "
    f"{r2 * 100:.1f}% of exam-score variation."
)


# ------------------------------------------------------------
# ACTUAL VS PREDICTED
# ------------------------------------------------------------

section_title(
    "target",
    "Actual vs Predicted Exam Scores"
)


plot_results = results_df.sample(
    min(
        15000,
        len(results_df)
    ),
    random_state=42
)


fig_actual = px.scatter(
    plot_results,
    x="Actual Score",
    y="Predicted Score",
    opacity=0.30
)


fig_actual.update_traces(
    marker=dict(
        color=accent
    )
)


fig_actual.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=100,
    y1=100,

    line=dict(
        color=chart_muted,
        dash="dash",
        width=2
    )
)


fig_actual.update_layout(
    height=430,

    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20
    ),

    xaxis_title="Actual Exam Score",
    yaxis_title="Predicted Exam Score"
)


fig_actual.update_xaxes(
    range=[0, 100]
)


fig_actual.update_yaxes(
    range=[0, 100]
)


style_chart(
    fig_actual
)


st.plotly_chart(
    fig_actual,
    width="stretch",
    theme=None,

    config={
        "displayModeBar": False
    }
)


st.caption(
    "Points closer to the dashed diagonal line represent "
    "more accurate predictions."
)


# ------------------------------------------------------------
# RESIDUAL ANALYSIS
# ------------------------------------------------------------

section_title(
    "chart",
    "Residual Analysis"
)


st.caption(
    "Residual = Actual Score − Predicted Score"
)


residual_col1, residual_col2 = (
    st.columns(2)
)


# ------------------------------------------------------------
# RESIDUALS VS PREDICTED
# ------------------------------------------------------------

with residual_col1:

    fig_residual = px.scatter(
        plot_results,
        x="Predicted Score",
        y="Residual",
        opacity=0.30
    )


    fig_residual.update_traces(
        marker=dict(
            color=accent
        )
    )


    fig_residual.add_hline(
        y=0,
        line_dash="dash",
        line_color=chart_muted
    )


    fig_residual.update_layout(
        height=350,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20
        ),

        xaxis_title="Predicted Exam Score",
        yaxis_title="Residual"
    )


    style_chart(
        fig_residual
    )


    st.plotly_chart(
        fig_residual,
        width="stretch",
        theme=None,

        config={
            "displayModeBar": False
        }
    )


# ------------------------------------------------------------
# RESIDUAL DISTRIBUTION
# ------------------------------------------------------------

with residual_col2:

    fig_residual_hist = px.histogram(
        results_df,
        x="Residual",
        nbins=50
    )


    fig_residual_hist.update_traces(
        marker_color=accent
    )


    fig_residual_hist.update_layout(
        height=350,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20
        ),

        xaxis_title="Residual",
        yaxis_title="Students",

        showlegend=False
    )


    style_chart(
        fig_residual_hist
    )


    st.plotly_chart(
        fig_residual_hist,
        width="stretch",
        theme=None,

        config={
            "displayModeBar": False
        }
    )


st.info(
    "Residuals centered around zero suggest that the model "
    "does not consistently overpredict or underpredict."
)


# ------------------------------------------------------------
# FEATURE INFLUENCE
# ------------------------------------------------------------

section_title(
    "sliders",
    "Feature Influence"
)


linear_model = load_linear_model()


coefficients = (
    linear_model
    .named_steps["model"]
    .coef_
)


coefficient_df = pd.DataFrame(
    {
        "Feature": [
            "Previous Exam Score",
            "Previous GPA",
            "Attendance",
            "Assignment Completion",
            "Study Hours",
            "Practice Tests"
        ],

        "Coefficient":
            coefficients
    }
)


coefficient_df["Absolute Influence"] = (
    coefficient_df["Coefficient"]
    .abs()
)


coefficient_df = (
    coefficient_df
    .sort_values(
        "Absolute Influence",
        ascending=True
    )
)


fig_coeff = px.bar(
    coefficient_df,
    x="Coefficient",
    y="Feature",
    orientation="h",
    text="Coefficient"
)


fig_coeff.update_traces(
    marker_color=accent,

    texttemplate="%{text:.2f}",
    textposition="outside"
)


fig_coeff.update_layout(
    height=340,

    margin=dict(
        l=20,
        r=60,
        t=10,
        b=20
    ),

    xaxis_title="Standardized Coefficient",
    yaxis_title="",

    showlegend=False
)


style_chart(
    fig_coeff
)


st.plotly_chart(
    fig_coeff,
    width="stretch",
    theme=None,

    config={
        "displayModeBar": False
    }
)


st.caption(
    "Because the predictors are standardized inside the pipeline, "
    "their coefficients are more directly comparable. Larger "
    "absolute values indicate stronger influence on the predicted score."
)


# ------------------------------------------------------------
# MULTICOLLINEARITY
# ------------------------------------------------------------

section_title(
    "scale",
    "Important Modeling Consideration"
)


warning1, warning2 = st.columns(
    [1, 1]
)


with warning1:

    icon_card(
        "scale",
        "High Feature Correlation",
        "Previous Exam Score and Previous GPA are highly "
        "correlated at approximately 0.95."
    )


with warning2:

    icon_card(
        "check",
        "Prediction Still Valid",
        "The model can still predict successfully, but individual "
        "coefficient values should be interpreted cautiously."
    )


st.warning(
    "This overlap is called multicollinearity. It does not prevent "
    "prediction, but it makes it harder to separate the unique effect "
    "of Previous Exam Score from Previous GPA."
)


# ------------------------------------------------------------
# MODEL INTERPRETATION
# ------------------------------------------------------------

section_title(
    "brain",
    "Model Interpretation"
)


interpret1, interpret2, interpret3 = (
    st.columns(3)
)


with interpret1:

    icon_card(
        "target",
        "Average Error",
        f"Predictions differ from actual exam scores by about "
        f"{mae:.2f} points on average."
    )


with interpret2:

    icon_card(
        "chart",
        "Explained Variation",
        f"The model explains approximately "
        f"{r2 * 100:.1f}% of exam-score variation."
    )


with interpret3:

    icon_card(
        "brain",
        "Overall Performance",
        "The model provides moderate predictive performance "
        "using only six pre-exam variables."
    )


# ------------------------------------------------------------
# CONCLUSION
# ------------------------------------------------------------

section_title(
    "check",
    "Conclusion"
)


st.success(
    "Linear Regression provides a useful numerical estimate of "
    "exam performance. Previous academic performance and study-related "
    "behavior contribute meaningful predictive information, while "
    "substantial variation remains unexplained."
)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

show_footer()