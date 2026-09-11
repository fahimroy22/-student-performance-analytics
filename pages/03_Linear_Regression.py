# ============================================================
# LINEAR REGRESSION PAGE
# Visual-first final version
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split

from components.styles import (
    apply_global_styles,
    get_plotly_template,
    get_theme_colors,
)

from components.footer import show_footer

from components.icons import (
    page_title,
    section_title,
    icon_card,
)

from core.model_loader import load_linear_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Linear Regression",
    page_icon="📈",
    layout="wide",
)

apply_global_styles()


# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
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

    .mini-explain-card {{
        background:{surface};
        border:1px solid {chart_border};
        border-radius:12px;
        padding:14px 15px;
        min-height:115px;
    }}

    .mini-explain-label {{
        color:{accent};
        font-size:11px;
        font-weight:700;
        margin-bottom:6px;
    }}

    .mini-explain-title {{
        color:{chart_text};
        font-size:15px;
        font-weight:700;
        margin-bottom:6px;
    }}

    .mini-explain-text {{
        color:{chart_muted};
        font-size:11px;
        line-height:1.45;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# COMMON CHART STYLE
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
# MODEL VARIABLES
# ============================================================

features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed",
]

target = "exam_score"


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


# ============================================================
# MODEL RESOURCE
# ============================================================

@st.cache_resource
def get_linear_model():

    return load_linear_model()


# ============================================================
# PREPARE TEST DATA + METRICS
# ============================================================

@st.cache_data
def prepare_regression_results():

    X = df[
        features
    ].copy()

    y = df[
        target
    ].copy()


    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )


    model = get_linear_model()


    predictions = model.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        predictions,
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )


    r2 = r2_score(
        y_test,
        predictions,
    )


    results_df = pd.DataFrame(
        {
            "Actual Score":
                y_test.values,

            "Predicted Score":
                predictions,
        }
    )


    results_df[
        "Residual"
    ] = (
        results_df[
            "Actual Score"
        ]
        - results_df[
            "Predicted Score"
        ]
    )


    results_df[
        "Absolute Error"
    ] = (
        results_df[
            "Residual"
        ]
        .abs()
    )


    return (
        results_df,
        X_train,
        X_test,
        mae,
        rmse,
        r2,
    )


(
    results_df,
    X_train,
    X_test,
    mae,
    rmse,
    r2,
) = prepare_regression_results()


train_n = len(
    X_train
)

test_n = len(
    X_test
)


# ============================================================
# EXTRA REAL ERROR METRICS
# ============================================================

mean_residual = (
    results_df[
        "Residual"
    ]
    .mean()
)

median_residual = (
    results_df[
        "Residual"
    ]
    .median()
)

within_5 = (
    (
        results_df[
            "Absolute Error"
        ] <= 5
    )
    .mean()
    * 100
)

within_10 = (
    (
        results_df[
            "Absolute Error"
        ] <= 10
    )
    .mean()
    * 100
)

within_15 = (
    (
        results_df[
            "Absolute Error"
        ] <= 15
    )
    .mean()
    * 100
)


# ============================================================
# ERROR BANDS
# ============================================================

error_bins = [
    0,
    5,
    10,
    15,
    20,
    np.inf,
]

error_labels = [
    "0–5",
    "5–10",
    "10–15",
    "15–20",
    "20+",
]


results_df[
    "Error Band"
] = pd.cut(
    results_df[
        "Absolute Error"
    ],
    bins=error_bins,
    labels=error_labels,
    include_lowest=True,
    right=True,
)


error_band_df = (
    results_df[
        "Error Band"
    ]
    .value_counts()
    .reindex(
        error_labels,
        fill_value=0,
    )
    .reset_index()
)


error_band_df.columns = [
    "Error Band",
    "Students",
]


# ============================================================
# HEADER
# ============================================================

page_title(
    "trending",
    "Linear Regression",
    "Predict numerical exam scores using six pre-exam "
    "academic and behavioral variables.",
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

section_title(
    "database",
    "Model Overview",
)


overview1, overview2, overview3, overview4 = (
    st.columns(4)
)


overview1.metric(
    "Training Records",
    f"{train_n:,}",
)

overview2.metric(
    "Testing Records",
    f"{test_n:,}",
)

overview3.metric(
    "Input Features",
    len(features),
)

overview4.metric(
    "Target",
    "Exam Score",
)


# ============================================================
# PERFORMANCE DASHBOARD
# ============================================================

section_title(
    "chart",
    "Model Performance",
)


m1, m2, m3 = st.columns(3)


m1.metric(
    "MAE",
    f"{mae:.2f}",
)

m2.metric(
    "RMSE",
    f"{rmse:.2f}",
)

m3.metric(
    "R²",
    f"{r2:.3f}",
)


performance_left, performance_right = (
    st.columns(
        [1.15, 1]
    )
)


# ------------------------------------------------------------
# ERROR METRICS
# ------------------------------------------------------------

with performance_left:

    error_metric_df = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
            ],

            "Value": [
                mae,
                rmse,
            ],
        }
    )


    fig_error_metrics = px.bar(
        error_metric_df,
        x="Metric",
        y="Value",
        text="Value",
    )


    fig_error_metrics.update_traces(
        marker_color=accent,
        texttemplate="%{text:.2f}",
        textposition="outside",
    )


    fig_error_metrics.update_layout(
        height=300,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=25,
        ),

        xaxis_title="",

        yaxis_title=(
            "Exam-Score Points"
        ),

        showlegend=False,
    )


    style_chart(
        fig_error_metrics
    )


    st.plotly_chart(
        fig_error_metrics,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ------------------------------------------------------------
# EXPLAINED VARIATION
# ------------------------------------------------------------

with performance_right:

    explained_percent = (
        max(
            min(
                r2 * 100,
                100,
            ),
            0,
        )
    )

    unexplained_percent = (
        100
        - explained_percent
    )


    variation_df = pd.DataFrame(
        {
            "Component": [
                "Explained",
                "Unexplained",
            ],

            "Percentage": [
                explained_percent,
                unexplained_percent,
            ],
        }
    )


    fig_variation = px.pie(
        variation_df,
        names="Component",
        values="Percentage",
        hole=0.66,
    )


    fig_variation.update_traces(
        textinfo="label+percent",
        textposition="inside",
    )


    fig_variation.update_layout(
        template=plotly_template,

        height=300,

        margin=dict(
            l=10,
            r=10,
            t=15,
            b=15,
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
        fig_variation,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


st.caption(
    f"Average absolute error = {mae:.2f} score points · "
    f"R² = {r2:.3f}, so the model explains about "
    f"{r2 * 100:.1f}% of test-set exam-score variation."
)


# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

section_title(
    "target",
    "Actual vs Predicted Exam Scores",
)


plot_results = results_df.sample(
    min(
        15000,
        len(results_df),
    ),
    random_state=42,
)


actual_left, actual_right = (
    st.columns(
        [1.5, 1]
    )
)


# ------------------------------------------------------------
# SCATTER
# ------------------------------------------------------------

with actual_left:

    fig_actual = px.scatter(
        plot_results,
        x="Actual Score",
        y="Predicted Score",
        opacity=0.30,
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
            width=2,
        ),
    )


    fig_actual.update_layout(
        height=410,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=25,
        ),

        xaxis_title=(
            "Actual Exam Score"
        ),

        yaxis_title=(
            "Predicted Exam Score"
        ),
    )


    fig_actual.update_xaxes(
        range=[
            0,
            100,
        ]
    )


    fig_actual.update_yaxes(
        range=[
            0,
            100,
        ]
    )


    style_chart(
        fig_actual
    )


    st.plotly_chart(
        fig_actual,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ------------------------------------------------------------
# VISUAL INTERPRETATION
# ------------------------------------------------------------

with actual_right:

    st.html(
        f"""
        <div class="mini-explain-card">

            <div class="mini-explain-label">
                ON THE DIAGONAL
            </div>

            <div class="mini-explain-title">
                Accurate Prediction
            </div>

            <div class="mini-explain-text">
                Points close to the dashed line have predicted
                scores close to the actual exam score.
            </div>

        </div>
        """
    )

    st.write("")

    st.html(
        f"""
        <div class="mini-explain-card">

            <div class="mini-explain-label">
                ABOVE THE LINE
            </div>

            <div class="mini-explain-title">
                Overprediction
            </div>

            <div class="mini-explain-text">
                The predicted score is higher than the student's
                actual exam score.
            </div>

        </div>
        """
    )

    st.write("")

    st.html(
        f"""
        <div class="mini-explain-card">

            <div class="mini-explain-label">
                BELOW THE LINE
            </div>

            <div class="mini-explain-title">
                Underprediction
            </div>

            <div class="mini-explain-text">
                The predicted score is lower than the student's
                actual exam score.
            </div>

        </div>
        """
    )


# ============================================================
# SCORE DISTRIBUTIONS
# ============================================================

section_title(
    "chart",
    "Actual vs Predicted Score Distribution",
)


distribution_df = pd.DataFrame(
    {
        "Score": pd.concat(
            [
                results_df[
                    "Actual Score"
                ],

                results_df[
                    "Predicted Score"
                ],
            ],
            ignore_index=True,
        ),

        "Type": (
            ["Actual"]
            * len(
                results_df
            )
            +
            ["Predicted"]
            * len(
                results_df
            )
        ),
    }
)


fig_distribution = px.histogram(
    distribution_df,
    x="Score",
    color="Type",
    nbins=45,
    barmode="overlay",
    opacity=0.55,
)


fig_distribution.update_layout(
    height=340,

    margin=dict(
        l=20,
        r=20,
        t=15,
        b=25,
    ),

    xaxis_title=(
        "Exam Score"
    ),

    yaxis_title=(
        "Students"
    ),

    legend_title="",
)


style_chart(
    fig_distribution
)


st.plotly_chart(
    fig_distribution,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)


st.caption(
    "This comparison shows whether the predicted-score distribution "
    "resembles the overall distribution of actual exam scores."
)


# ============================================================
# RESIDUAL ANALYSIS
# ============================================================

section_title(
    "chart",
    "Residual Analysis",
)


st.caption(
    "Residual = Actual Score − Predicted Score"
)


residual_col1, residual_col2 = (
    st.columns(2)
)


with residual_col1:

    fig_residual = px.scatter(
        plot_results,
        x="Predicted Score",
        y="Residual",
        opacity=0.30,
    )


    fig_residual.update_traces(
        marker=dict(
            color=accent
        )
    )


    fig_residual.add_hline(
        y=0,
        line_dash="dash",
        line_color=chart_muted,
    )


    fig_residual.update_layout(
        height=340,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),

        xaxis_title=(
            "Predicted Exam Score"
        ),

        yaxis_title=(
            "Residual"
        ),
    )


    style_chart(
        fig_residual
    )


    st.plotly_chart(
        fig_residual,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


with residual_col2:

    fig_residual_hist = px.histogram(
        results_df,
        x="Residual",
        nbins=50,
    )


    fig_residual_hist.update_traces(
        marker_color=accent
    )


    fig_residual_hist.add_vline(
        x=0,
        line_dash="dash",
        line_color=chart_muted,
    )


    fig_residual_hist.update_layout(
        height=340,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20,
        ),

        xaxis_title="Residual",

        yaxis_title="Students",

        showlegend=False,
    )


    style_chart(
        fig_residual_hist
    )


    st.plotly_chart(
        fig_residual_hist,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ============================================================
# ERROR INTERPRETATION METRICS
# ============================================================

section_title(
    "target",
    "How Close Are the Predictions?",
)


e1, e2, e3, e4, e5 = (
    st.columns(5)
)


e1.metric(
    "Mean Residual",
    f"{mean_residual:.2f}",
)

e2.metric(
    "Median Residual",
    f"{median_residual:.2f}",
)

e3.metric(
    "Within ±5",
    f"{within_5:.1f}%",
)

e4.metric(
    "Within ±10",
    f"{within_10:.1f}%",
)

e5.metric(
    "Within ±15",
    f"{within_15:.1f}%",
)


# ============================================================
# ERROR MAGNITUDE
# ============================================================

section_title(
    "chart",
    "Prediction Error Magnitude",
)


fig_error_band = px.bar(
    error_band_df,
    x="Error Band",
    y="Students",
    text="Students",
)


fig_error_band.update_traces(
    marker_color=accent,
    textposition="outside",
)


fig_error_band.update_layout(
    height=320,

    margin=dict(
        l=20,
        r=20,
        t=15,
        b=25,
    ),

    xaxis_title=(
        "Absolute Error in Exam-Score Points"
    ),

    yaxis_title=(
        "Students"
    ),

    showlegend=False,
)


style_chart(
    fig_error_band
)


st.plotly_chart(
    fig_error_band,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)


st.caption(
    "This groups test predictions by the size of their absolute error."
)


# ============================================================
# FEATURE INFLUENCE
# ============================================================

section_title(
    "sliders",
    "Feature Influence",
)


linear_model = get_linear_model()


coefficients = (
    linear_model
    .named_steps[
        "model"
    ]
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
            "Practice Tests",
        ],

        "Coefficient":
            coefficients,
    }
)


coefficient_df[
    "Direction"
] = np.where(
    coefficient_df[
        "Coefficient"
    ] >= 0,
    "Positive",
    "Negative",
)


coefficient_df[
    "Absolute Influence"
] = (
    coefficient_df[
        "Coefficient"
    ]
    .abs()
)


coefficient_df = (
    coefficient_df
    .sort_values(
        "Absolute Influence",
        ascending=True,
    )
)


positive_color = "#5FA879"
negative_color = "#C86B6B"


bar_colors = [
    (
        positive_color
        if value >= 0
        else negative_color
    )
    for value
    in coefficient_df[
        "Coefficient"
    ]
]


fig_coeff = go.Figure()


fig_coeff.add_trace(
    go.Bar(
        x=coefficient_df[
            "Coefficient"
        ],

        y=coefficient_df[
            "Feature"
        ],

        orientation="h",

        marker_color=bar_colors,

        text=coefficient_df[
            "Coefficient"
        ],

        texttemplate=(
            "%{text:.2f}"
        ),

        textposition="outside",

        hovertemplate=(
            "%{y}"
            "<br>"
            "Standardized coefficient: %{x:.3f}"
            "<extra></extra>"
        ),
    )
)


fig_coeff.add_vline(
    x=0,
    line_dash="dash",
    line_color=chart_muted,
)


fig_coeff.update_layout(
    height=340,

    margin=dict(
        l=20,
        r=65,
        t=10,
        b=20,
    ),

    xaxis_title=(
        "Standardized Coefficient"
    ),

    yaxis_title="",

    showlegend=False,
)


style_chart(
    fig_coeff
)


st.plotly_chart(
    fig_coeff,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)


st.caption(
    "Green coefficients increase the predicted exam score; red coefficients "
    "decrease it. Because predictors are standardized, coefficient magnitudes "
    "are more directly comparable."
)


# ============================================================
# FEATURE CORRELATION
# ============================================================

section_title(
    "scale",
    "Feature Correlation",
)


correlation_data = (
    df[
        features
    ]
    .corr()
)


correlation_display_names = [
    feature_labels[
        feature
    ]
    for feature
    in features
]


fig_corr = go.Figure(
    data=go.Heatmap(

        z=correlation_data.values,

        x=correlation_display_names,

        y=correlation_display_names,

        zmin=-1,
        zmax=1,

        colorscale="RdBu",

        reversescale=True,

        text=np.round(
            correlation_data.values,
            2,
        ),

        texttemplate="%{text:.2f}",

        hovertemplate=(
            "%{y}"
            " vs "
            "%{x}"
            "<br>"
            "Correlation: %{z:.2f}"
            "<extra></extra>"
        ),

        colorbar=dict(
            title="Correlation"
        ),
    )
)


fig_corr.update_layout(
    height=470,

    margin=dict(
        l=20,
        r=20,
        t=15,
        b=60,
    ),
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


# ============================================================
# FOCUSED MULTICOLLINEARITY VISUAL
# ============================================================

corr_pair = (
    df[
        [
            "previous_exam_score",
            "previous_gpa",
        ]
    ]
    .dropna()
)


pair_corr = (
    corr_pair[
        "previous_exam_score"
    ]
    .corr(
        corr_pair[
            "previous_gpa"
        ]
    )
)


multi_left, multi_right = (
    st.columns(
        [1.5, 1]
    )
)


with multi_left:

    sampled_corr = corr_pair.sample(
        min(
            10000,
            len(
                corr_pair
            ),
        ),
        random_state=42,
    )


    fig_pair = px.scatter(
        sampled_corr,
        x="previous_exam_score",
        y="previous_gpa",
        opacity=0.25,
    )


    fig_pair.update_traces(
        marker_color=accent
    )


    fig_pair.update_layout(
        height=350,

        margin=dict(
            l=20,
            r=20,
            t=15,
            b=25,
        ),

        xaxis_title=(
            "Previous Exam Score"
        ),

        yaxis_title=(
            "Previous GPA"
        ),
    )


    style_chart(
        fig_pair
    )


    st.plotly_chart(
        fig_pair,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


with multi_right:

    icon_card(
        "scale",
        "High Feature Correlation",
        (
            "Previous Exam Score and Previous GPA have a strong "
            f"correlation of approximately {pair_corr:.2f}."
        ),
    )

    st.write("")

    icon_card(
        "check",
        "Why It Matters",
        "The model can still make predictions, but these two predictors "
        "contain overlapping information, so their individual coefficients "
        "should be interpreted cautiously.",
    )


# ============================================================
# FINAL TAKEAWAY
# ============================================================

section_title(
    "brain",
    "Linear Regression Takeaway",
)


take1, take2, take3 = (
    st.columns(3)
)


with take1:

    icon_card(
        "target",
        "Typical Error",
        (
            f"Average absolute prediction error is "
            f"{mae:.2f} exam-score points."
        ),
    )


with take2:

    icon_card(
        "chart",
        "Explained Variation",
        (
            f"The six predictors explain approximately "
            f"{r2 * 100:.1f}% of exam-score variation."
        ),
    )


with take3:

    icon_card(
        "scale",
        "Key Limitation",
        (
            "The model provides a useful estimate, but substantial score "
            "variation remains unexplained and some predictors are correlated."
        ),
    )


# ============================================================
# CONCLUSION
# ============================================================

st.success(
    "Linear Regression provides a useful numerical estimate of exam "
    "performance from six pre-exam variables. The test-set results show "
    "moderate predictive performance rather than exact score prediction."
)


# ============================================================
# FOOTER
# ============================================================

show_footer()