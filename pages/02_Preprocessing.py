# ============================================================
# DATA PREPROCESSING PAGE
# Visual-first final version
# ============================================================

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="⚙️",
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

    .pipeline-flow {{
        display:grid;
        grid-template-columns:
            1fr 0.16fr 1fr 0.16fr 1fr 0.16fr 1fr;
        gap:8px;
        align-items:center;
        margin:8px 0 18px 0;
    }}

    .pipeline-step {{
        background:{surface};
        border:1px solid {chart_border};
        border-radius:12px;
        padding:15px 14px;
        min-height:118px;
    }}

    .pipeline-step-label {{
        color:{accent};
        font-size:11px;
        font-weight:700;
        margin-bottom:6px;
    }}

    .pipeline-step-title {{
        color:{chart_text};
        font-size:15px;
        font-weight:700;
        margin-bottom:6px;
    }}

    .pipeline-step-text {{
        color:{chart_muted};
        font-size:11px;
        line-height:1.45;
    }}

    .pipeline-arrow {{
        color:{chart_muted};
        text-align:center;
        font-size:23px;
    }}

    .preprocess-table-container {{
        overflow-x:auto;
        border:1px solid {chart_border};
        border-radius:10px;
        background:{surface};
        margin-bottom:14px;
    }}

    .preprocess-table {{
        width:100%;
        border-collapse:collapse;
        font-size:13px;
        color:{chart_text};
        background:{surface};
    }}

    .preprocess-table thead th {{
        text-align:left;
        padding:10px 11px;
        font-weight:600;
        color:{chart_muted};
        background:{surface};
        border-bottom:1px solid {chart_border};
        white-space:nowrap;
    }}

    .preprocess-table tbody td {{
        padding:9px 11px;
        color:{chart_text};
        background:{surface};
        border-bottom:1px solid {chart_border};
        white-space:nowrap;
    }}

    .preprocess-table tbody tr:last-child td {{
        border-bottom:none;
    }}

    .feature-grid {{
        display:grid;
        grid-template-columns:repeat(3,1fr);
        gap:9px;
        margin-top:7px;
    }}

    .feature-pill {{
        background:{surface};
        border:1px solid {chart_border};
        border-radius:9px;
        padding:10px 12px;
        color:{chart_text};
        font-size:12px;
        font-weight:600;
        text-align:center;
    }}

    .target-grid {{
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:10px;
        margin-top:7px;
    }}

    .target-card {{
        background:{surface};
        border:1px solid {chart_border};
        border-radius:12px;
        padding:14px 15px;
        min-height:105px;
    }}

    .target-label {{
        color:{chart_muted};
        font-size:11px;
        margin-bottom:7px;
    }}

    .target-value {{
        color:{chart_text};
        font-size:20px;
        font-weight:700;
        margin-bottom:6px;
    }}

    .target-note {{
        color:{chart_muted};
        font-size:11px;
        line-height:1.4;
    }}

    .imputation-row {{
        display:grid;
        grid-template-columns:
            1.25fr 1fr 0.25fr 0.7fr;
        gap:10px;
        align-items:center;
        padding:10px 12px;
        border-bottom:1px solid {chart_border};
    }}

    .imputation-row:last-child {{
        border-bottom:none;
    }}

    .imputation-name {{
        color:{chart_text};
        font-size:12px;
        font-weight:600;
    }}

    .imputation-bar-wrap {{
        height:12px;
        background:{chart_border};
        border-radius:10px;
        overflow:hidden;
    }}

    .imputation-bar {{
        height:100%;
        background:{accent};
        border-radius:10px;
    }}

    .imputation-arrow {{
        color:{chart_muted};
        text-align:center;
        font-weight:700;
    }}

    .imputation-after {{
        color:#5FA879;
        font-weight:700;
        font-size:12px;
        text-align:right;
    }}

    .imputation-container {{
        border:1px solid {chart_border};
        border-radius:12px;
        background:{surface};
        overflow:hidden;
    }}

    @media(max-width:900px) {{

        .pipeline-flow {{
            grid-template-columns:1fr;
        }}

        .pipeline-arrow {{
            transform:rotate(90deg);
        }}

        .feature-grid {{
            grid-template-columns:1fr 1fr;
        }}

        .target-grid {{
            grid-template-columns:1fr;
        }}

        .imputation-row {{
            grid-template-columns:1fr;
        }}

        .imputation-after {{
            text-align:left;
        }}

    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CHART STYLE
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
# TABLE
# ============================================================

def show_theme_table(table_df):

    clean_df = (
        table_df
        .copy()
        .fillna("—")
    )

    table_html = clean_df.to_html(
        index=False,
        escape=True,
        border=0,
        classes="preprocess-table",
    )

    st.html(
        f"""
        <div class="preprocess-table-container">
            {table_html}
        </div>
        """
    )


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

regression_target = "exam_score"
classification_target = "pass_status"

model_columns = (
    features
    + [
        regression_target,
        classification_target,
    ]
)


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

    "exam_score":
        "Exam Score",

    "pass_status":
        "Pass / Fail",
}


def friendly_name(column):

    return feature_labels.get(
        column,
        column.replace(
            "_",
            " ",
        ).title(),
    )


# ============================================================
# ACTUAL PREPROCESSING DATA
# ============================================================

@st.cache_data
def create_preprocessing_data():

    X = df[
        features
    ].copy()

    y = df[
        regression_target
    ]


    X_train, X_test, _, _ = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )


    imputer = SimpleImputer(
        strategy="median"
    )

    X_train_imputed_array = (
        imputer.fit_transform(
            X_train
        )
    )


    X_train_imputed = pd.DataFrame(
        X_train_imputed_array,
        columns=features,
        index=X_train.index,
    )


    scaler = StandardScaler()

    X_train_scaled_array = (
        scaler.fit_transform(
            X_train_imputed
        )
    )


    X_train_scaled = pd.DataFrame(
        X_train_scaled_array,
        columns=features,
        index=X_train.index,
    )


    medians = pd.Series(
        imputer.statistics_,
        index=features,
    )


    return (
        X_train,
        X_test,
        X_train_imputed,
        X_train_scaled,
        medians,
    )


(
    X_train,
    X_test,
    X_train_imputed,
    X_train_scaled,
    training_medians,
) = create_preprocessing_data()


# ============================================================
# HEADER
# ============================================================

page_title(
    "settings",
    "Data Preprocessing",
    "Transform raw student data into model-ready inputs "
    "for regression and classification.",
)


# ============================================================
# WORKFLOW
# ============================================================

section_title(
    "settings",
    "Preprocessing Workflow",
)


workflow_cols = st.columns(6)


workflow = [
    (
        "database",
        "Raw Data",
        "100,000 student records",
    ),
    (
        "filter",
        "Select Features",
        "Six pre-exam predictors",
    ),
    (
        "wrench",
        "Impute Missing",
        "Training-set median",
    ),
    (
        "split",
        "Train / Test",
        "80% / 20%",
    ),
    (
        "sliders",
        "Standardize",
        "Mean 0 · SD 1",
    ),
    (
        "brain",
        "Model Ready",
        "Ready for ML",
    ),
]


for col, (
    icon_name,
    title,
    description,
) in zip(
    workflow_cols,
    workflow,
):

    with col:

        icon_card(
            icon_name,
            title,
            description,
        )


# ============================================================
# DATASET STATUS
# ============================================================

section_title(
    "database",
    "Dataset Status",
)


c1, c2, c3, c4 = (
    st.columns(4)
)


c1.metric(
    "Records",
    f"{len(df):,}",
)

c2.metric(
    "Variables",
    df.shape[1],
)

c3.metric(
    "Missing Cells",
    f"{int(df.isna().sum().sum()):,}",
)

c4.metric(
    "Duplicates",
    f"{int(df.duplicated().sum()):,}",
)


# ============================================================
# VARIABLES USED
# ============================================================

section_title(
    "filter",
    "Variables Used by the Models",
)


feature_col, target_col = (
    st.columns(
        [1.6, 1]
    )
)


with feature_col:

    st.markdown(
        "#### Six Prediction Features"
    )

    st.html(
        """
        <div class="feature-grid">

            <div class="feature-pill">
                Previous Exam Score
            </div>

            <div class="feature-pill">
                Previous GPA
            </div>

            <div class="feature-pill">
                Attendance
            </div>

            <div class="feature-pill">
                Assignment Completion
            </div>

            <div class="feature-pill">
                Study Hours
            </div>

            <div class="feature-pill">
                Practice Tests
            </div>

        </div>
        """
    )


with target_col:

    st.markdown(
        "#### Prediction Targets"
    )

    st.html(
        """
        <div class="target-grid">

            <div class="target-card">

                <div class="target-label">
                    Regression
                </div>

                <div class="target-value">
                    Exam Score
                </div>

                <div class="target-note">
                    Numerical prediction
                </div>

            </div>


            <div class="target-card">

                <div class="target-label">
                    Classification
                </div>

                <div class="target-value">
                    Pass / Fail
                </div>

                <div class="target-note">
                    Class + probability
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# MISSING-VALUE PROFILE
# ============================================================

section_title(
    "wrench",
    "Missing-Value Profile",
)


missing_counts = (
    df[
        features
    ]
    .isna()
    .sum()
)


missing_percentages = (
    missing_counts
    / len(df)
    * 100
)


missing_df = pd.DataFrame(
    {
        "Feature": [
            friendly_name(
                col
            )
            for col in features
        ],

        "Missing": [
            int(
                missing_counts[
                    col
                ]
            )
            for col in features
        ],

        "Missing %": [
            float(
                missing_percentages[
                    col
                ]
            )
            for col in features
        ],
    }
)


missing_left, missing_right = (
    st.columns(
        [1.6, 1]
    )
)


with missing_left:

    fig_missing = go.Figure()


    fig_missing.add_trace(
        go.Bar(
            x=missing_df[
                "Missing"
            ],

            y=missing_df[
                "Feature"
            ],

            orientation="h",

            marker_color=accent,

            text=[
                (
                    f"{count:,} "
                    f"({percent:.1f}%)"
                )
                for count, percent
                in zip(
                    missing_df[
                        "Missing"
                    ],
                    missing_df[
                        "Missing %"
                    ],
                )
            ],

            textposition="outside",
        )
    )


    fig_missing.update_layout(
        height=300,

        margin=dict(
            l=20,
            r=110,
            t=15,
            b=30,
        ),

        xaxis_title=(
            "Missing Records"
        ),

        yaxis_title="",

        showlegend=False,
    )


    style_chart(
        fig_missing
    )


    st.plotly_chart(
        fig_missing,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


with missing_right:

    total_feature_cells = (
        len(df)
        * len(features)
    )

    selected_missing_total = int(
        missing_counts.sum()
    )

    complete_feature_cells = (
        total_feature_cells
        - selected_missing_total
    )


    completeness_df = pd.DataFrame(
        {
            "Status": [
                "Complete",
                "Missing",
            ],

            "Cells": [
                complete_feature_cells,
                selected_missing_total,
            ],
        }
    )


    fig_completeness = px.pie(
        completeness_df,
        names="Status",
        values="Cells",
        hole=0.67,
    )


    fig_completeness.update_traces(
        textinfo="label+percent",
        textposition="inside",
    )


    fig_completeness.update_layout(
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
        fig_completeness,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


st.caption(
    "Only the six predictor columns are shown here. "
    "Missing numerical predictor values are handled inside the model pipeline."
)


# ============================================================
# BEFORE VS AFTER IMPUTATION
# ============================================================

section_title(
    "wrench",
    "Before vs After Median Imputation",
)


training_missing_counts = (
    X_train
    .isna()
    .sum()
)


max_missing = max(
    int(
        training_missing_counts.max()
    ),
    1,
)


imputation_rows = ""


for feature in features:

    before_count = int(
        training_missing_counts[
            feature
        ]
    )

    percentage_width = (
        before_count
        / max_missing
        * 100
    )

    imputation_rows += f"""
    <div class="imputation-row">

        <div class="imputation-name">
            {friendly_name(feature)}
        </div>

        <div>

            <div class="imputation-bar-wrap">

                <div
                    class="imputation-bar"
                    style="width:{percentage_width:.1f}%;">
                </div>

            </div>

            <div style="
                color:{chart_muted};
                font-size:10px;
                margin-top:4px;
            ">
                {before_count:,} missing
            </div>

        </div>

        <div class="imputation-arrow">
            →
        </div>

        <div class="imputation-after">
            0 missing ✓
        </div>

    </div>
    """


st.html(
    f"""
    <div class="imputation-container">
        {imputation_rows}
    </div>
    """
)


st.caption(
    "Missing predictor values are replaced using medians learned "
    "only from the training set."
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

section_title(
    "split",
    "Train / Test Split",
)


split_left, split_right = (
    st.columns(
        [1, 1]
    )
)


with split_left:

    train1, train2 = (
        st.columns(2)
    )


    train1.metric(
        "Training",
        f"{len(X_train):,}",
        "80%",
    )


    train2.metric(
        "Testing",
        f"{len(X_test):,}",
        "20%",
    )


    st.metric(
        "Random State",
        "42",
    )


with split_right:

    split_df = pd.DataFrame(
        {
            "Dataset": [
                "Training",
                "Testing",
            ],

            "Records": [
                len(X_train),
                len(X_test),
            ],
        }
    )


    fig_split = px.pie(
        split_df,
        names="Dataset",
        values="Records",
        hole=0.67,
    )


    fig_split.update_traces(
        textinfo=(
            "label+percent"
        ),
        textposition="inside",
    )


    fig_split.update_layout(
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
        fig_split,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ============================================================
# TRANSFORMATION EXPLORER
# ============================================================

section_title(
    "sliders",
    "Transformation Explorer",
)


st.caption(
    "Choose a predictor to see how the actual training data changes "
    "through the preprocessing pipeline."
)


selected_feature_label = (
    st.selectbox(
        "Feature to inspect",
        options=[
            friendly_name(
                feature
            )
            for feature in features
        ],
    )
)


selected_feature = next(
    feature
    for feature in features
    if friendly_name(
        feature
    )
    == selected_feature_label
)


raw_series = (
    X_train[
        selected_feature
    ]
)


imputed_series = (
    X_train_imputed[
        selected_feature
    ]
)


scaled_series = (
    X_train_scaled[
        selected_feature
    ]
)


transform_col1, transform_col2, transform_col3 = (
    st.columns(3)
)


with transform_col1:

    fig_raw = go.Figure()


    fig_raw.add_trace(
        go.Histogram(
            x=raw_series.dropna(),
            nbinsx=35,
            marker_color=accent,
        )
    )


    fig_raw.update_layout(
        height=285,

        title=dict(
            text="1 · Raw Training Data",
            font=dict(
                size=14
            ),
        ),

        margin=dict(
            l=20,
            r=15,
            t=40,
            b=30,
        ),

        xaxis_title=(
            selected_feature_label
        ),

        yaxis_title="Count",

        showlegend=False,
    )


    style_chart(
        fig_raw
    )


    st.plotly_chart(
        fig_raw,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


    st.caption(
        f"Missing: "
        f"{int(raw_series.isna().sum()):,}"
    )


with transform_col2:

    fig_imputed = go.Figure()


    fig_imputed.add_trace(
        go.Histogram(
            x=imputed_series,
            nbinsx=35,
            marker_color=accent,
        )
    )


    fig_imputed.update_layout(
        height=285,

        title=dict(
            text="2 · Median Imputed",
            font=dict(
                size=14
            ),
        ),

        margin=dict(
            l=20,
            r=15,
            t=40,
            b=30,
        ),

        xaxis_title=(
            selected_feature_label
        ),

        yaxis_title="Count",

        showlegend=False,
    )


    style_chart(
        fig_imputed
    )


    st.plotly_chart(
        fig_imputed,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


    st.caption(
        f"Median used: "
        f"{training_medians[selected_feature]:.2f}"
    )


with transform_col3:

    fig_scaled = go.Figure()


    fig_scaled.add_trace(
        go.Histogram(
            x=scaled_series,
            nbinsx=35,
            marker_color=accent,
        )
    )


    fig_scaled.add_vline(
        x=0,
        line_dash="dash",
        line_color=chart_muted,
    )


    fig_scaled.update_layout(
        height=285,

        title=dict(
            text="3 · Standardized",
            font=dict(
                size=14
            ),
        ),

        margin=dict(
            l=20,
            r=15,
            t=40,
            b=30,
        ),

        xaxis_title=(
            "Standardized Value"
        ),

        yaxis_title="Count",

        showlegend=False,
    )


    style_chart(
        fig_scaled
    )


    st.plotly_chart(
        fig_scaled,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


    st.caption(
        f"Mean ≈ 0.00 · "
        f"SD ≈ {scaled_series.std(ddof=0):.2f}"
    )


# ============================================================
# STANDARDIZATION CHECK
# ============================================================

section_title(
    "chart",
    "Standardization Check",
)


# Round tiny floating-point residuals to zero for display.

standardization_summary = pd.DataFrame(
    {
        "Feature": [
            friendly_name(
                col
            )
            for col in features
        ],

        "Mean": [
            0.0
            for _ in features
        ],

        "Std. Dev.": [
            round(
                float(
                    X_train_scaled[
                        col
                    ]
                    .std(
                        ddof=0
                    )
                ),
                4,
            )
            for col in features
        ],
    }
)


std_left, std_right = (
    st.columns(2)
)


with std_left:

    fig_means = go.Figure()


    fig_means.add_trace(
        go.Scatter(
            x=standardization_summary[
                "Feature"
            ],

            y=standardization_summary[
                "Mean"
            ],

            mode="markers",

            marker=dict(
                size=11,
                color=accent,
            ),

            text=[
                "0.00"
                for _ in features
            ],

            hovertemplate=(
                "%{x}<br>"
                "Mean: 0.00"
                "<extra></extra>"
            ),
        )
    )


    fig_means.add_hline(
        y=0,
        line_dash="dash",
        line_color=chart_muted,
    )


    fig_means.update_layout(
        height=280,

        title=dict(
            text=(
                "Standardized Feature Means"
            ),
            font=dict(
                size=14
            ),
        ),

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=55,
        ),

        xaxis_title="",

        yaxis=dict(
            title="Mean",
            range=[
                -0.10,
                0.10,
            ],
        ),

        showlegend=False,
    )


    style_chart(
        fig_means
    )


    st.plotly_chart(
        fig_means,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


with std_right:

    fig_std = go.Figure()


    fig_std.add_trace(
        go.Bar(
            x=standardization_summary[
                "Feature"
            ],

            y=standardization_summary[
                "Std. Dev."
            ],

            marker_color=accent,

            text=standardization_summary[
                "Std. Dev."
            ],

            texttemplate=(
                "%{text:.2f}"
            ),

            textposition=(
                "outside"
            ),
        )
    )


    fig_std.add_hline(
        y=1,
        line_dash="dash",
        line_color=chart_muted,
    )


    fig_std.update_layout(
        height=280,

        title=dict(
            text=(
                "Standardized Feature Spread"
            ),
            font=dict(
                size=14
            ),
        ),

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=55,
        ),

        xaxis_title="",

        yaxis=dict(
            title=(
                "Standard Deviation"
            ),

            range=[
                0,
                1.15,
            ],
        ),

        showlegend=False,
    )


    style_chart(
        fig_std
    )


    st.plotly_chart(
        fig_std,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


st.caption(
    "StandardScaler centers the training features around 0 "
    "and scales each feature to a standard deviation of approximately 1. "
    "Any extremely small numerical residuals are floating-point noise "
    "and are displayed here as 0.00."
)


# ============================================================
# MODEL FLOW
# ============================================================

section_title(
    "brain",
    "What Enters Each Model?",
)


pipeline_html = f"""
<div class="pipeline-flow">

    <div class="pipeline-step">

        <div class="pipeline-step-label">
            INPUT
        </div>

        <div class="pipeline-step-title">
            6 Predictors
        </div>

        <div class="pipeline-step-text">
            Previous score · GPA · attendance · assignments ·
            study hours · practice tests
        </div>

    </div>


    <div class="pipeline-arrow">
        →
    </div>


    <div class="pipeline-step">

        <div class="pipeline-step-label">
            STEP 1
        </div>

        <div class="pipeline-step-title">
            Median Imputation
        </div>

        <div class="pipeline-step-text">
            Missing numerical values are filled using training-set medians.
        </div>

    </div>


    <div class="pipeline-arrow">
        →
    </div>


    <div class="pipeline-step">

        <div class="pipeline-step-label">
            STEP 2
        </div>

        <div class="pipeline-step-title">
            Standardization
        </div>

        <div class="pipeline-step-text">
            Numerical inputs are centered and scaled.
        </div>

    </div>


    <div class="pipeline-arrow">
        →
    </div>


    <div class="pipeline-step">

        <div class="pipeline-step-label">
            MODEL
        </div>

        <div class="pipeline-step-title">
            Linear / Logistic
        </div>

        <div class="pipeline-step-text">
            Exam-score prediction or Pass / Fail probability.
        </div>

    </div>

</div>
"""


st.html(
    pipeline_html
)


pipeline1, pipeline2 = (
    st.columns(2)
)


with pipeline1:

    icon_card(
        "trending",
        "Linear Regression Pipeline",
        "6 inputs → median imputation → standardization → "
        "Linear Regression → exam score",
    )


with pipeline2:

    icon_card(
        "target",
        "Balanced Logistic Pipeline",
        "6 inputs → median imputation → standardization → "
        "Balanced Logistic Regression → Pass / Fail probability",
    )


# ============================================================
# DATA LEAKAGE PROTECTION
# ============================================================

section_title(
    "check",
    "Data Leakage Protection",
)


leak1, leak2 = (
    st.columns(2)
)


with leak1:

    icon_card(
        "check",
        "Pre-Exam Inputs",
        "The target exam score is not used as one of the six predictors.",
    )


with leak2:

    icon_card(
        "check",
        "Training-Only Fitting",
        "Imputation and scaling are fitted on training data before "
        "being applied to unseen test data.",
    )


# ============================================================
# MODEL-READY STRUCTURE
# ============================================================

section_title(
    "database",
    "Model-Ready Structure",
)


preview = (
    df[
        model_columns
    ]
    .head(8)
    .copy()
)


preview = preview.rename(
    columns=feature_labels
)


show_theme_table(
    preview
)


st.caption(
    "The prediction interface uses this same six-feature structure "
    "when a new student profile is submitted."
)


# ============================================================
# FOOTER
# ============================================================

show_footer()