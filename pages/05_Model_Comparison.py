# ============================================================
# MODEL COMPARISON PAGE
# ============================================================

import streamlit as st
import pandas as pd
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
    icon_card
)


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Model Comparison",
    page_icon="⚖️",
    layout="wide"
)

apply_global_styles()


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
# THEME-AWARE HTML TABLE
# ------------------------------------------------------------

def show_theme_table(
    table_df,
    max_height=None
):

    clean_df = (
        table_df
        .copy()
        .fillna("—")
    )

    table_html = clean_df.to_html(
        index=False,
        escape=True,
        border=0,
        classes="comparison-table"
    )

    if max_height is not None:

        container_style = (
            f"max-height:{max_height}px;"
            "overflow:auto;"
        )

    else:

        container_style = (
            "overflow-x:auto;"
        )

    html = f"""
    <style>

    .comparison-table-container {{
        {container_style}
        border:1px solid {chart_border};
        border-radius:10px;
        background:{surface};
        margin-bottom:14px;
    }}

    .comparison-table {{
        width:100%;
        border-collapse:collapse;
        font-size:13px;
        color:{chart_text};
        background:{surface};
    }}

    .comparison-table thead th {{
        text-align:left;
        padding:10px 11px;
        font-weight:600;
        color:{chart_muted};
        background:{surface};
        border-bottom:1px solid {chart_border};
        white-space:nowrap;
    }}

    .comparison-table tbody td {{
        padding:9px 11px;
        color:{chart_text};
        background:{surface};
        border-bottom:1px solid {chart_border};
        white-space:nowrap;
    }}

    .comparison-table tbody tr:last-child td {{
        border-bottom:none;
    }}

    </style>

    <div class="comparison-table-container">
        {table_html}
    </div>
    """

    st.html(
        html
    )


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

page_title(
    "scale",
    "Model Comparison",
    "Compare the regression and classification models used "
    "in the final prediction system."
)


# ------------------------------------------------------------
# MODEL PURPOSE
# ------------------------------------------------------------

section_title(
    "target",
    "What Does Each Model Predict?"
)


c1, c2 = st.columns(2)


with c1:

    icon_card(
        "trending",
        "Linear Regression",
        "Predicts the student's expected numerical exam score "
        "on a 0–100 scale."
    )

    st.caption(
        "Example output: Predicted Exam Score = 68.4"
    )


with c2:

    icon_card(
        "target",
        "Balanced Logistic Regression",
        "Predicts whether the student is likely to Pass or Fail "
        "and provides a probability."
    )

    st.caption(
        "Example output: Pass with 73% probability"
    )


st.info(
    "Regression and classification solve different prediction tasks, "
    "so their evaluation metrics should be interpreted separately."
)


# ------------------------------------------------------------
# LINEAR REGRESSION PERFORMANCE
# ------------------------------------------------------------

section_title(
    "trending",
    "Linear Regression Performance"
)


r1, r2, r3 = st.columns(3)


with r1:

    with st.container(
        border=True
    ):

        st.metric(
            "MAE",
            "9.38"
        )

        st.caption(
            "Average absolute error"
        )


with r2:

    with st.container(
        border=True
    ):

        st.metric(
            "RMSE",
            "11.76"
        )

        st.caption(
            "Penalizes larger errors"
        )


with r3:

    with st.container(
        border=True
    ):

        st.metric(
            "R²",
            "0.444"
        )

        st.caption(
            "Variation explained"
        )


st.caption(
    "The model predicts exam scores with an average absolute error "
    "of 9.38 points and explains about 44.4% of score variation."
)


# ------------------------------------------------------------
# REGRESSION VISUALS
# ------------------------------------------------------------

reg_col1, reg_col2 = st.columns(2)


# ------------------------------------------------------------
# REGRESSION ERROR METRICS
# ------------------------------------------------------------

with reg_col1:

    section_title(
        "chart",
        "Regression Error Metrics"
    )


    regression_error_df = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE"
            ],
            "Value": [
                9.38,
                11.76
            ]
        }
    )


    fig_reg_error = px.bar(
        regression_error_df,
        x="Metric",
        y="Value",
        text="Value"
    )


    fig_reg_error.update_traces(
        marker_color=accent,
        texttemplate="%{text:.2f}",
        textposition="outside"
    )


    fig_reg_error.update_layout(
        height=330,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=20
        ),

        yaxis_title="Exam-Score Points",
        xaxis_title="Metric",

        showlegend=False
    )


    style_chart(
        fig_reg_error
    )


    st.plotly_chart(
        fig_reg_error,
        width="stretch",
        theme=None,

        config={
            "displayModeBar": False
        }
    )


    st.caption(
        "Lower MAE and RMSE values indicate better numerical predictions."
    )


# ------------------------------------------------------------
# EXPLAINED VARIATION
# ------------------------------------------------------------

with reg_col2:

    section_title(
        "chart",
        "Explained Variation"
    )


    variation_df = pd.DataFrame(
        {
            "Component": [
                "Explained",
                "Unexplained"
            ],
            "Percentage": [
                44.4,
                55.6
            ]
        }
    )


    fig_variation = px.pie(
        variation_df,
        names="Component",
        values="Percentage",
        hole=0.62
    )


    fig_variation.update_traces(
        textinfo="label+percent",
        textposition="inside"
    )


    fig_variation.update_layout(
        template=plotly_template,

        height=330,

        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10
        ),

        showlegend=False,

        paper_bgcolor=chart_background,
        plot_bgcolor=chart_background,

        font=dict(
            color=chart_text
        )
    )


    st.plotly_chart(
        fig_variation,
        width="stretch",
        theme=None,

        config={
            "displayModeBar": False
        }
    )


    st.caption(
        "R² = 0.444 means 44.4% of exam-score variation "
        "is explained by the six selected predictors."
    )


# ------------------------------------------------------------
# LOGISTIC REGRESSION PERFORMANCE
# ------------------------------------------------------------

section_title(
    "target",
    "Balanced Logistic Regression Performance"
)


metric_names = [
    ("Accuracy", "72.5%"),
    ("Precision", "90.6%"),
    ("Recall", "71.9%"),
    ("F1 Score", "80.1%"),
    ("Balanced Accuracy", "73.2%")
]


metric_cols = st.columns(5)


for col, (
    name,
    value
) in zip(
    metric_cols,
    metric_names
):

    with col:

        with st.container(
            border=True
        ):

            st.metric(
                name,
                value
            )


st.caption(
    "Balanced Logistic Regression sacrifices some ordinary accuracy "
    "to classify Pass and Fail students more evenly."
)


# ------------------------------------------------------------
# CLASSIFICATION METRICS
# ------------------------------------------------------------

section_title(
    "chart",
    "Classification Evaluation Metrics"
)


classification_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Balanced Accuracy"
        ],

        "Score": [
            72.5,
            90.6,
            71.9,
            80.1,
            73.2
        ]
    }
)


fig_metrics = px.bar(
    classification_metrics_df,
    x="Metric",
    y="Score",
    text="Score"
)


fig_metrics.update_traces(
    marker_color=accent,
    texttemplate="%{text:.1f}%",
    textposition="outside"
)


fig_metrics.update_layout(
    height=380,

    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20
    ),

    xaxis_title="",

    yaxis=dict(
        title="Score (%)",
        range=[0, 100],
        ticksuffix="%"
    ),

    showlegend=False
)


style_chart(
    fig_metrics
)


st.plotly_chart(
    fig_metrics,
    width="stretch",
    theme=None,

    config={
        "displayModeBar": False
    }
)


st.caption(
    "Precision is highest, while balanced accuracy is especially "
    "important because the Pass and Fail classes are imbalanced."
)


# ------------------------------------------------------------
# CLASSIFICATION MODEL COMPARISON
# ------------------------------------------------------------

section_title(
    "scale",
    "Classification Model Comparison"
)


classification_df = pd.DataFrame(
    {
        "Model": [
            "Dummy Baseline",
            "Standard Logistic Regression",
            "Balanced Logistic Regression"
        ],

        "Accuracy": [
            77.4,
            80.9,
            72.5
        ],

        "Balanced Accuracy": [
            50.0,
            64.8,
            73.2
        ]
    }
)


classification_display = (
    classification_df
    .copy()
)


classification_display[
    "Accuracy"
] = (
    classification_display[
        "Accuracy"
    ]
    .map(
        lambda value:
            f"{value:.1f}%"
    )
)


classification_display[
    "Balanced Accuracy"
] = (
    classification_display[
        "Balanced Accuracy"
    ]
    .map(
        lambda value:
            f"{value:.1f}%"
    )
)


show_theme_table(
    classification_display
)


classification_long = (
    classification_df
    .melt(
        id_vars="Model",

        value_vars=[
            "Accuracy",
            "Balanced Accuracy"
        ],

        var_name="Metric",
        value_name="Score"
    )
)


fig_compare = px.bar(
    classification_long,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group"
)


fig_compare.update_layout(
    height=420,

    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20
    ),

    xaxis_title="",

    yaxis=dict(
        title="Score (%)",
        range=[0, 100],
        ticksuffix="%"
    ),

    legend_title=""
)


style_chart(
    fig_compare
)


st.plotly_chart(
    fig_compare,
    width="stretch",
    theme=None,

    config={
        "displayModeBar": False
    }
)


st.warning(
    "The Dummy Classifier appears strong on ordinary accuracy because "
    "most students pass. Its Balanced Accuracy is only 50%, showing "
    "that accuracy alone can be misleading."
)


# ------------------------------------------------------------
# BALANCED ACCURACY RANKING
# ------------------------------------------------------------

section_title(
    "chart",
    "Balanced Accuracy Ranking"
)


balanced_ranking_df = (
    classification_df[
        [
            "Model",
            "Balanced Accuracy"
        ]
    ]
    .sort_values(
        "Balanced Accuracy",
        ascending=True
    )
)


fig_balanced_rank = px.bar(
    balanced_ranking_df,
    x="Balanced Accuracy",
    y="Model",
    orientation="h",
    text="Balanced Accuracy"
)


fig_balanced_rank.update_traces(
    marker_color=accent,
    texttemplate="%{text:.1f}%",
    textposition="outside"
)


fig_balanced_rank.update_layout(
    height=300,

    margin=dict(
        l=20,
        r=70,
        t=10,
        b=20
    ),

    xaxis=dict(
        title="Balanced Accuracy (%)",
        range=[0, 100],
        ticksuffix="%"
    ),

    yaxis_title="",

    showlegend=False
)


style_chart(
    fig_balanced_rank
)


st.plotly_chart(
    fig_balanced_rank,
    width="stretch",
    theme=None,

    config={
        "displayModeBar": False
    }
)


st.caption(
    "Balanced Logistic Regression achieves the strongest balanced "
    "accuracy and is therefore selected as the final classifier."
)


# ------------------------------------------------------------
# FINAL MODEL SUMMARY
# ------------------------------------------------------------

section_title(
    "brain",
    "Final Model Summary"
)


summary_df = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Balanced Logistic Regression"
        ],

        "Prediction Type": [
            "Regression",
            "Classification"
        ],

        "Target": [
            "Exam Score",
            "Pass / Fail"
        ],

        "Main Output": [
            "Numerical score",
            "Class + probability"
        ],

        "Key Evaluation": [
            "MAE 9.38 | RMSE 11.76 | R² 0.444",
            "Balanced Accuracy 73.2% | F1 80.1% | AUC 0.814"
        ]
    }
)


show_theme_table(
    summary_df
)


# ------------------------------------------------------------
# MODEL ROLES
# ------------------------------------------------------------

section_title(
    "brain",
    "Model Roles in the Final System"
)


role1, plus_col, role2 = st.columns(
    [1, 0.15, 1]
)


with role1:

    icon_card(
        "trending",
        "Numerical Prediction",
        "Linear Regression converts the six student inputs "
        "into an expected exam score."
    )

    st.caption(
        "Example: 68.4 / 100"
    )


with plus_col:

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:30px;
            padding-top:45px;
            color:{chart_muted};
        ">
            +
        </div>
        """,
        unsafe_allow_html=True
    )


with role2:

    icon_card(
        "target",
        "Classification Prediction",
        "Balanced Logistic Regression produces Pass / Fail "
        "status together with pass probability."
    )

    st.caption(
        "Example: Pass · 73% probability"
    )


# ------------------------------------------------------------
# WHEN TO USE EACH MODEL
# ------------------------------------------------------------

section_title(
    "filter",
    "When Should Each Model Be Used?"
)


use1, use2 = st.columns(2)


with use1:

    icon_card(
        "trending",
        "Use Linear Regression",
        "When you want an estimate of the student's expected "
        "numerical exam score."
    )


with use2:

    icon_card(
        "target",
        "Use Logistic Regression",
        "When you want Pass / Fail classification together "
        "with a probability estimate."
    )


# ------------------------------------------------------------
# WHY BOTH MODELS ARE USED
# ------------------------------------------------------------

section_title(
    "check",
    "Why Use Both Models?"
)


st.success(
    "The two models complement each other. Linear Regression estimates "
    "the expected exam score, while Balanced Logistic Regression estimates "
    "Pass / Fail status and pass probability."
)


st.caption(
    "The prediction page uses the same six inputs once and returns "
    "both types of model output."
)


# ------------------------------------------------------------
# PROJECT TAKEAWAY
# ------------------------------------------------------------

section_title(
    "graduation",
    "Project Takeaway"
)


take1, take2, take3 = st.columns(3)


with take1:

    icon_card(
        "database",
        "Useful Inputs",
        "Previous academic performance and study behavior contain "
        "meaningful predictive information."
    )


with take2:

    icon_card(
        "trending",
        "Score Estimate",
        "Linear Regression provides a useful numerical estimate "
        "rather than an exact exam result."
    )


with take3:

    icon_card(
        "scale",
        "Balanced Classification",
        "Balanced Logistic Regression provides a more even evaluation "
        "of Pass and Fail outcomes."
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

show_footer()