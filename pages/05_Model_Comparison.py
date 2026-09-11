# ============================================================
# MODEL COMPARISON PAGE
# Visual-first final version
# ============================================================

import html

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
    icon_card,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Comparison",
    page_icon="⚖️",
    layout="wide",
)

apply_global_styles()


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
# THEME-AWARE HTML TABLE
# ============================================================

def show_theme_table(table_df):

    clean_df = (
        table_df
        .copy()
        .fillna("—")
    )

    headers = "".join(
        f"<th>{html.escape(str(col))}</th>"
        for col in clean_df.columns
    )

    rows = ""

    for _, row in clean_df.iterrows():

        cells = "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in row
        )

        rows += f"<tr>{cells}</tr>"

    table_html = f"""
    <style>

    .model-table-wrap {{
        width:100%;
        overflow-x:auto;
        margin:6px 0 14px 0;
        border:1px solid {chart_border};
        border-radius:10px;
        background:{surface};
    }}

    .model-table {{
        width:100%;
        border-collapse:collapse;
        font-size:13px;
        color:{chart_text};
        background:{surface};
    }}

    .model-table th {{
        text-align:left;
        padding:11px 12px;
        font-weight:600;
        color:{chart_muted};
        background:{surface};
        border-bottom:1px solid {chart_border};
        white-space:nowrap;
    }}

    .model-table td {{
        padding:10px 12px;
        color:{chart_text};
        background:{surface};
        border-bottom:1px solid {chart_border};
        white-space:nowrap;
    }}

    .model-table tbody tr:last-child td {{
        border-bottom:none;
    }}

    .model-table th:not(:last-child),
    .model-table td:not(:last-child) {{
        border-right:1px solid {chart_border};
    }}

    .model-table tbody tr:hover td {{
        background:rgba(91,141,184,0.08);
    }}

    </style>

    <div class="model-table-wrap">

        <table class="model-table">

            <thead>
                <tr>{headers}</tr>
            </thead>

            <tbody>
                {rows}
            </tbody>

        </table>

    </div>
    """

    st.html(
        table_html
    )


# ============================================================
# VERIFIED PROJECT METRICS
# ============================================================

REGRESSION_METRICS = {
    "MAE": 9.38,
    "RMSE": 11.76,
    "R²": 0.444,
}

BALANCED_LOGISTIC = {
    "Accuracy": 72.5,
    "Precision": 90.6,
    "Recall": 71.9,
    "F1 Score": 80.1,
    "Balanced Accuracy": 73.2,
    "AUC": 81.4,
}

CLASSIFICATION_COMPARISON = pd.DataFrame(
    {
        "Model": [
            "Dummy Baseline",
            "Standard Logistic",
            "Balanced Logistic",
        ],
        "Accuracy": [
            77.4,
            80.9,
            72.5,
        ],
        "Balanced Accuracy": [
            50.0,
            64.8,
            73.2,
        ],
    }
)

CLASSIFICATION_COMPARISON[
    "Accuracy Gap"
] = (
    CLASSIFICATION_COMPARISON[
        "Accuracy"
    ]
    - CLASSIFICATION_COMPARISON[
        "Balanced Accuracy"
    ]
)


# ============================================================
# HEADER
# ============================================================

page_title(
    "scale",
    "Model Comparison",
    "A visual comparison of the two prediction tasks and the "
    "models selected for the final system.",
)


# ============================================================
# MODEL SYSTEM OVERVIEW
# ============================================================

section_title(
    "brain",
    "How the Two Models Work Together",
)

system_html = f"""
<style>

.model-flow {{
    display:grid;
    grid-template-columns:1.25fr .30fr 1.25fr .30fr 1.25fr;
    gap:10px;
    align-items:center;
    margin:8px 0 22px 0;
}}

.flow-card {{
    background:{surface};
    border:1px solid {chart_border};
    border-radius:14px;
    padding:18px 16px;
    min-height:118px;
}}

.flow-label {{
    color:{accent};
    font-size:11px;
    font-weight:700;
    margin-bottom:6px;
}}

.flow-title {{
    color:{chart_text};
    font-size:16px;
    font-weight:700;
    margin-bottom:6px;
}}

.flow-text {{
    color:{chart_muted};
    font-size:11px;
    line-height:1.45;
}}

.flow-arrow {{
    text-align:center;
    color:{chart_muted};
    font-size:28px;
}}

@media(max-width:900px) {{

    .model-flow {{
        grid-template-columns:1fr;
    }}

    .flow-arrow {{
        transform:rotate(90deg);
    }}
}}

</style>

<div class="model-flow">

    <div class="flow-card">
        <div class="flow-label">INPUT</div>
        <div class="flow-title">6 Student Predictors</div>
        <div class="flow-text">
            Previous score · GPA · Attendance · Assignment completion ·
            Study hours · Practice tests
        </div>
    </div>

    <div class="flow-arrow">→</div>

    <div class="flow-card">
        <div class="flow-label">MODEL 1</div>
        <div class="flow-title">Linear Regression</div>
        <div class="flow-text">
            Produces a continuous numerical estimate of the student's
            expected exam score.
        </div>
    </div>

    <div class="flow-arrow">→</div>

    <div class="flow-card">
        <div class="flow-label">OUTPUT</div>
        <div class="flow-title">Exam Score</div>
        <div class="flow-text">
            Example format: 68.4 / 100
        </div>
    </div>

</div>

<div class="model-flow">

    <div class="flow-card">
        <div class="flow-label">SAME INPUT</div>
        <div class="flow-title">6 Student Predictors</div>
        <div class="flow-text">
            The classification model receives the same six
            pre-exam variables.
        </div>
    </div>

    <div class="flow-arrow">→</div>

    <div class="flow-card">
        <div class="flow-label">MODEL 2</div>
        <div class="flow-title">Balanced Logistic Regression</div>
        <div class="flow-text">
            Estimates the probability of passing and converts that
            probability into Pass / Fail.
        </div>
    </div>

    <div class="flow-arrow">→</div>

    <div class="flow-card">
        <div class="flow-label">OUTPUT</div>
        <div class="flow-title">Class + Probability</div>
        <div class="flow-text">
            Example format: Pass · 73% probability
        </div>
    </div>

</div>
"""

st.html(
    system_html
)

st.info(
    "The models solve different tasks. Their metrics should therefore "
    "be interpreted separately rather than directly compared as if they "
    "were competing for the same target."
)


# ============================================================
# LINEAR REGRESSION
# ============================================================

section_title(
    "trending",
    "Linear Regression",
)


# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------

r1, r2, r3 = st.columns(3)

r1.metric(
    "MAE",
    "9.38",
)

r2.metric(
    "RMSE",
    "11.76",
)

r3.metric(
    "R²",
    "0.444",
)

st.caption(
    "Score prediction · Lower MAE/RMSE is better · Higher R² is better."
)


# ============================================================
# REGRESSION VISUALS
# ============================================================

reg1, reg2, reg3 = st.columns(
    [1, 1, 1]
)


# ------------------------------------------------------------
# MAE vs RMSE
# ------------------------------------------------------------

with reg1:

    st.markdown(
        "#### Prediction Error"
    )

    regression_error_df = pd.DataFrame(
        {
            "Metric": [
                "MAE",
                "RMSE",
            ],
            "Value": [
                9.38,
                11.76,
            ],
        }
    )

    fig_error = px.bar(
        regression_error_df,
        x="Metric",
        y="Value",
        text="Value",
    )

    fig_error.update_traces(
        marker_color=accent,
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    fig_error.update_layout(
        height=315,
        margin=dict(
            l=15,
            r=15,
            t=10,
            b=20,
        ),
        xaxis_title="",
        yaxis_title="Score Points",
        showlegend=False,
    )

    style_chart(
        fig_error
    )

    st.plotly_chart(
        fig_error,
        width="stretch",
        theme=None,
        config={
            "displayModeBar": False
        },
    )


# ------------------------------------------------------------
# R² GAUGE
# ------------------------------------------------------------

with reg2:

    st.markdown(
        "#### Variance Explained"
    )

    fig_r2 = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=44.4,
            number={
                "suffix": "%"
            },
            title={
                "text":
                    "R² = 0.444"
            },
            gauge={
                "axis": {
                    "range": [
                        0,
                        100,
                    ]
                },
                "bar": {
                    "color": accent
                },
                "bgcolor":
                    surface,
                "bordercolor":
                    chart_border,
            },
        )
    )

    fig_r2.update_layout(
        height=315,
        margin=dict(
            l=25,
            r=25,
            t=35,
            b=20,
        ),
        paper_bgcolor=chart_background,
        font=dict(
            color=chart_text
        ),
    )

    st.plotly_chart(
        fig_r2,
        width="stretch",
        theme=None,
        config={
            "displayModeBar": False
        },
    )


# ------------------------------------------------------------
# EXPLAINED VS UNEXPLAINED
# ------------------------------------------------------------

with reg3:

    st.markdown(
        "#### Explained vs Unexplained"
    )

    variance_df = pd.DataFrame(
        {
            "Component": [
                "Explained",
                "Unexplained",
            ],
            "Percentage": [
                44.4,
                55.6,
            ],
        }
    )

    fig_variance = px.pie(
        variance_df,
        names="Component",
        values="Percentage",
        hole=0.64,
    )

    fig_variance.update_traces(
        textinfo="label+percent",
        textposition="inside",
    )

    fig_variance.update_layout(
        template=plotly_template,
        height=315,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        showlegend=False,
        paper_bgcolor=chart_background,
        plot_bgcolor=chart_background,
        font=dict(
            color=chart_text
        ),
    )

    st.plotly_chart(
        fig_variance,
        width="stretch",
        theme=None,
        config={
            "displayModeBar": False
        },
    )


# ============================================================
# REGRESSION INTERPRETATION
# ============================================================

reg_note1, reg_note2, reg_note3 = (
    st.columns(3)
)

with reg_note1:

    icon_card(
        "chart",
        "MAE = 9.38",
        "Predictions differ from the actual score by about "
        "9.38 points on average.",
    )

with reg_note2:

    icon_card(
        "chart",
        "RMSE = 11.76",
        "Larger prediction errors receive a stronger penalty.",
    )

with reg_note3:

    icon_card(
        "trending",
        "R² = 0.444",
        "The six predictors explain 44.4% of the observed "
        "variation in exam scores.",
    )


# ============================================================
# BALANCED LOGISTIC REGRESSION
# ============================================================

section_title(
    "target",
    "Balanced Logistic Regression",
)


# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------

classification_metric_cols = (
    st.columns(6)
)

classification_metric_data = [
    ("Accuracy", "72.5%"),
    ("Precision", "90.6%"),
    ("Recall", "71.9%"),
    ("F1", "80.1%"),
    (
        "Balanced Acc.",
        "73.2%",
    ),
    ("AUC", "0.814"),
]

for col, (
    metric_name,
    metric_value,
) in zip(
    classification_metric_cols,
    classification_metric_data,
):

    with col:

        st.metric(
            metric_name,
            metric_value,
        )


# ============================================================
# CLASSIFICATION METRIC PROFILE
# ============================================================

cls_left, cls_right = (
    st.columns(
        [1.4, 1]
    )
)


# ------------------------------------------------------------
# BAR PROFILE
# ------------------------------------------------------------

with cls_left:

    section_title(
        "chart",
        "Classification Metric Profile",
    )

    classification_metric_df = (
        pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "Balanced Accuracy",
                    "AUC",
                ],
                "Score": [
                    72.5,
                    90.6,
                    71.9,
                    80.1,
                    73.2,
                    81.4,
                ],
            }
        )
    )

    fig_cls_metrics = (
        px.bar(
            classification_metric_df,
            x="Score",
            y="Metric",
            orientation="h",
            text="Score",
        )
    )

    fig_cls_metrics.update_traces(
        marker_color=accent,
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    fig_cls_metrics.update_layout(
        height=390,
        margin=dict(
            l=20,
            r=65,
            t=10,
            b=20,
        ),
        xaxis=dict(
            title="Score (%)",
            range=[
                0,
                100,
            ],
            ticksuffix="%",
        ),
        yaxis_title="",
        showlegend=False,
    )

    style_chart(
        fig_cls_metrics
    )

    st.plotly_chart(
        fig_cls_metrics,
        width="stretch",
        theme=None,
        config={
            "displayModeBar": False
        },
    )


# ------------------------------------------------------------
# RADAR CHART
# ------------------------------------------------------------

with cls_right:

    section_title(
        "chart",
        "Metric Balance",
    )

    radar_labels = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Balanced Acc.",
        "AUC",
    ]

    radar_values = [
        72.5,
        90.6,
        71.9,
        80.1,
        73.2,
        81.4,
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

    fig_radar = (
        go.Figure()
    )

    fig_radar.add_trace(
        go.Scatterpolar(
            r=radar_values_closed,
            theta=radar_labels_closed,
            fill="toself",
            name=(
                "Balanced Logistic"
            ),
            line=dict(
                color=accent
            ),
        )
    )

    fig_radar.update_layout(
        template=plotly_template,
        height=390,
        margin=dict(
            l=45,
            r=45,
            t=25,
            b=25,
        ),
        polar=dict(
            bgcolor=chart_background,
            radialaxis=dict(
                visible=True,
                range=[
                    0,
                    100,
                ],
                ticksuffix="%",
                gridcolor=chart_border,
            ),
            angularaxis=dict(
                gridcolor=chart_border,
            ),
        ),
        showlegend=False,
        paper_bgcolor=chart_background,
        font=dict(
            color=chart_text
        ),
    )

    st.plotly_chart(
        fig_radar,
        width="stretch",
        theme=None,
        config={
            "displayModeBar": False
        },
    )


# ============================================================
# CLASSIFICATION MODEL COMPARISON
# ============================================================

section_title(
    "scale",
    "Which Classifier Should Be Used?",
)


# ------------------------------------------------------------
# TABLE
# ------------------------------------------------------------

classification_display = (
    CLASSIFICATION_COMPARISON[
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
        ]
    ]
    .copy()
)

classification_display[
    "Accuracy"
] = classification_display[
    "Accuracy"
].map(
    lambda x:
        f"{x:.1f}%"
)

classification_display[
    "Balanced Accuracy"
] = classification_display[
    "Balanced Accuracy"
].map(
    lambda x:
        f"{x:.1f}%"
)

show_theme_table(
    classification_display
)


# ============================================================
# ACCURACY VS BALANCED ACCURACY
# ============================================================

classification_long = (
    CLASSIFICATION_COMPARISON
    .melt(
        id_vars="Model",
        value_vars=[
            "Accuracy",
            "Balanced Accuracy",
        ],
        var_name="Metric",
        value_name="Score",
    )
)

fig_compare = (
    px.bar(
        classification_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
    )
)

fig_compare.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside",
)

fig_compare.update_layout(
    height=420,
    margin=dict(
        l=20,
        r=20,
        t=15,
        b=20,
    ),
    xaxis_title="",
    yaxis=dict(
        title="Score (%)",
        range=[
            0,
            100,
        ],
        ticksuffix="%",
    ),
    legend_title="",
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
    },
)


# ============================================================
# ACCURACY GAP VISUAL
# ============================================================

section_title(
    "chart",
    "Why Ordinary Accuracy Can Be Misleading",
)

gap_df = (
    CLASSIFICATION_COMPARISON[
        [
            "Model",
            "Accuracy Gap",
        ]
    ]
    .copy()
)

fig_gap = (
    px.bar(
        gap_df,
        x="Accuracy Gap",
        y="Model",
        orientation="h",
        text="Accuracy Gap",
    )
)

fig_gap.update_traces(
    marker_color=accent,
    texttemplate="%{text:+.1f} pts",
    textposition="outside",
)

fig_gap.add_vline(
    x=0,
    line_dash="dash",
    line_color=chart_muted,
)

fig_gap.update_layout(
    height=290,
    margin=dict(
        l=20,
        r=80,
        t=10,
        b=20,
    ),
    xaxis_title=(
        "Accuracy − Balanced Accuracy "
        "(percentage points)"
    ),
    yaxis_title="",
    showlegend=False,
)

style_chart(
    fig_gap
)

st.plotly_chart(
    fig_gap,
    width="stretch",
    theme=None,
    config={
        "displayModeBar": False
    },
)

st.caption(
    "A large positive gap suggests that ordinary accuracy may be "
    "benefiting from the majority class. Balanced Logistic Regression "
    "has nearly equal Accuracy and Balanced Accuracy."
)


# ============================================================
# BALANCED ACCURACY RANKING
# ============================================================

section_title(
    "chart",
    "Balanced Accuracy Ranking",
)

balanced_ranking_df = (
    CLASSIFICATION_COMPARISON[
        [
            "Model",
            "Balanced Accuracy",
        ]
    ]
    .sort_values(
        "Balanced Accuracy",
        ascending=True,
    )
)

fig_rank = (
    px.bar(
        balanced_ranking_df,
        x="Balanced Accuracy",
        y="Model",
        orientation="h",
        text="Balanced Accuracy",
    )
)

fig_rank.update_traces(
    marker_color=accent,
    texttemplate="%{text:.1f}%",
    textposition="outside",
)

fig_rank.update_layout(
    height=300,
    margin=dict(
        l=20,
        r=70,
        t=10,
        b=20,
    ),
    xaxis=dict(
        title=(
            "Balanced Accuracy (%)"
        ),
        range=[
            0,
            100,
        ],
        ticksuffix="%",
    ),
    yaxis_title="",
    showlegend=False,
)

style_chart(
    fig_rank
)

st.plotly_chart(
    fig_rank,
    width="stretch",
    theme=None,
    config={
        "displayModeBar": False
    },
)


# ============================================================
# CLASSIFIER DECISION CARDS
# ============================================================

section_title(
    "check",
    "Classification Model Selection",
)

sel1, sel2, sel3 = (
    st.columns(3)
)

with sel1:

    icon_card(
        "database",
        "Dummy Baseline",
        "Accuracy looks high because most students belong "
        "to the Pass class.",
    )

    st.metric(
        "Balanced Accuracy",
        "50.0%",
    )


with sel2:

    icon_card(
        "target",
        "Standard Logistic",
        "Improves overall classification but still favors "
        "the majority class.",
    )

    st.metric(
        "Balanced Accuracy",
        "64.8%",
    )


with sel3:

    icon_card(
        "check",
        "Balanced Logistic",
        "Provides the strongest performance across both "
        "Pass and Fail classes.",
    )

    st.metric(
        "Balanced Accuracy",
        "73.2%",
    )


st.success(
    "Selected classifier: Balanced Logistic Regression — "
    "highest Balanced Accuracy among the evaluated classification models."
)


# ============================================================
# FINAL SYSTEM OUTPUT
# ============================================================

section_title(
    "sparkles",
    "Final Prediction System",
)

final1, final2 = (
    st.columns(2)
)

with final1:

    icon_card(
        "trending",
        "Exam Score Prediction",
        "Linear Regression",
    )

    st.metric(
        "Typical Error",
        "MAE 9.38",
    )

    st.caption(
        "Produces a continuous numerical score estimate."
    )


with final2:

    icon_card(
        "target",
        "Pass / Fail Prediction",
        "Balanced Logistic Regression",
    )

    st.metric(
        "Balanced Accuracy",
        "73.2%",
    )

    st.caption(
        "Produces Pass / Fail together with Pass probability."
    )


# ============================================================
# FINAL MODEL SUMMARY TABLE
# ============================================================

section_title(
    "brain",
    "Final Model Summary",
)

summary_df = pd.DataFrame(
    {
        "Model": [
            "Linear Regression",
            "Balanced Logistic Regression",
        ],
        "Task": [
            "Regression",
            "Classification",
        ],
        "Target": [
            "Exam Score",
            "Pass / Fail",
        ],
        "Output": [
            "Numerical score",
            "Class + probability",
        ],
        "Key Performance": [
            (
                "MAE 9.38 | "
                "RMSE 11.76 | "
                "R² 0.444"
            ),
            (
                "Balanced Accuracy 73.2% | "
                "F1 80.1% | "
                "AUC 0.814"
            ),
        ],
        "Final Use": [
            "Selected",
            "Selected",
        ],
    }
)

show_theme_table(
    summary_df
)


# ============================================================
# MODEL SELECTION LOGIC
# ============================================================

section_title(
    "filter",
    "Why These Two Models?",
)

reason1, reason2, reason3 = (
    st.columns(3)
)

with reason1:

    icon_card(
        "brain",
        "Interpretable",
        "Both models are straightforward to explain and "
        "appropriate for an academic ML project.",
    )


with reason2:

    icon_card(
        "scale",
        "Different Tasks",
        "Regression predicts score while classification "
        "predicts Pass / Fail.",
    )


with reason3:

    icon_card(
        "check",
        "Useful Together",
        "A single student profile produces both a numerical "
        "estimate and a classification outcome.",
    )


# ============================================================
# RESPONSIBLE INTERPRETATION
# ============================================================

section_title(
    "settings",
    "Model Limitations",
)

lim1, lim2, lim3 = (
    st.columns(3)
)

with lim1:

    icon_card(
        "chart",
        "Moderate R²",
        "The regression model explains 44.4% of score variation, "
        "so substantial variation remains unexplained.",
    )


with lim2:

    icon_card(
        "scale",
        "Class Imbalance",
        "Pass students are more common than Fail students, "
        "which makes Balanced Accuracy important.",
    )


with lim3:

    icon_card(
        "sliders",
        "Prediction, Not Certainty",
        "Outputs are model estimates and should not be interpreted "
        "as guaranteed student outcomes.",
    )


# ============================================================
# TAKEAWAY
# ============================================================

section_title(
    "graduation",
    "Model Comparison Takeaway",
)

st.success(
    "The final system uses Linear Regression for exam-score estimation "
    "and Balanced Logistic Regression for Pass / Fail classification. "
    "The two models complement each other rather than compete."
)


# ============================================================
# FOOTER
# ============================================================

show_footer()