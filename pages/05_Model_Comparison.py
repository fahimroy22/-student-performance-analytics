# ============================================================
# MODEL COMPARISON PAGE
# Visual-first final version
# + Fairness / Subgroup Model Audit
# ============================================================

from pathlib import Path
import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
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

from core.model_loader import (
    load_linear_model,
    load_logistic_model,
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
# PATHS + DATA
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
    return pd.read_csv(DATA_PATH)


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
secondary_color = "#5EA8A1"


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
# MODEL AUDIT SETUP
# ============================================================

features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed",
]


@st.cache_resource
def get_linear_model():
    return load_linear_model()


@st.cache_resource
def get_logistic_model():
    return load_logistic_model()


linear_model = get_linear_model()
logistic_model = get_logistic_model()


# ============================================================
# FAIRNESS / SUBGROUP TEST RESULTS
# ============================================================

@st.cache_data
def prepare_subgroup_test_results():

    X = df[
        features
    ].copy()


    # --------------------------------------------------------
    # REGRESSION HELD-OUT SET
    # --------------------------------------------------------

    y_reg = df[
        "exam_score"
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


    y_pred_reg = (
        linear_model.predict(
            X_test_reg
        )
    )


    regression_results = pd.DataFrame(
        {
            "Actual Score":
                y_test_reg.values,

            "Predicted Score":
                y_pred_reg,

            "Residual":
                y_test_reg.values
                - y_pred_reg,

            "Absolute Error":
                np.abs(
                    y_test_reg.values
                    - y_pred_reg
                ),
        },
        index=X_test_reg.index,
    )


    # --------------------------------------------------------
    # CLASSIFICATION HELD-OUT SET
    # --------------------------------------------------------

    y_cls = (
        df[
            "pass_status"
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


    y_pred_cls = (
        logistic_model.predict(
            X_test_cls
        )
    )


    classes = list(
        logistic_model
        .named_steps["model"]
        .classes_
    )


    pass_index = classes.index(
        "Pass"
    )


    pass_probability = (
        logistic_model
        .predict_proba(
            X_test_cls
        )[:, pass_index]
    )


    classification_results = pd.DataFrame(
        {
            "Actual Outcome":
                y_test_cls.values,

            "Predicted Outcome":
                y_pred_cls,

            "Pass Probability":
                pass_probability,
        },
        index=X_test_cls.index,
    )


    return (
        regression_results,
        classification_results,
    )


(
    regression_audit,
    classification_audit,
) = prepare_subgroup_test_results()


# ============================================================
# SUBGROUP METRIC CALCULATION
# ============================================================

def build_subgroup_metrics(
    subgroup_column
):

    # --------------------------------------------------------
    # REGRESSION
    # --------------------------------------------------------

    reg = (
        regression_audit
        .copy()
    )


    reg[
        "Subgroup"
    ] = (
        df.loc[
            reg.index,
            subgroup_column,
        ]
        .astype("object")
        .fillna("Missing / Unknown")
        .astype(str)
    )


    regression_group_rows = []


    for group_name, group_data in (
        reg.groupby(
            "Subgroup"
        )
    ):

        regression_group_rows.append(
            {
                "Subgroup":
                    group_name,

                "Regression N":
                    len(
                        group_data
                    ),

                "MAE":
                    mean_absolute_error(
                        group_data[
                            "Actual Score"
                        ],
                        group_data[
                            "Predicted Score"
                        ],
                    ),

                "Mean Residual":
                    group_data[
                        "Residual"
                    ].mean(),
            }
        )


    regression_groups = (
        pd.DataFrame(
            regression_group_rows
        )
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    cls = (
        classification_audit
        .copy()
    )


    cls[
        "Subgroup"
    ] = (
        df.loc[
            cls.index,
            subgroup_column,
        ]
        .astype("object")
        .fillna("Missing / Unknown")
        .astype(str)
    )


    classification_group_rows = []


    for group_name, group_data in (
        cls.groupby(
            "Subgroup"
        )
    ):

        cm_group = confusion_matrix(
            group_data[
                "Actual Outcome"
            ],
            group_data[
                "Predicted Outcome"
            ],
            labels=[
                "Fail",
                "Pass",
            ],
        )


        tn, fp, fn, tp = (
            cm_group.ravel()
        )


        fail_total = (
            tn + fp
        )

        pass_total = (
            tp + fn
        )


        fail_recall = (
            tn / fail_total
            if fail_total > 0
            else np.nan
        )


        pass_recall = (
            tp / pass_total
            if pass_total > 0
            else np.nan
        )


        if (
            np.isnan(
                fail_recall
            )
            or np.isnan(
                pass_recall
            )
        ):

            balanced_accuracy = (
                np.nan
            )

        else:

            balanced_accuracy = (
                (
                    fail_recall
                    + pass_recall
                )
                / 2
            )


        classification_group_rows.append(
            {
                "Subgroup":
                    group_name,

                "Classification N":
                    len(
                        group_data
                    ),

                "Balanced Accuracy":
                    balanced_accuracy
                    * 100,

                "Pass Recall":
                    pass_recall
                    * 100,

                "Fail Recall":
                    fail_recall
                    * 100,
            }
        )


    classification_groups = (
        pd.DataFrame(
            classification_group_rows
        )
    )


    # --------------------------------------------------------
    # MERGE
    # --------------------------------------------------------

    subgroup_summary = (
        regression_groups
        .merge(
            classification_groups,
            on="Subgroup",
            how="outer",
        )
    )


    return (
        subgroup_summary,
        reg,
        cls,
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
# RESPONSIVE SIDE-BY-SIDE MODEL PANELS
# ============================================================

# Scope layout rules to this comparison only. Flex wrapping responds to
# available content width, including when the sidebar is open.
st.html("""
<style>
.st-key-model-comparison-panels [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-linear-model-panel) {
    flex-wrap: wrap;
    align-items: flex-start;
    gap: 1.5rem;
}
.st-key-model-comparison-panels [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-linear-model-panel) > [data-testid="stColumn"] {
    flex: 1 1 420px !important;
    min-width: min(100%, 420px) !important;
    width: auto !important;
}
.st-key-linear-model-panel,
.st-key-logistic-model-panel {
    min-width: 0;
}
.st-key-model-comparison-panels [data-testid="stMetricLabel"] {
    white-space: normal;
    overflow-wrap: anywhere;
}
.st-key-model-comparison-panels [data-testid="stMetricValue"] {
    font-size: clamp(1.35rem, 2.2vw, 2rem);
}

/* Header alignment is based on available page width, not viewport width. */
.st-key-model-comparison-panels { container-type: inline-size; }
@container (min-width: 880px) {
    .st-key-linear-model-header,
    .st-key-logistic-model-header { min-height: 280px; }
}
.st-key-regression-chart-heading,
.st-key-classification-chart-heading { min-height: 64px; }
/* Compact supporting visuals and interpretations wrap on narrow panels. */
.st-key-regression-variance-row [data-testid="stHorizontalBlock"],
.st-key-regression-notes-row [data-testid="stHorizontalBlock"] {
    flex-wrap: wrap;
    gap: 0.75rem;
}
.st-key-regression-variance-row [data-testid="stColumn"] {
    flex: 1 1 220px !important;
    min-width: min(100%, 220px) !important;
    width: auto !important;
}
.st-key-regression-notes-row [data-testid="stColumn"] {
    flex: 1 1 150px !important;
    min-width: min(100%, 150px) !important;
    width: auto !important;
}
.st-key-regression-variance-row h4 { min-height: 3.5rem; }
</style>
""")

with st.container(key="model-comparison-panels"):
    linear_column, logistic_column = st.columns(2, gap="large")

    with linear_column:
        with st.container(key="linear-model-panel", border=True):
            # ============================================================
            # LINEAR REGRESSION
            # ============================================================

            with st.container(key="linear-model-header"):
                section_title(
                    "trending",
                    "Linear Regression",
                )


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

            reg1 = st.container()
            with st.container(key="regression-variance-row"):
                reg2, reg3 = st.columns(2, gap="small")


            with reg1:

                with st.container(key="regression-chart-heading"):
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
                    height=300,

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
                        "displayModeBar":
                            False
                    },
                )


            with reg2:

                st.markdown(
                    "#### Variance Explained"
                )


                fig_r2 = go.Figure(
                    go.Indicator(
                        mode="gauge+number",

                        value=44.4,

                        number={
                            "suffix":
                                "%"
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
                                "color":
                                    accent
                            },

                            "bgcolor":
                                surface,

                            "bordercolor":
                                chart_border,
                        },
                    )
                )


                fig_r2.update_layout(
                    height=250,

                    margin=dict(
                        l=25,
                        r=25,
                        t=35,
                        b=20,
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

                    height=250,

                    margin=dict(
                        l=10,
                        r=10,
                        t=10,
                        b=10,
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
                    fig_variance,
                    width="stretch",
                    theme=None,
                    config={
                        "displayModeBar":
                            False
                    },
                )


            # ============================================================
            # REGRESSION INTERPRETATION
            # ============================================================

            with st.container(key="regression-notes-row"):
                reg_note1, reg_note2, reg_note3 = st.columns(3, gap="small")


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



    with logistic_column:
        with st.container(key="logistic-model-panel", border=True):
            # ============================================================
            # BALANCED LOGISTIC REGRESSION
            # ============================================================

            with st.container(key="logistic-model-header"):
                section_title(
                    "target",
                    "Balanced Logistic Regression",
                )


                classification_metric_cols = [
                    *st.columns(3),
                    *st.columns(3),
                ]


                classification_metric_data = [
                    (
                        "Accuracy",
                        "72.5%",
                    ),
                    (
                        "Precision",
                        "90.6%",
                    ),
                    (
                        "Recall",
                        "71.9%",
                    ),
                    (
                        "F1",
                        "80.1%",
                    ),
                    (
                        "Balanced Acc.",
                        "73.2%",
                    ),
                    (
                        "AUC",
                        "0.814",
                    ),
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

            cls_left, cls_right = (st.container() for _ in range(2))


            with cls_left:

                with st.container(key="classification-chart-heading"):
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


                fig_cls_metrics = px.bar(
                    classification_metric_df,
                    x="Score",
                    y="Metric",
                    orientation="h",
                    text="Score",
                )


                fig_cls_metrics.update_traces(
                    marker_color=accent,
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                )


                fig_cls_metrics.update_layout(
                    height=300,

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
                        "displayModeBar":
                            False
                    },
                )


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


                fig_radar = go.Figure()


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

                    height=420,

                    margin=dict(
                        l=45,
                        r=45,
                        t=25,
                        b=25,
                    ),

                    polar=dict(
                        bgcolor=(
                            chart_background
                        ),

                        radialaxis=dict(
                            visible=True,

                            range=[
                                0,
                                100,
                            ],

                            ticksuffix="%",

                            gridcolor=(
                                chart_border
                            ),
                        ),

                        angularaxis=dict(
                            gridcolor=(
                                chart_border
                            ),
                        ),
                    ),

                    showlegend=False,

                    paper_bgcolor=(
                        chart_background
                    ),

                    font=dict(
                        color=chart_text
                    ),
                )


                st.plotly_chart(
                    fig_radar,
                    width="stretch",
                    theme=None,
                    config={
                        "displayModeBar":
                            False
                    },
                )



# ============================================================
# CLASSIFICATION MODEL COMPARISON
# ============================================================

section_title(
    "scale",
    "Which Classifier Should Be Used?",
)


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


fig_compare = px.bar(
    classification_long,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text="Score",
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
        "displayModeBar":
            False
    },
)


# ============================================================
# ACCURACY GAP
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


fig_gap = px.bar(
    gap_df,
    x="Accuracy Gap",
    y="Model",
    orientation="h",
    text="Accuracy Gap",
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
        "displayModeBar":
            False
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


fig_rank = px.bar(
    balanced_ranking_df,
    x="Balanced Accuracy",
    y="Model",
    orientation="h",
    text="Balanced Accuracy",
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
        "displayModeBar":
            False
    },
)


# ============================================================
# CLASSIFIER DECISION
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
# FINAL MODEL SUMMARY
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
# FAIRNESS & SUBGROUP ANALYSIS
# ============================================================

section_title(
    "scale",
    "Fairness & Subgroup Analysis",
)


st.caption(
    "Audit model performance across student groups. "
    "These subgroup variables are used only for evaluation — "
    "they are not inputs to either prediction model."
)


# ------------------------------------------------------------
# AVAILABLE GROUPS
# ------------------------------------------------------------

subgroup_options = {
    "Gender":
        "gender",

    "Education Level":
        "education_level",

    "School Type":
        "school_type",

    "Family Income":
        "family_income",

    "Urban / Rural":
        "urban_rural",
}


available_subgroups = {
    label: column
    for label, column
    in subgroup_options.items()
    if column in df.columns
}


selected_subgroup_label = (
    st.selectbox(
        "Compare performance by",
        list(
            available_subgroups.keys()
        ),
        index=0,
    )
)


selected_subgroup_column = (
    available_subgroups[
        selected_subgroup_label
    ]
)


(
    subgroup_summary,
    regression_subgroup_rows,
    classification_subgroup_rows,
) = build_subgroup_metrics(
    selected_subgroup_column
)


# ============================================================
# AUDIT SUMMARY VALUES
# ============================================================

valid_ba = (
    subgroup_summary[
        "Balanced Accuracy"
    ]
    .dropna()
)


valid_mae = (
    subgroup_summary[
        "MAE"
    ]
    .dropna()
)


number_groups = (
    subgroup_summary[
        "Subgroup"
    ]
    .nunique()
)


largest_group = int(
    subgroup_summary[
        "Classification N"
    ]
    .fillna(0)
    .max()
)


ba_gap = (
    valid_ba.max()
    - valid_ba.min()
    if len(
        valid_ba
    ) > 1
    else 0
)


mae_gap = (
    valid_mae.max()
    - valid_mae.min()
    if len(
        valid_mae
    ) > 1
    else 0
)


audit1, audit2, audit3, audit4 = (
    st.columns(4)
)


audit1.metric(
    "Groups",
    number_groups,
)


audit2.metric(
    "Largest Test Group",
    f"{largest_group:,}",
)


audit3.metric(
    "Balanced Acc. Gap",
    f"{ba_gap:.1f} pts",
)


audit4.metric(
    "MAE Gap",
    f"{mae_gap:.2f}",
)


# ============================================================
# FIRST ROW OF AUDIT VISUALS
# ============================================================

audit_left, audit_right = (
    st.columns(2)
)


# ------------------------------------------------------------
# GROUP SIZE
# ------------------------------------------------------------

with audit_left:

    section_title(
        "database",
        "Test-Set Group Size",
    )


    size_df = (
        subgroup_summary[
            [
                "Subgroup",
                "Classification N",
            ]
        ]
        .sort_values(
            "Classification N",
            ascending=True,
        )
    )


    fig_size = px.bar(
        size_df,
        x="Classification N",
        y="Subgroup",
        orientation="h",
        text="Classification N",
    )


    fig_size.update_traces(
        marker_color=accent,
        textposition="outside",
        texttemplate="%{text:,}",
    )


    fig_size.update_layout(
        height=340,

        margin=dict(
            l=20,
            r=65,
            t=10,
            b=20,
        ),

        xaxis_title=(
            "Held-Out Test Students"
        ),

        yaxis_title="",

        showlegend=False,
    )


    style_chart(
        fig_size
    )


    st.plotly_chart(
        fig_size,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ------------------------------------------------------------
# REGRESSION MAE
# ------------------------------------------------------------

with audit_right:

    section_title(
        "trending",
        "Regression Error by Group",
    )


    mae_df = (
        subgroup_summary[
            [
                "Subgroup",
                "MAE",
            ]
        ]
        .dropna()
        .sort_values(
            "MAE",
            ascending=True,
        )
    )


    fig_group_mae = px.bar(
        mae_df,
        x="MAE",
        y="Subgroup",
        orientation="h",
        text="MAE",
    )


    fig_group_mae.update_traces(
        marker_color=negative_color,

        texttemplate=(
            "%{text:.2f}"
        ),

        textposition="outside",
    )


    fig_group_mae.update_layout(
        height=340,

        margin=dict(
            l=20,
            r=65,
            t=10,
            b=20,
        ),

        xaxis_title=(
            "Mean Absolute Error"
        ),

        yaxis_title="",

        showlegend=False,
    )


    style_chart(
        fig_group_mae
    )


    st.plotly_chart(
        fig_group_mae,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ============================================================
# SECOND ROW
# ============================================================

audit_cls_left, audit_cls_right = (
    st.columns(2)
)


# ------------------------------------------------------------
# BALANCED ACCURACY
# ------------------------------------------------------------

with audit_cls_left:

    section_title(
        "target",
        "Balanced Accuracy by Group",
    )


    ba_df = (
        subgroup_summary[
            [
                "Subgroup",
                "Balanced Accuracy",
            ]
        ]
        .dropna()
        .sort_values(
            "Balanced Accuracy",
            ascending=True,
        )
    )


    fig_group_ba = px.bar(
        ba_df,
        x="Balanced Accuracy",
        y="Subgroup",
        orientation="h",
        text="Balanced Accuracy",
    )


    fig_group_ba.update_traces(
        marker_color=positive_color,

        texttemplate=(
            "%{text:.1f}%"
        ),

        textposition="outside",
    )


    fig_group_ba.update_layout(
        height=350,

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
        fig_group_ba
    )


    st.plotly_chart(
        fig_group_ba,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ------------------------------------------------------------
# PASS VS FAIL RECALL
# ------------------------------------------------------------

with audit_cls_right:

    section_title(
        "chart",
        "Pass vs. Fail Recall",
    )


    recall_df = (
        subgroup_summary[
            [
                "Subgroup",
                "Pass Recall",
                "Fail Recall",
            ]
        ]
        .dropna()
    )


    fig_recall = go.Figure()


    fig_recall.add_trace(
        go.Bar(
            x=recall_df[
                "Subgroup"
            ],

            y=recall_df[
                "Pass Recall"
            ],

            name="Pass Recall",

            marker_color=(
                positive_color
            ),

            text=recall_df[
                "Pass Recall"
            ],

            texttemplate=(
                "%{text:.1f}%"
            ),

            textposition=(
                "outside"
            ),
        )
    )


    fig_recall.add_trace(
        go.Bar(
            x=recall_df[
                "Subgroup"
            ],

            y=recall_df[
                "Fail Recall"
            ],

            name="Fail Recall",

            marker_color=(
                negative_color
            ),

            text=recall_df[
                "Fail Recall"
            ],

            texttemplate=(
                "%{text:.1f}%"
            ),

            textposition=(
                "outside"
            ),
        )
    )


    fig_recall.update_layout(
        barmode="group",

        height=350,

        margin=dict(
            l=20,
            r=20,
            t=10,
            b=45,
        ),

        xaxis_title="",

        yaxis=dict(
            title="Recall (%)",

            range=[
                0,
                100,
            ],

            ticksuffix="%",
        ),

        legend=dict(
            title="",
            orientation="h",
            y=1.08,
            x=1,
            xanchor="right",
        ),
    )


    style_chart(
        fig_recall
    )


    st.plotly_chart(
        fig_recall,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

section_title(
    "chart",
    "Prediction Error Distribution",
)


fig_error_distribution = px.box(
    regression_subgroup_rows,
    x="Subgroup",
    y="Absolute Error",
    points=False,
)


fig_error_distribution.update_traces(
    line_color=accent,
    marker_color=accent,
)


fig_error_distribution.update_layout(
    height=380,

    margin=dict(
        l=20,
        r=20,
        t=10,
        b=45,
    ),

    xaxis_title="",

    yaxis_title=(
        "Absolute Prediction Error"
    ),

    showlegend=False,
)


style_chart(
    fig_error_distribution
)


st.plotly_chart(
    fig_error_distribution,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)


st.caption(
    "The box plots show the distribution of absolute regression error "
    "within each subgroup, not just the average error."
)


# ============================================================
# FAIRNESS MAP
# ============================================================

section_title(
    "sparkles",
    "Subgroup Performance Map",
)


fairness_map_df = (
    subgroup_summary[
        [
            "Subgroup",
            "MAE",
            "Balanced Accuracy",
            "Classification N",
        ]
    ]
    .dropna()
)


fig_fairness_map = px.scatter(
    fairness_map_df,

    x="MAE",

    y="Balanced Accuracy",

    size="Classification N",

    text="Subgroup",

    hover_name="Subgroup",

    size_max=48,
)


fig_fairness_map.update_traces(
    marker=dict(
        color=accent,
        opacity=0.78,

        line=dict(
            color=chart_border,
            width=1,
        ),
    ),

    textposition="top center",
)


fig_fairness_map.update_layout(
    height=430,

    margin=dict(
        l=25,
        r=25,
        t=10,
        b=25,
    ),

    xaxis_title=(
        "Regression MAE → lower is better"
    ),

    yaxis=dict(
        title=(
            "Balanced Accuracy (%) → higher is better"
        ),

        range=[
            0,
            100,
        ],

        ticksuffix="%",
    ),

    showlegend=False,
)


style_chart(
    fig_fairness_map
)


st.plotly_chart(
    fig_fairness_map,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)


st.caption(
    "Bubble size represents subgroup test-set size. "
    "The most favorable area is toward the upper-left: "
    "lower regression error and higher classification Balanced Accuracy."
)


# ============================================================
# SUBGROUP SUMMARY TABLE
# ============================================================

section_title(
    "database",
    "Subgroup Audit Summary",
)


audit_display = (
    subgroup_summary[
        [
            "Subgroup",
            "Regression N",
            "MAE",
            "Balanced Accuracy",
            "Pass Recall",
            "Fail Recall",
        ]
    ]
    .copy()
)


audit_display[
    "Regression N"
] = (
    audit_display[
        "Regression N"
    ]
    .map(
        lambda value:
            f"{int(value):,}"
        if pd.notna(
            value
        )
        else "—"
    )
)


audit_display[
    "MAE"
] = (
    audit_display[
        "MAE"
    ]
    .map(
        lambda value:
            f"{value:.2f}"
        if pd.notna(
            value
        )
        else "—"
    )
)


for metric in [
    "Balanced Accuracy",
    "Pass Recall",
    "Fail Recall",
]:

    audit_display[
        metric
    ] = (
        audit_display[
            metric
        ]
        .map(
            lambda value:
                f"{value:.1f}%"
            if pd.notna(
                value
            )
            else "—"
        )
    )


show_theme_table(
    audit_display
)


st.info(
    "This section is a model audit, not a causal or discrimination claim. "
    "Gender, education level, school type, family income, and urban/rural "
    "status are not part of the six prediction inputs. Differences across "
    "groups can also reflect subgroup size, data composition, or other "
    "variables not represented in the model."
)


# ============================================================
# MODEL LIMITATIONS
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
    "The subgroup audit adds another layer of evaluation by checking "
    "whether held-out performance changes across different student groups."
)


# ============================================================
# FOOTER
# ============================================================

show_footer()