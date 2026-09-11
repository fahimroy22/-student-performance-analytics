# ============================================================
# CLASSIFICATION THRESHOLD EXPLORER
# Final polished version
#
# Uses:
# - Real held-out 20% test set
# - Saved Balanced Logistic Regression pipeline
# - Real predicted probabilities
#
# No fabricated model results are used.
# ============================================================

import html

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
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
)

from core.model_loader import load_logistic_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Classification Threshold Explorer",
    page_icon="🎯",
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


def style_chart(fig):
    """
    Apply the current application theme to Plotly charts.
    """

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
# THEME-AWARE HTML TABLE
# ============================================================

def render_threshold_table(dataframe):
    """
    Render a fully theme-aware comparison table.
    """

    header_html = ""

    for column in dataframe.columns:
        header_html += (
            f"<th>{html.escape(str(column))}</th>"
        )

    body_html = ""

    for _, row in dataframe.iterrows():

        body_html += "<tr>"

        for value in row:
            body_html += (
                f"<td>{html.escape(str(value))}</td>"
            )

        body_html += "</tr>"

    table_html = f"""
    <style>

    .threshold-table-wrap {{
        width: 100%;
        overflow-x: auto;
        margin-top: 6px;
        margin-bottom: 8px;
    }}

    .threshold-table {{
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;

        background: {surface};

        border: 1px solid {border};
        border-radius: 10px;

        overflow: hidden;

        font-size: 13px;

        color: {text_color};
    }}

    .threshold-table th {{
        background: {surface};

        color: {muted};

        text-align: left;

        font-weight: 600;

        padding: 11px 12px;

        border-bottom: 1px solid {border};

        white-space: nowrap;
    }}

    .threshold-table td {{
        background: {surface};

        color: {text_color};

        padding: 11px 12px;

        border-bottom: 1px solid {border};

        white-space: nowrap;
    }}

    .threshold-table tbody tr:last-child td {{
        border-bottom: none;
    }}

    .threshold-table tbody tr {{
        transition: background 0.15s ease;
    }}

    .threshold-table tbody tr:hover td {{
        background: rgba(91, 141, 184, 0.08);
    }}

    .threshold-table th:not(:last-child),
    .threshold-table td:not(:last-child) {{
        border-right: 1px solid {border};
    }}

    </style>

    <div class="threshold-table-wrap">

        <table class="threshold-table">

            <thead>
                <tr>
                    {header_html}
                </tr>
            </thead>

            <tbody>
                {body_html}
            </tbody>

        </table>

    </div>
    """

    st.html(table_html)


# ============================================================
# PATHS + FEATURES
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

FEATURES = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed",
]


# ============================================================
# LOAD TEST DATA
# ============================================================

@st.cache_data
def load_test_data():

    df = pd.read_csv(
        DATA_PATH
    )

    X = df[
        FEATURES
    ]

    y = df[
        "pass_status"
    ]

    (
        _,
        X_test,
        _,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return (
        X_test.reset_index(
            drop=True
        ),
        y_test.reset_index(
            drop=True
        ),
    )


# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_logistic_model():

    return load_logistic_model()


# ============================================================
# PREDICTED PROBABILITIES
# ============================================================

@st.cache_data
def calculate_probabilities():

    X_test, y_test = (
        load_test_data()
    )

    model = (
        get_logistic_model()
    )

    model_classes = list(
        model
        .named_steps["model"]
        .classes_
    )

    pass_index = (
        model_classes
        .index("Pass")
    )

    pass_probabilities = (
        model
        .predict_proba(
            X_test
        )[:, pass_index]
    )

    return (
        y_test,
        pass_probabilities,
    )


y_test, pass_probabilities = (
    calculate_probabilities()
)

y_true_binary = (
    y_test
    .eq("Pass")
    .astype(int)
    .to_numpy()
)


# ============================================================
# METRIC CALCULATION
# ============================================================

def metrics_at_threshold(
    threshold_value,
):

    predictions = (
        pass_probabilities
        >= threshold_value
    ).astype(int)

    accuracy = (
        accuracy_score(
            y_true_binary,
            predictions,
        )
    )

    precision = (
        precision_score(
            y_true_binary,
            predictions,
            zero_division=0,
        )
    )

    recall = (
        recall_score(
            y_true_binary,
            predictions,
            zero_division=0,
        )
    )

    f1 = (
        f1_score(
            y_true_binary,
            predictions,
            zero_division=0,
        )
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true_binary,
            predictions,
        )
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true_binary,
            predictions,
            labels=[0, 1],
        )
        .ravel()
    )

    return {
        "accuracy":
            accuracy,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "balanced_accuracy":
            balanced_accuracy,

        "tn":
            tn,

        "fp":
            fp,

        "fn":
            fn,

        "tp":
            tp,
    }


# ============================================================
# BEST BALANCED THRESHOLD
# ============================================================

@st.cache_data
def find_best_balanced_threshold():

    threshold_values = (
        np.arange(
            0.10,
            0.901,
            0.005,
        )
    )

    rows = []

    for threshold_value in threshold_values:

        metric = (
            metrics_at_threshold(
                threshold_value
            )
        )

        rows.append(
            {
                "threshold":
                    threshold_value,

                "balanced_accuracy":
                    metric[
                        "balanced_accuracy"
                    ],

                "f1":
                    metric["f1"],

                "precision":
                    metric["precision"],

                "recall":
                    metric["recall"],

                "accuracy":
                    metric["accuracy"],
            }
        )

    result_df = (
        pd.DataFrame(
            rows
        )
    )

    best_row = (
        result_df
        .sort_values(
            [
                "balanced_accuracy",
                "f1",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return (
        result_df,
        float(
            best_row[
                "threshold"
            ]
        ),
    )


(
    threshold_scan_df,
    best_balanced_threshold,
) = (
    find_best_balanced_threshold()
)


# ============================================================
# HEADER
# ============================================================

page_title(
    "scale",
    "Classification Threshold Explorer",
    (
        "Explore how changing the decision threshold affects "
        "classification performance using the actual held-out test set."
    ),
)

st.info(
    "This page uses the real 20% held-out test set and the saved "
    "Balanced Logistic Regression model. Changing the threshold "
    "does not retrain the model."
)


# ============================================================
# THRESHOLD CONTROL
# ============================================================

section_title(
    "sliders",
    "Decision Threshold",
)

threshold = st.slider(
    "Pass classification threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.01,
    help=(
        "A student is classified as Pass when the predicted "
        "Pass probability is greater than or equal to this threshold."
    ),
)

st.caption(
    f"Current rule: predict PASS when Pass Probability ≥ "
    f"{threshold:.0%}."
)


# ============================================================
# METRICS
# ============================================================

current = (
    metrics_at_threshold(
        threshold
    )
)

baseline = (
    metrics_at_threshold(
        0.50
    )
)

best_balanced = (
    metrics_at_threshold(
        best_balanced_threshold
    )
)


# ============================================================
# BEST BALANCED THRESHOLD
# ============================================================

section_title(
    "target",
    "Best Balanced Threshold",
)

r1, r2, r3, r4 = (
    st.columns(4)
)

r1.metric(
    "Best Threshold",
    f"{best_balanced_threshold:.2f}",
)

r2.metric(
    "Balanced Accuracy",
    f"{best_balanced['balanced_accuracy']:.1%}",
)

r3.metric(
    "F1 Score",
    f"{best_balanced['f1']:.1%}",
)

r4.metric(
    "Difference from 0.50",
    f"{best_balanced_threshold - 0.50:+.2f}",
)

st.caption(
    "This threshold maximizes Balanced Accuracy on the held-out "
    "test set. It is therefore the best threshold specifically "
    "for the Balanced Accuracy objective, rather than a universal "
    "best threshold for every possible classification goal."
)


# ============================================================
# CURRENT PERFORMANCE
# ============================================================

section_title(
    "target",
    "Performance at This Threshold",
)

m1, m2, m3, m4, m5 = (
    st.columns(5)
)

m1.metric(
    "Accuracy",
    f"{current['accuracy']:.1%}",
    delta=(
        f"{current['accuracy'] - baseline['accuracy']:+.1%}"
        " vs 0.50"
    ),
)

m2.metric(
    "Precision",
    f"{current['precision']:.1%}",
    delta=(
        f"{current['precision'] - baseline['precision']:+.1%}"
        " vs 0.50"
    ),
)

m3.metric(
    "Recall",
    f"{current['recall']:.1%}",
    delta=(
        f"{current['recall'] - baseline['recall']:+.1%}"
        " vs 0.50"
    ),
)

m4.metric(
    "F1 Score",
    f"{current['f1']:.1%}",
    delta=(
        f"{current['f1'] - baseline['f1']:+.1%}"
        " vs 0.50"
    ),
)

m5.metric(
    "Balanced Accuracy",
    f"{current['balanced_accuracy']:.1%}",
    delta=(
        f"{current['balanced_accuracy'] - baseline['balanced_accuracy']:+.1%}"
        " vs 0.50"
    ),
)

st.caption(
    "Metric deltas show how the current threshold compares "
    "with the standard 0.50 classification threshold."
)


# ============================================================
# CLASSIFICATION OUTCOMES
# ============================================================

section_title(
    "database",
    "Classification Outcomes",
)

c1, c2, c3, c4 = (
    st.columns(4)
)

c1.metric(
    "True Positive",
    f"{current['tp']:,}",
)

c2.metric(
    "False Positive",
    f"{current['fp']:,}",
)

c3.metric(
    "True Negative",
    f"{current['tn']:,}",
)

c4.metric(
    "False Negative",
    f"{current['fn']:,}",
)


# ============================================================
# ERROR INTERPRETATION
# ============================================================

error_left, error_right = (
    st.columns(2)
)


with error_left:

    with st.container(
        border=True
    ):

        st.markdown(
            "#### False Positive"
        )

        st.write(
            "**Predicted Pass, but actually Fail.**"
        )

        st.caption(
            "The model estimated enough probability to classify "
            "the student as Pass, but the actual outcome was Fail."
        )


with error_right:

    with st.container(
        border=True
    ):

        st.markdown(
            "#### False Negative"
        )

        st.write(
            "**Predicted Fail, but actually Pass.**"
        )

        st.caption(
            "The predicted probability did not reach the selected "
            "threshold even though the student actually passed."
        )


# ============================================================
# THRESHOLD TRADE-OFF
# ============================================================

section_title(
    "chart",
    "Threshold Trade-Off",
)

curve_thresholds = (
    np.arange(
        0.10,
        0.901,
        0.02,
    )
)

curve_rows = []

for threshold_value in curve_thresholds:

    metric = (
        metrics_at_threshold(
            float(
                threshold_value
            )
        )
    )

    curve_rows.append(
        {
            "Threshold":
                threshold_value,

            "Precision":
                metric["precision"],

            "Recall":
                metric["recall"],

            "F1":
                metric["f1"],

            "Balanced Accuracy":
                metric[
                    "balanced_accuracy"
                ],
        }
    )


curve_df = (
    pd.DataFrame(
        curve_rows
    )
)

fig_tradeoff = (
    go.Figure()
)

for metric_name in [
    "Precision",
    "Recall",
    "F1",
    "Balanced Accuracy",
]:

    fig_tradeoff.add_trace(
        go.Scatter(
            x=curve_df[
                "Threshold"
            ],

            y=curve_df[
                metric_name
            ],

            mode="lines",

            name=metric_name,

            hovertemplate=(
                "Threshold: %{x:.2f}"
                "<br>"
                + metric_name
                + ": %{y:.1%}"
                "<extra></extra>"
            ),
        )
    )


# ------------------------------------------------------------
# CURRENT THRESHOLD LINE
# ------------------------------------------------------------

fig_tradeoff.add_vline(
    x=threshold,
    line_dash="dash",
    line_color=accent,
)


# ------------------------------------------------------------
# BEST BALANCED THRESHOLD LINE
# ------------------------------------------------------------

fig_tradeoff.add_vline(
    x=best_balanced_threshold,
    line_dash="dot",
    line_color=muted,
)


# ------------------------------------------------------------
# CURRENT THRESHOLD ANNOTATION
# Inside plot, upper region
# ------------------------------------------------------------

fig_tradeoff.add_annotation(
    x=threshold,
    y=0.94,
    xref="x",
    yref="paper",

    text=(
        f"Current {threshold:.2f}"
    ),

    showarrow=True,

    arrowhead=0,

    arrowcolor=accent,

    ax=-50,
    ay=-10,

    bgcolor=background,

    bordercolor=accent,

    borderwidth=1,

    borderpad=4,

    font=dict(
        color=accent,
        size=11,
    ),
)


# ------------------------------------------------------------
# BEST BALANCED ANNOTATION
# Inside plot, lower region
# ------------------------------------------------------------

fig_tradeoff.add_annotation(
    x=best_balanced_threshold,
    y=0.13,
    xref="x",
    yref="paper",

    text=(
        f"Best balanced "
        f"{best_balanced_threshold:.2f}"
    ),

    showarrow=True,

    arrowhead=0,

    arrowcolor=muted,

    ax=70,
    ay=0,

    bgcolor=background,

    bordercolor=muted,

    borderwidth=1,

    borderpad=4,

    font=dict(
        color=muted,
        size=11,
    ),
)


fig_tradeoff.update_layout(
    height=430,

    margin=dict(
        l=20,
        r=30,
        t=35,
        b=45,
    ),

    xaxis_title=(
        "Classification Threshold"
    ),

    yaxis_title=(
        "Metric Value"
    ),

    yaxis_tickformat=".0%",

    yaxis_range=[
        0,
        1,
    ],

    legend_title=(
        "Metric"
    ),
)

style_chart(
    fig_tradeoff
)

st.plotly_chart(
    fig_tradeoff,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

section_title(
    "chart",
    "Predicted Probability Distribution",
)

st.caption(
    "This chart shows how predicted Pass probabilities are "
    "distributed for students who actually passed and failed."
)

probability_df = (
    pd.DataFrame(
        {
            "Pass Probability":
                pass_probabilities,

            "Actual Outcome":
                y_test,
        }
    )
)

fig_distribution = (
    px.histogram(
        probability_df,

        x="Pass Probability",

        color="Actual Outcome",

        nbins=40,

        barmode="overlay",

        opacity=0.65,

        histnorm=(
            "probability density"
        ),
    )
)


# ------------------------------------------------------------
# THRESHOLD LINES
# ------------------------------------------------------------

fig_distribution.add_vline(
    x=threshold,
    line_dash="dash",
    line_color=accent,
)

fig_distribution.add_vline(
    x=best_balanced_threshold,
    line_dash="dot",
    line_color=muted,
)


# ------------------------------------------------------------
# CURRENT THRESHOLD ANNOTATION
# Inside chart, upper region
# ------------------------------------------------------------

fig_distribution.add_annotation(
    x=threshold,
    y=0.94,
    xref="x",
    yref="paper",

    text=(
        f"Current {threshold:.2f}"
    ),

    showarrow=True,

    arrowhead=0,

    arrowcolor=accent,

    ax=-55,
    ay=-10,

    bgcolor=background,

    bordercolor=accent,

    borderwidth=1,

    borderpad=4,

    font=dict(
        color=accent,
        size=11,
    ),
)


# ------------------------------------------------------------
# BEST BALANCED ANNOTATION
# Inside chart, lower region
# ------------------------------------------------------------

fig_distribution.add_annotation(
    x=best_balanced_threshold,
    y=0.12,
    xref="x",
    yref="paper",

    text=(
        f"Best balanced "
        f"{best_balanced_threshold:.2f}"
    ),

    showarrow=True,

    arrowhead=0,

    arrowcolor=muted,

    ax=72,
    ay=0,

    bgcolor=background,

    bordercolor=muted,

    borderwidth=1,

    borderpad=4,

    font=dict(
        color=muted,
        size=11,
    ),
)


fig_distribution.update_layout(
    height=420,

    margin=dict(
        l=20,
        r=30,
        t=35,
        b=50,
    ),

    xaxis_title=(
        "Predicted Pass Probability"
    ),

    yaxis_title=(
        "Probability Density"
    ),

    legend_title=(
        "Actual Outcome"
    ),
)

fig_distribution.update_xaxes(
    tickformat=".0%"
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
    "Where the Pass and Fail distributions overlap, the model "
    "has greater difficulty separating the two classes. "
    "The threshold determines where the final decision boundary is placed."
)


# ============================================================
# PRECISION–RECALL CURVE
# ============================================================

section_title(
    "chart",
    "Precision–Recall Curve",
)

(
    precision_values,
    recall_values,
    pr_thresholds,
) = (
    precision_recall_curve(
        y_true_binary,
        pass_probabilities,
    )
)

average_precision = (
    average_precision_score(
        y_true_binary,
        pass_probabilities,
    )
)

fig_pr = (
    go.Figure()
)

fig_pr.add_trace(
    go.Scatter(
        x=recall_values,
        y=precision_values,

        mode="lines",

        name=(
            "Precision–Recall"
        ),

        hovertemplate=(
            "Recall: %{x:.1%}"
            "<br>"
            "Precision: %{y:.1%}"
            "<extra></extra>"
        ),
    )
)

fig_pr.add_trace(
    go.Scatter(
        x=[
            current["recall"]
        ],

        y=[
            current["precision"]
        ],

        mode="markers",

        name=(
            f"Threshold "
            f"{threshold:.2f}"
        ),

        marker=dict(
            size=12,
        ),

        hovertemplate=(
            f"Threshold: "
            f"{threshold:.2f}"
            "<br>"
            "Recall: %{x:.1%}"
            "<br>"
            "Precision: %{y:.1%}"
            "<extra></extra>"
        ),
    )
)


fig_pr.update_layout(
    height=400,

    margin=dict(
        l=20,
        r=20,
        t=20,
        b=35,
    ),

    xaxis_title=(
        "Recall"
    ),

    yaxis_title=(
        "Precision"
    ),

    xaxis_tickformat=".0%",

    yaxis_tickformat=".0%",

    xaxis_range=[
        0,
        1,
    ],

    yaxis_range=[
        0,
        1,
    ],
)

style_chart(
    fig_pr
)

st.plotly_chart(
    fig_pr,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)

st.caption(
    f"Average Precision = {average_precision:.3f}. "
    "The highlighted point represents the precision and recall "
    "produced by the currently selected threshold."
)


# ============================================================
# THRESHOLD COMPARISON
# ============================================================

section_title(
    "scale",
    "Threshold Comparison",
)

comparison_thresholds = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
]

comparison_rows = []

for threshold_value in comparison_thresholds:

    metric = (
        metrics_at_threshold(
            threshold_value
        )
    )

    comparison_rows.append(
        {
            "Threshold":
                f"{threshold_value:.2f}",

            "Accuracy":
                f"{metric['accuracy']:.1%}",

            "Precision":
                f"{metric['precision']:.1%}",

            "Recall":
                f"{metric['recall']:.1%}",

            "F1":
                f"{metric['f1']:.1%}",

            "Balanced Accuracy":
                f"{metric['balanced_accuracy']:.1%}",

            "False Positives":
                f"{metric['fp']:,}",

            "False Negatives":
                f"{metric['fn']:,}",
        }
    )

comparison_df = (
    pd.DataFrame(
        comparison_rows
    )
)


# Custom dark/light theme-aware table

render_threshold_table(
    comparison_df
)

st.caption(
    "Lower thresholds generally increase Recall but can create "
    "more False Positives. Higher thresholds usually increase "
    "Precision but may create more False Negatives."
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

section_title(
    "chart",
    "Confusion Matrix",
)

matrix = [
    [
        current["tn"],
        current["fp"],
    ],
    [
        current["fn"],
        current["tp"],
    ],
]

matrix_text = [
    [
        f"{current['tn']:,}",
        f"{current['fp']:,}",
    ],
    [
        f"{current['fn']:,}",
        f"{current['tp']:,}",
    ],
]

fig_cm = (
    go.Figure(
        data=go.Heatmap(
            z=matrix,

            x=[
                "Predicted Fail",
                "Predicted Pass",
            ],

            y=[
                "Actual Fail",
                "Actual Pass",
            ],

            text=matrix_text,

            texttemplate=(
                "%{text}"
            ),

            hovertemplate=(
                "%{y}"
                "<br>"
                "%{x}"
                "<br>"
                "Students: %{z:,}"
                "<extra></extra>"
            ),

            colorscale=(
                "Blues"
            ),

            showscale=False,
        )
    )
)

fig_cm.update_layout(
    height=305,

    margin=dict(
        l=20,
        r=20,
        t=5,
        b=30,
    ),

    xaxis_title=(
        "Predicted Class"
    ),

    yaxis_title=(
        "Actual Class"
    ),
)

style_chart(
    fig_cm
)

st.plotly_chart(
    fig_cm,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)


# ============================================================
# INTERPRETATION
# ============================================================

section_title(
    "brain",
    "How to Interpret the Threshold",
)

left, middle, right = (
    st.columns(3)
)


with left:

    with st.container(
        border=True
    ):

        st.markdown(
            "#### Lower Threshold"
        )

        st.write(
            "The model requires less probability before predicting Pass."
        )

        st.caption(
            "Recall generally increases because fewer actual Pass "
            "students are missed, but False Positives may increase."
        )


with middle:

    with st.container(
        border=True
    ):

        st.markdown(
            "#### Balanced Threshold"
        )

        st.write(
            "A middle threshold tries to balance performance across both classes."
        )

        st.caption(
            "Balanced Accuracy is useful because the dataset contains "
            "more Pass students than Fail students."
        )


with right:

    with st.container(
        border=True
    ):

        st.markdown(
            "#### Higher Threshold"
        )

        st.write(
            "The model requires stronger evidence before predicting Pass."
        )

        st.caption(
            "Precision can increase, but more actual Pass students "
            "may become False Negatives."
        )


st.warning(
    "The classification threshold is a decision rule applied "
    "after Logistic Regression produces probabilities. "
    "Changing the threshold does not retrain the model and does "
    "not change the underlying predicted probabilities."
)


# ============================================================
# RETURN TO PREDICTOR
# ============================================================

back1, back2, back3 = (
    st.columns(
        [
            1,
            1.4,
            1,
        ]
    )
)

with back2:

    if st.button(
        "Open Student Predictor",
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