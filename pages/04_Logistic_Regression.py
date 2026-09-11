# ============================================================
# LOGISTIC REGRESSION PAGE
# Visual-first final version
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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

from core.model_loader import load_logistic_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Logistic Regression",
    page_icon="📊",
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

    .logistic-table-container {{
        overflow-x:auto;
        border:1px solid {chart_border};
        border-radius:10px;
        background:{surface};
        margin-bottom:14px;
    }}

    .logistic-table {{
        width:100%;
        border-collapse:collapse;
        font-size:13px;
        color:{chart_text};
        background:{surface};
    }}

    .logistic-table thead th {{
        text-align:left;
        padding:10px 11px;
        font-weight:600;
        color:{chart_muted};
        background:{surface};
        border-bottom:1px solid {chart_border};
        white-space:nowrap;
    }}

    .logistic-table tbody td {{
        padding:9px 11px;
        color:{chart_text};
        background:{surface};
        border-bottom:1px solid {chart_border};
        white-space:nowrap;
    }}

    .logistic-table tbody tr:last-child td {{
        border-bottom:none;
    }}

    .metric-note {{
        color:{chart_muted};
        font-size:11px;
        line-height:1.45;
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
        classes="logistic-table",
    )

    st.html(
        f"""
        <div class="logistic-table-container">
            {table_html}
        </div>
        """
    )


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

target = "pass_status"


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
def get_logistic_model():

    return load_logistic_model()


# ============================================================
# PREPARE DATA + EVALUATE MODELS
# ============================================================

@st.cache_data
def prepare_classification_results():

    X = df[
        features
    ].copy()

    y = (
        df[
            target
        ]
        .astype(str)
    )


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
        stratify=y,
    )


    # --------------------------------------------------------
    # SAVED BALANCED MODEL
    # --------------------------------------------------------

    balanced_model = (
        get_logistic_model()
    )


    y_pred_balanced = (
        balanced_model.predict(
            X_test
        )
    )


    probability_matrix = (
        balanced_model.predict_proba(
            X_test
        )
    )


    balanced_classes = list(
        balanced_model
        .named_steps["model"]
        .classes_
    )


    pass_index = (
        balanced_classes.index(
            "Pass"
        )
    )


    y_prob_balanced = (
        probability_matrix[
            :,
            pass_index
        ]
    )


    # --------------------------------------------------------
    # STANDARD LOGISTIC
    # --------------------------------------------------------

    standard_model = Pipeline(
        [
            (
                "preprocessing",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )


    standard_model.fit(
        X_train,
        y_train,
    )


    y_pred_standard = (
        standard_model.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # DUMMY
    # --------------------------------------------------------

    dummy_model = DummyClassifier(
        strategy="most_frequent"
    )


    dummy_model.fit(
        X_train,
        y_train,
    )


    y_pred_dummy = (
        dummy_model.predict(
            X_test
        )
    )


    # --------------------------------------------------------
    # METRIC FUNCTION
    # --------------------------------------------------------

    def get_metrics(
        y_true,
        y_pred,
    ):

        return {
            "Accuracy":
                accuracy_score(
                    y_true,
                    y_pred,
                ),

            "Precision":
                precision_score(
                    y_true,
                    y_pred,
                    pos_label="Pass",
                    zero_division=0,
                ),

            "Recall":
                recall_score(
                    y_true,
                    y_pred,
                    pos_label="Pass",
                    zero_division=0,
                ),

            "F1 Score":
                f1_score(
                    y_true,
                    y_pred,
                    pos_label="Pass",
                    zero_division=0,
                ),

            "Balanced Accuracy":
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                ),
        }


    balanced_metrics = (
        get_metrics(
            y_test,
            y_pred_balanced,
        )
    )


    standard_metrics = (
        get_metrics(
            y_test,
            y_pred_standard,
        )
    )


    dummy_metrics = (
        get_metrics(
            y_test,
            y_pred_dummy,
        )
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm_balanced = confusion_matrix(
        y_test,
        y_pred_balanced,
        labels=[
            "Fail",
            "Pass",
        ],
    )


    tn, fp, fn, tp = (
        cm_balanced.ravel()
    )


    fail_recall = (
        tn
        / (
            tn
            + fp
        )
        if (
            tn + fp
        ) > 0
        else 0
    )


    pass_recall = (
        tp
        / (
            tp
            + fn
        )
        if (
            tp + fn
        ) > 0
        else 0
    )


    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    y_test_binary = (
        y_test
        .eq("Pass")
        .astype(int)
    )


    auc = roc_auc_score(
        y_test_binary,
        y_prob_balanced,
    )


    fpr, tpr, _ = roc_curve(
        y_test_binary,
        y_prob_balanced,
    )


    # --------------------------------------------------------
    # PRECISION-RECALL
    # --------------------------------------------------------

    (
        pr_precision,
        pr_recall,
        _,
    ) = precision_recall_curve(
        y_test_binary,
        y_prob_balanced,
    )


    average_precision = (
        average_precision_score(
            y_test_binary,
            y_prob_balanced,
        )
    )


    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    comparison_df = pd.DataFrame(
        {
            "Model": [
                "Dummy Baseline",
                "Standard Logistic",
                "Balanced Logistic",
            ],

            "Accuracy": [
                dummy_metrics[
                    "Accuracy"
                ],
                standard_metrics[
                    "Accuracy"
                ],
                balanced_metrics[
                    "Accuracy"
                ],
            ],

            "Balanced Accuracy": [
                dummy_metrics[
                    "Balanced Accuracy"
                ],
                standard_metrics[
                    "Balanced Accuracy"
                ],
                balanced_metrics[
                    "Balanced Accuracy"
                ],
            ],
        }
    )


    comparison_df[
        "Accuracy Gap"
    ] = (
        comparison_df[
            "Accuracy"
        ]
        - comparison_df[
            "Balanced Accuracy"
        ]
    )


    # --------------------------------------------------------
    # PROBABILITY DATA
    # --------------------------------------------------------

    probability_df = pd.DataFrame(
        {
            "Pass Probability":
                y_prob_balanced,

            "Actual Outcome":
                y_test.values,
        }
    )


    return (
        balanced_metrics,
        standard_metrics,
        dummy_metrics,
        cm_balanced,
        fail_recall,
        pass_recall,
        auc,
        fpr,
        tpr,
        pr_precision,
        pr_recall,
        average_precision,
        comparison_df,
        probability_df,
        len(X_train),
        len(X_test),
    )


(
    balanced_metrics,
    standard_metrics,
    dummy_metrics,
    cm_balanced,
    fail_recall,
    pass_recall,
    auc,
    fpr,
    tpr,
    pr_precision,
    pr_recall,
    average_precision,
    comparison_df,
    probability_df,
    train_n,
    test_n,
) = prepare_classification_results()


# ============================================================
# HEADER
# ============================================================

page_title(
    "target",
    "Logistic Regression",
    "Predict Pass or Fail using the same six pre-exam "
    "student-performance variables.",
)


# ============================================================
# MODEL OVERVIEW
# ============================================================

section_title(
    "database",
    "Model Overview",
)


o1, o2, o3, o4 = (
    st.columns(4)
)


o1.metric(
    "Training Records",
    f"{train_n:,}",
)

o2.metric(
    "Testing Records",
    f"{test_n:,}",
)

o3.metric(
    "Input Features",
    len(features),
)

o4.metric(
    "Final Model",
    "Balanced",
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

section_title(
    "chart",
    "Class Distribution",
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


class_left, class_right = (
    st.columns(
        [1, 1.2]
    )
)


with class_left:

    c1, c2 = (
        st.columns(2)
    )


    c1.metric(
        "Pass",
        f"{pass_count:,}",
        (
            f"{pass_count / len(df) * 100:.1f}%"
        ),
    )


    c2.metric(
        "Fail",
        f"{fail_count:,}",
        (
            f"{fail_count / len(df) * 100:.1f}%"
        ),
    )


    st.caption(
        "The majority class is Pass, so ordinary accuracy "
        "should not be interpreted by itself."
    )


with class_right:

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
        hole=0.64,
    )


    fig_class.update_traces(
        textinfo="percent+label",
        textposition="inside",
    )


    fig_class.update_layout(
        template=plotly_template,

        height=250,

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


# ============================================================
# PERFORMANCE DASHBOARD
# ============================================================

section_title(
    "target",
    "Balanced Logistic Regression Performance",
)


metric_cols = (
    st.columns(6)
)


metric_values = [
    (
        "Accuracy",
        balanced_metrics[
            "Accuracy"
        ] * 100,
    ),
    (
        "Precision",
        balanced_metrics[
            "Precision"
        ] * 100,
    ),
    (
        "Recall",
        balanced_metrics[
            "Recall"
        ] * 100,
    ),
    (
        "F1",
        balanced_metrics[
            "F1 Score"
        ] * 100,
    ),
    (
        "Balanced Acc.",
        balanced_metrics[
            "Balanced Accuracy"
        ] * 100,
    ),
    (
        "AUC",
        auc * 100,
    ),
]


for col, (
    metric_name,
    metric_value,
) in zip(
    metric_cols,
    metric_values,
):

    with col:

        st.metric(
            metric_name,
            f"{metric_value:.1f}%",
        )


# ============================================================
# METRIC PROFILE + RADAR
# ============================================================

profile_left, profile_right = (
    st.columns(
        [1.4, 1]
    )
)


with profile_left:

    section_title(
        "chart",
        "Metric Profile",
    )


    metric_profile_df = pd.DataFrame(
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
                balanced_metrics[
                    "Accuracy"
                ] * 100,

                balanced_metrics[
                    "Precision"
                ] * 100,

                balanced_metrics[
                    "Recall"
                ] * 100,

                balanced_metrics[
                    "F1 Score"
                ] * 100,

                balanced_metrics[
                    "Balanced Accuracy"
                ] * 100,

                auc * 100,
            ],
        }
    )


    fig_metric_profile = px.bar(
        metric_profile_df,
        x="Score",
        y="Metric",
        orientation="h",
        text="Score",
    )


    fig_metric_profile.update_traces(
        marker_color=accent,
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )


    fig_metric_profile.update_layout(
        height=375,

        margin=dict(
            l=20,
            r=65,
            t=10,
            b=20,
        ),

        xaxis=dict(
            range=[
                0,
                100,
            ],
            title="Score (%)",
            ticksuffix="%",
        ),

        yaxis_title="",

        showlegend=False,
    )


    style_chart(
        fig_metric_profile
    )


    st.plotly_chart(
        fig_metric_profile,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


with profile_right:

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
        balanced_metrics[
            "Accuracy"
        ] * 100,

        balanced_metrics[
            "Precision"
        ] * 100,

        balanced_metrics[
            "Recall"
        ] * 100,

        balanced_metrics[
            "F1 Score"
        ] * 100,

        balanced_metrics[
            "Balanced Accuracy"
        ] * 100,

        auc * 100,
    ]


    fig_radar = go.Figure()


    fig_radar.add_trace(
        go.Scatterpolar(
            r=(
                radar_values
                + [
                    radar_values[0]
                ]
            ),

            theta=(
                radar_labels
                + [
                    radar_labels[0]
                ]
            ),

            fill="toself",

            line=dict(
                color=accent
            ),

            name=(
                "Balanced Logistic"
            ),
        )
    )


    fig_radar.update_layout(
        template=plotly_template,

        height=375,

        margin=dict(
            l=45,
            r=45,
            t=20,
            b=20,
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
# CONFUSION MATRIX
# ============================================================

section_title(
    "chart",
    "Confusion Matrix",
)


tn, fp, fn, tp = (
    cm_balanced.ravel()
)


cm_df = pd.DataFrame(
    cm_balanced,

    index=[
        "Actual Fail",
        "Actual Pass",
    ],

    columns=[
        "Predicted Fail",
        "Predicted Pass",
    ],
)


fig_cm = px.imshow(
    cm_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
)


fig_cm.update_layout(
    template=plotly_template,

    height=330,

    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20,
    ),

    paper_bgcolor=(
        chart_background
    ),

    plot_bgcolor=(
        chart_background
    ),

    font=dict(
        color=chart_text
    ),

    coloraxis_colorbar=dict(
        title="Students"
    ),
)


fig_cm.update_xaxes(
    color=chart_text
)

fig_cm.update_yaxes(
    color=chart_text
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
# CONFUSION MATRIX METRICS
# ============================================================

conf1, conf2, conf3, conf4 = (
    st.columns(4)
)


conf1.metric(
    "True Fail",
    f"{tn:,}",
)

conf2.metric(
    "False Pass",
    f"{fp:,}",
)

conf3.metric(
    "False Fail",
    f"{fn:,}",
)

conf4.metric(
    "True Pass",
    f"{tp:,}",
)


recall1, recall2, recall3 = (
    st.columns(3)
)


recall1.metric(
    "Fail Recall",
    f"{fail_recall * 100:.1f}%",
)

recall2.metric(
    "Pass Recall",
    f"{pass_recall * 100:.1f}%",
)

recall3.metric(
    "Balanced Accuracy",
    (
        f"{balanced_metrics['Balanced Accuracy'] * 100:.1f}%"
    ),
)


st.caption(
    "Balanced Accuracy is the average of recall across both outcome classes."
)


# ============================================================
# PREDICTED PROBABILITY DISTRIBUTION
# ============================================================

section_title(
    "chart",
    "Predicted Pass Probability Distribution",
)


fig_probability = px.histogram(
    probability_df,

    x="Pass Probability",

    color="Actual Outcome",

    nbins=45,

    barmode="overlay",

    opacity=0.60,

    histnorm=(
        "probability density"
    ),
)


fig_probability.add_vline(
    x=0.50,
    line_dash="dash",
    line_color=chart_muted,
)


fig_probability.update_layout(
    height=370,

    margin=dict(
        l=20,
        r=20,
        t=15,
        b=30,
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


fig_probability.update_xaxes(
    tickformat=".0%"
)


style_chart(
    fig_probability
)


st.plotly_chart(
    fig_probability,
    width="stretch",
    theme=None,
    config={
        "displayModeBar":
            False
    },
)


st.caption(
    "Better class separation appears when actual Pass students receive "
    "higher probabilities and actual Fail students receive lower probabilities."
)


# ============================================================
# CLASSIFIER COMPARISON
# ============================================================

section_title(
    "scale",
    "Classification Model Comparison",
)


display_comparison = (
    comparison_df[
        [
            "Model",
            "Accuracy",
            "Balanced Accuracy",
        ]
    ]
    .copy()
)


display_comparison[
    "Accuracy"
] = display_comparison[
    "Accuracy"
].map(
    lambda value:
        f"{value * 100:.1f}%"
)


display_comparison[
    "Balanced Accuracy"
] = display_comparison[
    "Balanced Accuracy"
].map(
    lambda value:
        f"{value * 100:.1f}%"
)


show_theme_table(
    display_comparison
)


compare_long = (
    comparison_df.melt(
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
    compare_long,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
    text="Score",
)


fig_compare.update_traces(
    texttemplate="%{text:.1%}",
    textposition="outside",
)


fig_compare.update_layout(
    height=390,

    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20,
    ),

    xaxis_title="",

    yaxis=dict(
        title="Score",
        range=[
            0,
            1,
        ],
        tickformat=".0%",
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
    "Accuracy Gap",
)


gap_df = (
    comparison_df[
        [
            "Model",
            "Accuracy Gap",
        ]
    ]
    .copy()
)


gap_df[
    "Accuracy Gap"
] = (
    gap_df[
        "Accuracy Gap"
    ]
    * 100
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
    height=280,

    margin=dict(
        l=20,
        r=85,
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
    "A large positive gap suggests that ordinary accuracy is being "
    "boosted by the majority class. The balanced model has a much smaller gap."
)


# ============================================================
# BALANCED ACCURACY RANKING
# ============================================================

section_title(
    "chart",
    "Balanced Accuracy Ranking",
)


ranking_df = (
    comparison_df[
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
    ranking_df,
    x="Balanced Accuracy",
    y="Model",
    orientation="h",
    text="Balanced Accuracy",
)


fig_rank.update_traces(
    marker_color=accent,
    texttemplate="%{text:.1%}",
    textposition="outside",
)


fig_rank.update_layout(
    height=280,

    margin=dict(
        l=20,
        r=70,
        t=10,
        b=20,
    ),

    xaxis=dict(
        title=(
            "Balanced Accuracy"
        ),
        range=[
            0,
            1,
        ],
        tickformat=".0%",
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
# ROC + PRECISION-RECALL
# ============================================================

section_title(
    "trending",
    "Threshold-Independent Evaluation",
)


roc_col, pr_col = (
    st.columns(2)
)


# ------------------------------------------------------------
# ROC
# ------------------------------------------------------------

with roc_col:

    roc_df = pd.DataFrame(
        {
            "False Positive Rate":
                fpr,

            "True Positive Rate":
                tpr,
        }
    )


    fig_roc = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
    )


    fig_roc.update_traces(
        line=dict(
            color=accent,
            width=3,
        )
    )


    fig_roc.add_shape(
        type="line",

        x0=0,
        y0=0,

        x1=1,
        y1=1,

        line=dict(
            color=chart_muted,
            dash="dash",
            width=2,
        ),
    )


    fig_roc.update_layout(
        height=360,

        title=dict(
            text=(
                f"ROC Curve · AUC {auc:.3f}"
            ),
            font=dict(
                size=14
            ),
        ),

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=25,
        ),

        xaxis_title=(
            "False Positive Rate"
        ),

        yaxis_title=(
            "True Positive Rate"
        ),
    )


    fig_roc.update_xaxes(
        range=[
            0,
            1,
        ]
    )


    fig_roc.update_yaxes(
        range=[
            0,
            1,
        ]
    )


    style_chart(
        fig_roc
    )


    st.plotly_chart(
        fig_roc,
        width="stretch",
        theme=None,
        config={
            "displayModeBar":
                False
        },
    )


# ------------------------------------------------------------
# PRECISION-RECALL
# ------------------------------------------------------------

with pr_col:

    pr_df = pd.DataFrame(
        {
            "Recall":
                pr_recall,

            "Precision":
                pr_precision,
        }
    )


    fig_pr = px.line(
        pr_df,
        x="Recall",
        y="Precision",
    )


    fig_pr.update_traces(
        line=dict(
            color=accent,
            width=3,
        )
    )


    fig_pr.update_layout(
        height=360,

        title=dict(
            text=(
                f"Precision–Recall · AP "
                f"{average_precision:.3f}"
            ),
            font=dict(
                size=14
            ),
        ),

        margin=dict(
            l=20,
            r=20,
            t=40,
            b=25,
        ),

        xaxis_title="Recall",

        yaxis_title="Precision",
    )


    fig_pr.update_xaxes(
        range=[
            0,
            1,
        ],
        tickformat=".0%",
    )


    fig_pr.update_yaxes(
        range=[
            0,
            1,
        ],
        tickformat=".0%",
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
    "ROC AUC summarizes class separation across thresholds. "
    "Average Precision summarizes the precision–recall relationship."
)


# ============================================================
# FEATURE INFLUENCE
# ============================================================

section_title(
    "sliders",
    "Feature Influence",
)


balanced_model = (
    get_logistic_model()
)


coefficients = (
    balanced_model
    .named_steps["model"]
    .coef_[0]
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
        if coefficient >= 0
        else negative_color
    )
    for coefficient
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

        texttemplate="%{text:.2f}",

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
        "Standardized Logistic Coefficient"
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
    "Positive coefficients move the model toward Pass; negative coefficients "
    "move it toward Fail. These are model associations, not causal effects."
)


# ============================================================
# FINAL TAKEAWAY
# ============================================================

section_title(
    "brain",
    "Logistic Regression Takeaway",
)


take1, take2, take3 = (
    st.columns(3)
)


with take1:

    icon_card(
        "database",
        "Class Imbalance",
        (
            f"Pass students make up "
            f"{pass_count / len(df) * 100:.1f}% "
            f"of the dataset, compared with "
            f"{fail_count / len(df) * 100:.1f}% Fail."
        ),
    )


with take2:

    icon_card(
        "scale",
        "Balanced Performance",
        (
            f"Balanced Accuracy is "
            f"{balanced_metrics['Balanced Accuracy'] * 100:.1f}%, "
            f"with Pass Recall {pass_recall * 100:.1f}% and "
            f"Fail Recall {fail_recall * 100:.1f}%."
        ),
    )


with take3:

    icon_card(
        "check",
        "Final Selection",
        (
            "Balanced Logistic Regression is selected because it provides "
            "the strongest performance across both classes."
        ),
    )


st.success(
    "Balanced Logistic Regression is the final classifier because it "
    "handles the imbalanced Pass / Fail outcome more evenly than the "
    "Dummy baseline or Standard Logistic Regression."
)


# ============================================================
# FOOTER
# ============================================================

show_footer()