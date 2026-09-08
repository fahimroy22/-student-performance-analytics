# ============================================================
# LOGISTIC REGRESSION PAGE
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from components.styles import apply_global_styles
from components.footer import show_footer
from components.icons import (
    page_title,
    section_title,
    icon_card,
)

from core.model_loader import load_logistic_model


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Logistic Regression",
    page_icon="📊",
    layout="wide",
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
# MODEL VARIABLES
# ------------------------------------------------------------

features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed",
]

target = "pass_status"


# ------------------------------------------------------------
# PREPARE DATA AND EVALUATE MODELS
# ------------------------------------------------------------

@st.cache_data
def prepare_classification_results():

    # --------------------------------------------------------
    # INPUT FEATURES AND TARGET
    # --------------------------------------------------------

    X = df[features].copy()

    # Keep labels as text so they match the saved model.
    #
    # Saved model classes:
    # "Fail" and "Pass"
    #
    # Do NOT map these to 0 and 1 here.
    y = df[target].astype(str)


    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


    # --------------------------------------------------------
    # SAVED BALANCED LOGISTIC REGRESSION MODEL
    # --------------------------------------------------------

    balanced_model = load_logistic_model()

    y_pred_balanced = balanced_model.predict(
        X_test
    )

    probability_matrix = balanced_model.predict_proba(
        X_test
    )


    # --------------------------------------------------------
    # IDENTIFY THE "PASS" PROBABILITY COLUMN
    # --------------------------------------------------------

    balanced_classes = list(
        balanced_model.named_steps[
            "model"
        ].classes_
    )

    pass_index = balanced_classes.index(
        "Pass"
    )

    y_prob_balanced = probability_matrix[
        :,
        pass_index
    ]


    # --------------------------------------------------------
    # STANDARD LOGISTIC REGRESSION
    # --------------------------------------------------------

    standard_model = Pipeline([
        (
            "preprocessing",
            Pipeline([
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
            ]),
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1000
            ),
        ),
    ])

    standard_model.fit(
        X_train,
        y_train,
    )

    y_pred_standard = standard_model.predict(
        X_test
    )


    # --------------------------------------------------------
    # DUMMY BASELINE
    # --------------------------------------------------------

    dummy_model = DummyClassifier(
        strategy="most_frequent"
    )

    dummy_model.fit(
        X_train,
        y_train,
    )

    y_pred_dummy = dummy_model.predict(
        X_test
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


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    balanced_metrics = get_metrics(
        y_test,
        y_pred_balanced,
    )

    standard_metrics = get_metrics(
        y_test,
        y_pred_standard,
    )

    dummy_metrics = get_metrics(
        y_test,
        y_pred_dummy,
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


    # --------------------------------------------------------
    # ROC / AUC
    # --------------------------------------------------------
    #
    # ROC functions need a binary target.
    # Convert ONLY for ROC/AUC.
    # All other classification metrics continue using
    # "Fail" / "Pass" labels.
    # --------------------------------------------------------

    y_test_binary = (
        y_test == "Pass"
    ).astype(int)

    auc = roc_auc_score(
        y_test_binary,
        y_prob_balanced,
    )

    fpr, tpr, thresholds = roc_curve(
        y_test_binary,
        y_prob_balanced,
    )


    # --------------------------------------------------------
    # MODEL COMPARISON TABLE
    # --------------------------------------------------------

    comparison_df = pd.DataFrame({
        "Model": [
            "Dummy Baseline",
            "Standard Logistic Regression",
            "Balanced Logistic Regression",
        ],

        "Accuracy": [
            dummy_metrics["Accuracy"],
            standard_metrics["Accuracy"],
            balanced_metrics["Accuracy"],
        ],

        "Balanced Accuracy": [
            dummy_metrics["Balanced Accuracy"],
            standard_metrics["Balanced Accuracy"],
            balanced_metrics["Balanced Accuracy"],
        ],
    })


    return (
        balanced_metrics,
        standard_metrics,
        dummy_metrics,
        cm_balanced,
        auc,
        fpr,
        tpr,
        comparison_df,
        len(X_train),
        len(X_test),
    )


(
    balanced_metrics,
    standard_metrics,
    dummy_metrics,
    cm_balanced,
    auc,
    fpr,
    tpr,
    comparison_df,
    train_n,
    test_n,
) = prepare_classification_results()


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

page_title(
    "target",
    "Logistic Regression",
    "Predict Pass or Fail using the same six pre-exam "
    "student-performance variables.",
)


# ------------------------------------------------------------
# MODEL OVERVIEW
# ------------------------------------------------------------

section_title(
    "database",
    "Model Overview",
)


o1, o2, o3, o4 = st.columns(4)


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

with o4:

    st.caption(
        "Final Model"
    )

    st.markdown(
        "#### Balanced Logistic Regression"
    )


# ------------------------------------------------------------
# CLASS DISTRIBUTION
# ------------------------------------------------------------

section_title(
    "chart",
    "Class Distribution",
)


class_counts = (
    df["pass_status"]
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


class_col1, class_col2 = st.columns(
    [1, 1.2]
)


with class_col1:

    m1, m2 = st.columns(2)

    m1.metric(
        "Pass",
        f"{pass_count:,}",
        f"{pass_count / len(df) * 100:.1f}%",
    )

    m2.metric(
        "Fail",
        f"{fail_count:,}",
        f"{fail_count / len(df) * 100:.1f}%",
    )

    st.caption(
        "Pass is the majority class, so ordinary accuracy "
        "can give an overly optimistic view of performance."
    )


with class_col2:

    class_df = pd.DataFrame({
        "Status": [
            "Pass",
            "Fail",
        ],
        "Students": [
            pass_count,
            fail_count,
        ],
    })


    fig_class = px.pie(
        class_df,
        names="Status",
        values="Students",
        hole=0.62,
    )


    fig_class.update_traces(
        textinfo="percent+label",
        textposition="inside",
    )


    fig_class.update_layout(
        height=260,
        margin=dict(
            l=10,
            r=10,
            t=5,
            b=5,
        ),
        showlegend=False,
    )


    st.plotly_chart(
        fig_class,
        width="stretch",
        config={
            "displayModeBar": False
        },
    )


st.info(
    "Because the classes are imbalanced, Balanced Accuracy is "
    "especially useful because it gives equal importance to "
    "performance on both Pass and Fail students."
)


# ------------------------------------------------------------
# FINAL MODEL PERFORMANCE
# ------------------------------------------------------------

section_title(
    "target",
    "Balanced Logistic Regression Performance",
)


metric_cols = st.columns(5)


metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "Balanced Accuracy",
]


metric_descriptions = {
    "Accuracy":
        "Overall percentage of correct predictions.",

    "Precision":
        "How reliable positive Pass predictions are.",

    "Recall":
        "How many actual Pass students are identified.",

    "F1 Score":
        "Balance between precision and recall.",

    "Balanced Accuracy":
        "Average performance across both classes.",
}


for col, metric_name in zip(
    metric_cols,
    metric_names,
):

    with col:

        with st.container(
            border=True
        ):

            st.metric(
                metric_name,
                f"{balanced_metrics[metric_name] * 100:.1f}%",
            )

            st.caption(
                metric_descriptions[
                    metric_name
                ]
            )


# ------------------------------------------------------------
# PERFORMANCE SUMMARY
# ------------------------------------------------------------

summary1, summary2, summary3 = (
    st.columns(3)
)


with summary1:

    icon_card(
        "check",
        "Overall Accuracy",
        f"{balanced_metrics['Accuracy'] * 100:.1f}% "
        "of test predictions are correct.",
    )


with summary2:

    icon_card(
        "scale",
        "Balanced Accuracy",
        f"{balanced_metrics['Balanced Accuracy'] * 100:.1f}% "
        "when both classes are weighted equally.",
    )


with summary3:

    icon_card(
        "target",
        "ROC AUC",
        f"AUC = {auc:.3f}, indicating useful separation "
        "between Pass and Fail students.",
    )


# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

section_title(
    "chart",
    "Confusion Matrix",
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
    height=380,
    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20,
    ),
    coloraxis_colorbar=dict(
        title="Students"
    ),
)


st.plotly_chart(
    fig_cm,
    width="stretch",
    config={
        "displayModeBar": False
    },
)


tn, fp, fn, tp = (
    cm_balanced.ravel()
)


conf1, conf2, conf3, conf4 = (
    st.columns(4)
)


conf1.metric(
    "Correct Fails",
    f"{tn:,}",
)

conf2.metric(
    "Fail → Pass",
    f"{fp:,}",
)

conf3.metric(
    "Pass → Fail",
    f"{fn:,}",
)

conf4.metric(
    "Correct Passes",
    f"{tp:,}",
)


st.caption(
    "The balanced classifier places more emphasis on identifying "
    "the minority Fail class than an ordinary Logistic Regression model."
)


# ------------------------------------------------------------
# MODEL COMPARISON
# ------------------------------------------------------------

section_title(
    "scale",
    "Classification Model Comparison",
)


display_comparison = (
    comparison_df
    .copy()
)


display_comparison[
    "Accuracy"
] = (
    display_comparison[
        "Accuracy"
    ] * 100
).round(1)


display_comparison[
    "Balanced Accuracy"
] = (
    display_comparison[
        "Balanced Accuracy"
    ] * 100
).round(1)


display_comparison[
    "Accuracy"
] = (
    display_comparison[
        "Accuracy"
    ].astype(str)
    + "%"
)


display_comparison[
    "Balanced Accuracy"
] = (
    display_comparison[
        "Balanced Accuracy"
    ].astype(str)
    + "%"
)


st.dataframe(
    display_comparison,
    width="stretch",
    hide_index=True,
)


compare_long = comparison_df.melt(
    id_vars="Model",
    value_vars=[
        "Accuracy",
        "Balanced Accuracy",
    ],
    var_name="Metric",
    value_name="Score",
)


fig_compare = px.bar(
    compare_long,
    x="Model",
    y="Score",
    color="Metric",
    barmode="group",
)


fig_compare.update_layout(
    height=380,
    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20,
    ),
    yaxis_title="Score",
    xaxis_title="",
    yaxis_tickformat=".0%",
    legend_title="",
)


st.plotly_chart(
    fig_compare,
    width="stretch",
    config={
        "displayModeBar": False
    },
)


st.warning(
    "The Dummy Baseline appears strong on ordinary accuracy because "
    "most students pass. Its Balanced Accuracy is only 50%, showing "
    "why accuracy alone is misleading for this imbalanced dataset."
)


# ------------------------------------------------------------
# BALANCED ACCURACY RANKING
# ------------------------------------------------------------

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
    xaxis_title="Balanced Accuracy",
    yaxis_title="",
    xaxis_tickformat=".0%",
    showlegend=False,
)


st.plotly_chart(
    fig_rank,
    width="stretch",
    config={
        "displayModeBar": False
    },
)


# ------------------------------------------------------------
# ROC CURVE
# ------------------------------------------------------------

section_title(
    "trending",
    "ROC Curve",
)


roc_df = pd.DataFrame({
    "False Positive Rate": fpr,
    "True Positive Rate": tpr,
})


fig_roc = px.line(
    roc_df,
    x="False Positive Rate",
    y="True Positive Rate",
)


fig_roc.add_shape(
    type="line",
    x0=0,
    y0=0,
    x1=1,
    y1=1,
    line=dict(
        dash="dash",
        width=2,
    ),
)


fig_roc.update_layout(
    height=400,
    margin=dict(
        l=20,
        r=20,
        t=10,
        b=20,
    ),
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
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


st.plotly_chart(
    fig_roc,
    width="stretch",
    config={
        "displayModeBar": False
    },
)


st.caption(
    f"AUC = {auc:.3f}. The ROC curve evaluates how well the model "
    "separates Pass and Fail students across different probability thresholds."
)


# ------------------------------------------------------------
# MODEL INTERPRETATION
# ------------------------------------------------------------

section_title(
    "brain",
    "Model Interpretation",
)


interpret1, interpret2, interpret3 = (
    st.columns(3)
)


with interpret1:

    icon_card(
        "target",
        "Pass Predictions",
        f"Precision is "
        f"{balanced_metrics['Precision'] * 100:.1f}%, "
        "so predicted Pass outcomes are usually correct.",
    )


with interpret2:

    icon_card(
        "check",
        "Pass Detection",
        f"Recall is "
        f"{balanced_metrics['Recall'] * 100:.1f}%, "
        "showing how many actual Pass students are identified.",
    )


with interpret3:

    icon_card(
        "scale",
        "Class Balance",
        f"Balanced Accuracy is "
        f"{balanced_metrics['Balanced Accuracy'] * 100:.1f}%, "
        "reflecting performance across both classes.",
    )


# ------------------------------------------------------------
# WHY BALANCED MODEL
# ------------------------------------------------------------

section_title(
    "scale",
    "Why Use the Balanced Model?",
)


reason1, reason2 = st.columns(2)


with reason1:

    icon_card(
        "check",
        "Minority-Class Detection",
        "Class weighting increases attention to failing students, "
        "who form the smaller outcome group.",
    )


with reason2:

    icon_card(
        "scale",
        "More Balanced Evaluation",
        "The model sacrifices some ordinary accuracy in exchange "
        "for stronger performance across both classes.",
    )


st.success(
    "Balanced Logistic Regression is used in the final application "
    "because it provides the strongest Balanced Accuracy and avoids "
    "favoring only the majority Pass class."
)


# ------------------------------------------------------------
# CONCLUSION
# ------------------------------------------------------------

section_title(
    "check",
    "Conclusion",
)


st.caption(
    "Balanced Logistic Regression provides both a Pass / Fail "
    "classification and a probability estimate. Its stronger "
    "performance across both outcome classes makes it the preferred "
    "classifier for the final application."
)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

show_footer()