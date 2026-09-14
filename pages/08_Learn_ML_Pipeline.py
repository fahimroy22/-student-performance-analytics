import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from components.footer import show_footer
from components.icons import page_title, section_title
from components.styles import (
    apply_global_styles,
    get_plotly_template,
    get_theme_colors,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Learn the ML Pipeline",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# GLOBAL APP STYLE
# ============================================================

apply_global_styles()

theme = get_theme_colors()
plotly_template = get_plotly_template()

IS_DARK = st.session_state.get("dark_mode", False)

APP_BG = theme["background"]
SURFACE = theme["surface"]
TEXT = theme["text"]
MUTED = theme["muted"]
BORDER = theme["border"]
ACCENT = theme["accent"]

SOFT_BG = "#182235" if IS_DARK else "#F8FAFC"
CODE_BG = "#111827" if IS_DARK else "#F7F8FA"
CODE_HEADER_BG = "#1F2937" if IS_DARK else "#EEF2F7"
CODE_TEXT = "#E5E7EB" if IS_DARK else "#1F2937"
GRID_COLOR = "#334155" if IS_DARK else "#E5E7EB"


# ============================================================
# PAGE CSS
# ============================================================

st.markdown(
    f"""
    <style>

    .ml-chapter-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 22px 24px;
        margin: 30px 0 20px 0;
    }}

    .ml-chapter-number {{
        color: {ACCENT};
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }}

    .ml-chapter-title {{
        color: {TEXT};
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.25;
        margin-bottom: 5px;
    }}

    .ml-chapter-subtitle {{
        color: {MUTED};
        font-size: 0.93rem;
        line-height: 1.55;
    }}

    .ml-concept-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 4px solid {ACCENT};
        border-radius: 12px;
        padding: 16px 18px;
        margin: 12px 0 18px 0;
    }}

    .ml-concept-title {{
        color: {TEXT};
        font-weight: 700;
        margin-bottom: 6px;
    }}

    .ml-concept-text {{
        color: {MUTED};
        line-height: 1.6;
        font-size: 0.93rem;
    }}

    .ml-output-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 16px;
        min-height: 102px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }}

    .ml-output-label {{
        color: {MUTED};
        font-size: 0.8rem;
        margin-bottom: 7px;
    }}

    .ml-output-value {{
        color: {TEXT};
        font-size: 1.52rem;
        font-weight: 700;
    }}

    .ml-roadmap-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 13px;
        padding: 16px;
        min-height: 118px;
    }}

    .ml-roadmap-title {{
        color: {TEXT};
        font-weight: 700;
        margin-bottom: 7px;
    }}

    .ml-roadmap-text {{
        color: {MUTED};
        font-size: 0.86rem;
        line-height: 1.5;
    }}

    .ml-flow-box {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 11px;
        padding: 14px 10px;
        min-height: 70px;
        color: {TEXT};
        font-weight: 650;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }}

    .ml-flow-arrow {{
        color: {MUTED};
        text-align: center;
        font-size: 1.35rem;
        padding-top: 18px;
    }}

    .ml-table-wrap {{
        overflow-x: auto;
        width: 100%;
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 11px;
        margin: 10px 0 18px 0;
    }}

    .ml-table {{
        border-collapse: collapse;
        width: 100%;
        color: {TEXT};
        background: {SURFACE};
        font-size: 0.87rem;
    }}

    .ml-table th {{
        background: {SOFT_BG};
        color: {TEXT};
        font-weight: 650;
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid {BORDER};
        white-space: nowrap;
    }}

    .ml-table td {{
        color: {TEXT};
        padding: 9px 12px;
        border-bottom: 1px solid {BORDER};
        white-space: nowrap;
    }}

    .ml-table tr:last-child td {{
        border-bottom: none;
    }}

    .ml-code-block {{
        background: {CODE_BG};
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
        margin: 10px 0 18px 0;
    }}

    .ml-code-header {{
        background: {CODE_HEADER_BG};
        color: {MUTED};
        border-bottom: 1px solid {BORDER};
        padding: 8px 14px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }}

    .ml-code-pre {{
        background: {CODE_BG};
        color: {CODE_TEXT};
        margin: 0;
        padding: 18px;
        overflow-x: auto;
        line-height: 1.65;
        font-size: 0.91rem;
    }}

    .ml-code-pre code {{
        background: transparent !important;
        color: {CODE_TEXT} !important;
        font-family:
            "SFMono-Regular",
            Consolas,
            "Liberation Mono",
            Menlo,
            monospace;
        white-space: pre;
    }}

    [data-testid="stExpander"] {{
        background: {SURFACE} !important;
        border-color: {BORDER} !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
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

REGRESSION_TARGET = "exam_score"
CLASSIFICATION_TARGET = "pass_status"

FEATURE_LABELS = {
    "previous_exam_score": "Previous Exam Score",
    "previous_gpa": "Previous GPA",
    "attendance_percentage": "Attendance Percentage",
    "assignment_completion_rate": "Assignment Completion Rate",
    "study_hours_per_day": "Study Hours per Day",
    "practice_tests_completed": "Practice Tests Completed",
}


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# ============================================================
# TRAIN MODELS
# ============================================================

@st.cache_resource
def build_tutorial_models(data):

    # ---------------- REGRESSION ----------------

    X_reg = data[FEATURES]
    y_reg = data[REGRESSION_TARGET]

    (
        X_train_reg,
        X_test_reg,
        y_train_reg,
        y_test_reg,
    ) = train_test_split(
        X_reg,
        y_reg,
        test_size=0.20,
        random_state=42,
    )

    linear_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    linear_pipeline.fit(
        X_train_reg,
        y_train_reg,
    )

    y_pred_reg = linear_pipeline.predict(
        X_test_reg
    )

    mae = mean_absolute_error(
        y_test_reg,
        y_pred_reg,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test_reg,
            y_pred_reg,
        )
    )

    r2 = r2_score(
        y_test_reg,
        y_pred_reg,
    )

    # ---------------- CLASSIFICATION ----------------

    X_cls = data[FEATURES]
    y_cls = data[CLASSIFICATION_TARGET]

    (
        X_train_cls,
        X_test_cls,
        y_train_cls,
        y_test_cls,
    ) = train_test_split(
        X_cls,
        y_cls,
        test_size=0.20,
        random_state=42,
        stratify=y_cls,
    )

    balanced_logistic = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                ),
            ),
        ]
    )

    balanced_logistic.fit(
        X_train_cls,
        y_train_cls,
    )

    balanced_pred = balanced_logistic.predict(
        X_test_cls
    )

    classes = list(
        balanced_logistic
        .named_steps["model"]
        .classes_
    )

    pass_index = classes.index("Pass")

    pass_probability = (
        balanced_logistic
        .predict_proba(X_test_cls)
        [:, pass_index]
    )

    accuracy = accuracy_score(
        y_test_cls,
        balanced_pred,
    )

    precision = precision_score(
        y_test_cls,
        balanced_pred,
        pos_label="Pass",
    )

    recall = recall_score(
        y_test_cls,
        balanced_pred,
        pos_label="Pass",
    )

    f1 = f1_score(
        y_test_cls,
        balanced_pred,
        pos_label="Pass",
    )

    balanced_acc = balanced_accuracy_score(
        y_test_cls,
        balanced_pred,
    )

    y_binary = (
        y_test_cls == "Pass"
    ).astype(int)

    auc = roc_auc_score(
        y_binary,
        pass_probability,
    )

    fpr, tpr, _ = roc_curve(
        y_binary,
        pass_probability,
    )

    cm = confusion_matrix(
        y_test_cls,
        balanced_pred,
        labels=["Fail", "Pass"],
    )

    # ---------------- STANDARD LOGISTIC ----------------

    standard_logistic = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000
                ),
            ),
        ]
    )

    standard_logistic.fit(
        X_train_cls,
        y_train_cls,
    )

    standard_pred = standard_logistic.predict(
        X_test_cls
    )

    standard_accuracy = accuracy_score(
        y_test_cls,
        standard_pred,
    )

    standard_balanced = balanced_accuracy_score(
        y_test_cls,
        standard_pred,
    )

    # ---------------- DUMMY ----------------

    dummy_model = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                DummyClassifier(
                    strategy="most_frequent"
                ),
            ),
        ]
    )

    dummy_model.fit(
        X_train_cls,
        y_train_cls,
    )

    dummy_pred = dummy_model.predict(
        X_test_cls
    )

    dummy_accuracy = accuracy_score(
        y_test_cls,
        dummy_pred,
    )

    dummy_balanced = balanced_accuracy_score(
        y_test_cls,
        dummy_pred,
    )

    return {
        "linear_pipeline": linear_pipeline,
        "balanced_logistic": balanced_logistic,
        "y_test_reg": y_test_reg,
        "y_pred_reg": y_pred_reg,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "cm": cm,
        "standard_accuracy": standard_accuracy,
        "standard_balanced": standard_balanced,
        "dummy_accuracy": dummy_accuracy,
        "dummy_balanced": dummy_balanced,
    }


results = build_tutorial_models(df)


# ============================================================
# HELPERS
# ============================================================

def safe(value):
    return html.escape(str(value))


def chapter(number, title, subtitle):

    st.html(
        (
            '<div class="ml-chapter-card">'
            f'<div class="ml-chapter-number">Chapter {safe(number)}</div>'
            f'<div class="ml-chapter-title">{safe(title)}</div>'
            f'<div class="ml-chapter-subtitle">{safe(subtitle)}</div>'
            '</div>'
        )
    )


def concept(title, text):

    st.html(
        (
            '<div class="ml-concept-card">'
            f'<div class="ml-concept-title">{safe(title)}</div>'
            f'<div class="ml-concept-text">{safe(text)}</div>'
            '</div>'
        )
    )


def output_card(label, value):

    st.html(
        (
            '<div class="ml-output-card">'
            f'<div class="ml-output-label">{safe(label)}</div>'
            f'<div class="ml-output-value">{safe(value)}</div>'
            '</div>'
        )
    )


def roadmap_card(title, description):

    st.html(
        (
            '<div class="ml-roadmap-card">'
            f'<div class="ml-roadmap-title">{safe(title)}</div>'
            f'<div class="ml-roadmap-text">{safe(description)}</div>'
            '</div>'
        )
    )


def flow_box(text):

    st.html(
        (
            '<div class="ml-flow-box">'
            f'{safe(text)}'
            '</div>'
        )
    )


def flow_arrow():
    st.html(
        '<div class="ml-flow-arrow">→</div>'
    )


def step(number, title):

    st.caption(
        f"STEP {number}"
    )

    st.subheader(
        title
    )


def code_block(code, language="python"):

    escaped_code = html.escape(
        code.strip()
    )

    st.html(
        (
            '<div class="ml-code-block">'
            f'<div class="ml-code-header">{safe(language.title())}</div>'
            '<pre class="ml-code-pre">'
            f'<code>{escaped_code}</code>'
            '</pre>'
            '</div>'
        )
    )


def show_table(
    dataframe,
    index=False,
):

    html_table = dataframe.to_html(
        index=index,
        border=0,
        classes="ml-table",
        escape=True,
    )

    st.html(
        (
            '<div class="ml-table-wrap">'
            f'{html_table}'
            '</div>'
        )
    )


def style_plotly(fig, height=None):

    fig.update_layout(
        template=plotly_template,
        paper_bgcolor=APP_BG,
        plot_bgcolor=APP_BG,
        font=dict(
            color=TEXT
        ),
        legend=dict(
            font=dict(
                color=TEXT
            )
        ),
        margin=dict(
            l=30,
            r=25,
            t=65,
            b=35,
        ),
    )

    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        color=TEXT,
    )

    fig.update_yaxes(
        gridcolor=GRID_COLOR,
        zerolinecolor=GRID_COLOR,
        color=TEXT,
    )

    if height is not None:
        fig.update_layout(
            height=height
        )

    return fig


# ============================================================
# HEADER
# ============================================================

page_title(
    "graduation",
    "Learn the ML Pipeline",
    (
        "A step-by-step guide to understanding the "
        "machine-learning process used in this project."
    ),
)

st.info(
    "This tutorial focuses only on the machine-learning workflow: "
    "understanding the data, preparing it, training models, evaluating them, "
    "and making predictions."
)


# ============================================================
# LEARNING ROADMAP
# ============================================================

section_title(
    "sparkles",
    "Learning Roadmap",
)

roadmap_columns = st.columns(5)

roadmap_items = [
    (
        "1. Understand Data",
        "Explore rows, columns, distributions, missing values and class balance.",
    ),
    (
        "2. Prepare Data",
        "Choose features, define targets, split data and preprocess it.",
    ),
    (
        "3. Regression",
        "Use Linear Regression to predict exam scores.",
    ),
    (
        "4. Classification",
        "Use Logistic Regression to predict Pass or Fail.",
    ),
    (
        "5. Prediction",
        "Use both trained models for a new student.",
    ),
]

for col, item in zip(
    roadmap_columns,
    roadmap_items,
):

    with col:
        roadmap_card(
            item[0],
            item[1],
        )


# ============================================================
# COMPLETE ML FLOW
# ============================================================

section_title(
    "brain",
    "The Entire Machine-Learning Process",
)

flow_cols = st.columns(
    [
        1,
        0.18,
        1,
        0.18,
        1,
        0.18,
        1,
    ]
)

flow_items = [
    "Raw Dataset",
    "arrow",
    "Preprocessing",
    "arrow",
    "Model Training",
    "arrow",
    "Prediction",
]

for col, item in zip(
    flow_cols,
    flow_items,
):

    with col:

        if item == "arrow":
            flow_arrow()

        else:
            flow_box(item)


# ============================================================
# CHAPTER 1
# ============================================================

chapter(
    1,
    "Understanding the Dataset",
    (
        "Before building a model, we first need to understand "
        "what information the dataset contains."
    ),
)


# ============================================================
# STEP 1
# ============================================================

step(
    1,
    "What Is Machine Learning?",
)

concept(
    "Simple definition",
    (
        "Machine learning uses existing examples to learn patterns. "
        "After learning those patterns, a model can make predictions "
        "for new examples."
    ),
)

c1, c2 = st.columns(2)

with c1:

    st.markdown("### Regression")

    flow_box("Student Information")
    flow_arrow()
    flow_box("Linear Regression")
    flow_arrow()
    flow_box("Predicted Exam Score")


with c2:

    st.markdown("### Classification")

    flow_box("Student Information")
    flow_arrow()
    flow_box("Logistic Regression")
    flow_arrow()
    flow_box("Pass or Fail")


# ============================================================
# STEP 2
# ============================================================

step(
    2,
    "Import the Main Python Libraries",
)

code_block(
    """
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
""",
    "python",
)

library_df = pd.DataFrame(
    {
        "Library": [
            "pandas",
            "NumPy",
            "scikit-learn",
            "Plotly",
        ],
        "Purpose": [
            "Work with datasets and tables",
            "Perform numerical calculations",
            "Build and evaluate ML models",
            "Create interactive graphs",
        ],
    }
)

show_table(
    library_df
)


# ============================================================
# STEP 3
# ============================================================

step(
    3,
    "Load the Dataset",
)

code_block(
    """
import pandas as pd

df = pd.read_csv(
    "data/student_performance.csv"
)
""",
    "python",
)

concept(
    "What happens here?",
    (
        "pandas reads the CSV file and stores it in a DataFrame. "
        "A DataFrame is a table made of rows and columns."
    ),
)

metric_cols = st.columns(4)

with metric_cols[0]:
    output_card(
        "Student Records",
        f"{len(df):,}",
    )

with metric_cols[1]:
    output_card(
        "Variables",
        df.shape[1],
    )

with metric_cols[2]:
    output_card(
        "Missing Values",
        f"{df.isna().sum().sum():,}",
    )

with metric_cols[3]:
    output_card(
        "Duplicate Rows",
        f"{df.duplicated().sum():,}",
    )

st.markdown("### First Five Rows")

show_table(
    df.head(),
    index=False,
)


# ============================================================
# STEP 4
# ============================================================

step(
    4,
    "Understand Rows and Columns",
)

code_block(
    """
df.shape
""",
    "python",
)

code_block(
    f"""
({df.shape[0]}, {df.shape[1]})
""",
    "output",
)

concept(
    "Reading the result",
    (
        f"The dataset contains {df.shape[0]:,} rows and "
        f"{df.shape[1]} columns. Each row represents one student record. "
        "Each column represents one variable."
    ),
)


# ============================================================
# STEP 5
# ============================================================

step(
    5,
    "Look at the Dataset Variables",
)

with st.expander(
    "View all dataset variables"
):

    variable_df = pd.DataFrame(
        {
            "No.": range(
                1,
                len(df.columns) + 1
            ),
            "Variable": df.columns,
        }
    )

    show_table(
        variable_df
    )


# ============================================================
# STEP 6
# ============================================================

step(
    6,
    "Study Basic Statistics",
)

code_block(
    """
df.describe()
""",
    "python",
)

summary_df = (
    df[
        FEATURES
        + [REGRESSION_TARGET]
    ]
    .describe()
    .round(2)
    .T
    .reset_index()
    .rename(
        columns={
            "index":
                "Variable"
        }
    )
)

show_table(
    summary_df
)

stats_meaning_df = pd.DataFrame(
    {
        "Statistic": [
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max",
        ],
        "Meaning": [
            "Number of available observations",
            "Average",
            "Standard deviation",
            "Smallest value",
            "Median",
            "Largest value",
        ],
    }
)

show_table(
    stats_meaning_df
)


# ============================================================
# STEP 7
# ============================================================

step(
    7,
    "Visualize Exam Score Distribution",
)

code_block(
    """
fig = px.histogram(
    df,
    x="exam_score",
    nbins=30
)
""",
    "python",
)

fig_score = px.histogram(
    df,
    x="exam_score",
    nbins=30,
    title="Exam Score Distribution",
    labels={
        "exam_score":
            "Exam Score"
    },
)

fig_score.update_layout(
    yaxis_title="Number of Students"
)

style_plotly(
    fig_score,
    height=460,
)

st.plotly_chart(
    fig_score,
    width="stretch",
)

concept(
    "Reading the histogram",
    (
        "The horizontal axis shows exam-score ranges. "
        "The height of each bar shows how many students "
        "fall inside that score range."
    ),
)


# ============================================================
# STEP 8
# ============================================================

step(
    8,
    "Check Missing Values",
)

code_block(
    """
df.isnull().sum()
""",
    "python",
)

missing_series = (
    df.isna()
    .sum()
)

missing_series = (
    missing_series[
        missing_series > 0
    ]
    .sort_values(
        ascending=True
    )
)

missing_df = pd.DataFrame(
    {
        "Variable":
            missing_series.index,

        "Missing Values":
            missing_series.values,
    }
)

fig_missing = px.bar(
    missing_df,
    x="Missing Values",
    y="Variable",
    orientation="h",
    title="Missing Values by Variable",
)

style_plotly(
    fig_missing,
    height=440,
)

st.plotly_chart(
    fig_missing,
    width="stretch",
)

concept(
    "What does a missing value mean?",
    (
        "A missing value means that information was unavailable "
        "for that student and variable."
    ),
)


# ============================================================
# STEP 9
# ============================================================

step(
    9,
    "Handle Missing Values with Median Imputation",
)

code_block(
    """
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy="median"
)
""",
    "python",
)

example_imputation = pd.DataFrame(
    {
        "Before": [
            "3.40",
            "3.10",
            "Missing",
            "3.60",
            "3.20",
        ],
        "After Median Imputation": [
            "3.40",
            "3.10",
            "3.30",
            "3.60",
            "3.20",
        ],
    }
)

show_table(
    example_imputation
)

concept(
    "Why median?",
    (
        "Median imputation replaces a missing numeric value "
        "with the middle value learned from the training data."
    ),
)


# ============================================================
# STEP 10
# ============================================================

step(
    10,
    "Check Duplicate Records",
)

code_block(
    """
df.duplicated().sum()
""",
    "python",
)

st.metric(
    "Duplicate Rows",
    f"{df.duplicated().sum():,}",
)


# ============================================================
# STEP 11
# ============================================================

step(
    11,
    "Understand the Pass / Fail Distribution",
)

code_block(
    """
df["pass_status"].value_counts()
""",
    "python",
)

class_counts = (
    df[
        CLASSIFICATION_TARGET
    ]
    .value_counts()
    .reset_index()
)

class_counts.columns = [
    "Status",
    "Students",
]

fig_pass_fail = px.pie(
    class_counts,
    names="Status",
    values="Students",
    hole=0.55,
    title="Pass / Fail Distribution",
)

style_plotly(
    fig_pass_fail,
    height=450,
)

st.plotly_chart(
    fig_pass_fail,
    width="stretch",
)

pass_count = int(
    class_counts.loc[
        class_counts[
            "Status"
        ] == "Pass",
        "Students"
    ].iloc[0]
)

fail_count = int(
    class_counts.loc[
        class_counts[
            "Status"
        ] == "Fail",
        "Students"
    ].iloc[0]
)

c1, c2 = st.columns(2)

with c1:
    output_card(
        "Pass",
        (
            f"{pass_count:,} "
            f"({pass_count / len(df) * 100:.1f}%)"
        ),
    )

with c2:
    output_card(
        "Fail",
        (
            f"{fail_count:,} "
            f"({fail_count / len(df) * 100:.1f}%)"
        ),
    )

concept(
    "Class imbalance",
    (
        "The dataset contains considerably more Pass records than Fail records. "
        "This is why classification should not be judged using accuracy alone."
    ),
)


# ============================================================
# CHAPTER 2
# ============================================================

chapter(
    2,
    "Preparing the Data",
    (
        "Now we select predictors and targets, avoid leakage, "
        "split the dataset and create a preprocessing pipeline."
    ),
)


# ============================================================
# STEP 12
# ============================================================

step(
    12,
    "Understand Features and Targets",
)

concept(
    "Feature",
    (
        "A feature is an input variable used by the model "
        "to make a prediction."
    ),
)

concept(
    "Target",
    (
        "The target is the value we want the model to learn to predict."
    ),
)

feature_df = pd.DataFrame(
    {
        "Python Variable": FEATURES,
        "Readable Name": [
            FEATURE_LABELS[f]
            for f in FEATURES
        ],
    }
)

show_table(
    feature_df
)


# ============================================================
# STEP 13
# ============================================================

step(
    13,
    "Select the Six Predictors",
)

code_block(
    """
features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed"
]

X = df[features]
""",
    "python",
)

concept(
    "Why these six?",
    (
        "They describe prior academic performance, attendance, "
        "assignment completion and study behavior without directly "
        "revealing the final target."
    ),
)


# ============================================================
# STEP 14
# ============================================================

step(
    14,
    "Create Two Different Targets",
)

c1, c2 = st.columns(2)

with c1:

    st.markdown("### Regression")

    code_block(
        """
y_regression = df[
    "exam_score"
]
""",
        "python",
    )

    concept(
        "Regression target",
        (
            "exam_score is numerical, so Linear Regression "
            "is used to estimate a continuous score."
        ),
    )


with c2:

    st.markdown("### Classification")

    code_block(
        """
y_classification = df[
    "pass_status"
]
""",
        "python",
    )

    concept(
        "Classification target",
        (
            "pass_status contains categories, so Logistic Regression "
            "is used to estimate Pass or Fail."
        ),
    )


# ============================================================
# STEP 15
# ============================================================

step(
    15,
    "Avoid Target Leakage",
)

st.error(
    "Do not use exam_score as an input feature when trying to predict exam_score."
)

code_block(
    """
# BAD EXAMPLE

X = df[
    [
        "previous_exam_score",
        "exam_score"
    ]
]
""",
    "python",
)

concept(
    "Target leakage",
    (
        "Target leakage happens when information that reveals the answer "
        "is accidentally included in the model inputs."
    ),
)


# ============================================================
# STEP 16
# ============================================================

step(
    16,
    "Explore Correlations",
)

code_block(
    """
df[
    features + ["exam_score"]
].corr()
""",
    "python",
)

corr = (
    df[
        FEATURES
        + [REGRESSION_TARGET]
    ]
    .corr()
)

corr_names = [
    FEATURE_LABELS.get(
        col,
        "Exam Score",
    )
    for col in corr.columns
]

fig_corr = go.Figure(
    go.Heatmap(
        z=corr.values,
        x=corr_names,
        y=corr_names,
        zmin=-1,
        zmax=1,
        text=np.round(
            corr.values,
            2,
        ),
        texttemplate="%{text}",
        colorbar=dict(
            title="Correlation"
        ),
    )
)

fig_corr.update_layout(
    title="Correlation Matrix"
)

style_plotly(
    fig_corr,
    height=620,
)

st.plotly_chart(
    fig_corr,
    width="stretch",
)

concept(
    "Correlation",
    (
        "+1 means a strong positive linear relationship, "
        "0 means little linear relationship, and "
        "-1 means a strong negative linear relationship."
    ),
)

st.warning(
    "Previous Exam Score and Previous GPA are strongly related. "
    "Keep this in mind when interpreting individual coefficients."
)


# ============================================================
# STEP 17
# ============================================================

step(
    17,
    "Split the Data into Training and Testing Sets",
)

code_block(
    """
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
""",
    "python",
)

split_df = pd.DataFrame(
    {
        "Dataset": [
            "Training",
            "Testing",
        ],
        "Records": [
            80000,
            20000,
        ],
    }
)

fig_split = px.pie(
    split_df,
    names="Dataset",
    values="Records",
    hole=0.55,
    title="80 / 20 Train-Test Split",
)

style_plotly(
    fig_split,
    height=440,
)

st.plotly_chart(
    fig_split,
    width="stretch",
)

split_cols = st.columns(2)

with split_cols[0]:
    output_card(
        "Training Records",
        "80,000",
    )

with split_cols[1]:
    output_card(
        "Testing Records",
        "20,000",
    )

concept(
    "Why testing data?",
    (
        "The testing set represents unseen examples. "
        "The model does not learn from these records."
    ),
)


# ============================================================
# INTERACTIVE SPLIT
# ============================================================

with st.expander(
    "Try It Yourself — Change the Testing Percentage"
):

    test_percentage = st.slider(
        "Testing percentage",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
        key="ml_tutorial_test_percent",
    )

    calculated_test = int(
        len(df)
        * test_percentage
        / 100
    )

    calculated_train = (
        len(df)
        - calculated_test
    )

    c1, c2 = st.columns(2)

    with c1:
        st.metric(
            "Training",
            f"{calculated_train:,}",
        )

    with c2:
        st.metric(
            "Testing",
            f"{calculated_test:,}",
        )


# ============================================================
# STEP 18
# ============================================================

step(
    18,
    "Standardize the Features",
)

code_block(
    """
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
""",
    "python",
)

scale_example = pd.DataFrame(
    {
        "Feature": [
            "Previous GPA",
            "Attendance",
            "Study Hours",
        ],
        "Example Original Value": [
            3.40,
            92.00,
            4.00,
        ],
    }
)

show_table(
    scale_example
)

concept(
    "Why standardize?",
    (
        "Different predictors use different units and scales. "
        "Standardization places them on a comparable scale."
    ),
)


# ============================================================
# STEP 19
# ============================================================

step(
    19,
    "Build a Machine-Learning Pipeline",
)

code_block(
    """
pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LinearRegression()
    )
])
""",
    "python",
)

pipeline_cols = st.columns(
    [
        1,
        0.18,
        1,
        0.18,
        1,
        0.18,
        1,
    ]
)

pipeline_items = [
    "Raw Features",
    "arrow",
    "Median Imputation",
    "arrow",
    "Standardization",
    "arrow",
    "Model",
]

for col, item in zip(
    pipeline_cols,
    pipeline_items,
):

    with col:

        if item == "arrow":
            flow_arrow()

        else:
            flow_box(item)

concept(
    "Why use Pipeline?",
    (
        "The pipeline guarantees that the same preprocessing steps "
        "are used during training and prediction."
    ),
)


# ============================================================
# CHAPTER 3
# ============================================================

chapter(
    3,
    "Linear Regression",
    (
        "Linear Regression is used to predict the numerical exam score."
    ),
)


# ============================================================
# STEP 20
# ============================================================

step(
    20,
    "Understand Linear Regression",
)

concept(
    "Basic idea",
    (
        "Linear Regression learns coefficients that connect the input "
        "features to a numerical target."
    ),
)

st.latex(
    r"\hat{y} = b_0 + b_1x_1 + b_2x_2 + \cdots + b_nx_n"
)

st.caption(
    "ŷ = predicted exam score, x = input features, "
    "b = coefficients learned by the model."
)


# ============================================================
# STEP 21
# ============================================================

step(
    21,
    "Create the Linear Regression Pipeline",
)

code_block(
    """
linear_model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LinearRegression()
    )
])
""",
    "python",
)


# ============================================================
# STEP 22
# ============================================================

step(
    22,
    "Train the Linear Regression Model",
)

code_block(
    """
linear_model.fit(
    X_train,
    y_train
)
""",
    "python",
)

concept(
    "What does .fit() do?",
    (
        "The fit method learns the relationship between the six predictors "
        "and exam_score using the training data."
    ),
)


# ============================================================
# STEP 23
# ============================================================

step(
    23,
    "Predict Exam Scores",
)

code_block(
    """
y_pred = linear_model.predict(
    X_test
)
""",
    "python",
)

sample_predictions = pd.DataFrame(
    {
        "Actual Score":
            np.round(
                results[
                    "y_test_reg"
                ]
                .iloc[:10]
                .values,
                1,
            ),

        "Predicted Score":
            np.round(
                results[
                    "y_pred_reg"
                ][:10],
                1,
            ),
    }
)

show_table(
    sample_predictions
)


# ============================================================
# STEP 24
# ============================================================

step(
    24,
    "Compare Actual and Predicted Scores",
)

actual_predicted_df = pd.DataFrame(
    {
        "Actual Score":
            results[
                "y_test_reg"
            ]
            .iloc[:4000]
            .values,

        "Predicted Score":
            results[
                "y_pred_reg"
            ][:4000],
    }
)

fig_actual_predicted = px.scatter(
    actual_predicted_df,
    x="Actual Score",
    y="Predicted Score",
    opacity=0.35,
    title="Actual vs Predicted Exam Scores",
)

minimum = min(
    actual_predicted_df[
        "Actual Score"
    ].min(),
    actual_predicted_df[
        "Predicted Score"
    ].min(),
)

maximum = max(
    actual_predicted_df[
        "Actual Score"
    ].max(),
    actual_predicted_df[
        "Predicted Score"
    ].max(),
)

fig_actual_predicted.add_trace(
    go.Scatter(
        x=[
            minimum,
            maximum,
        ],
        y=[
            minimum,
            maximum,
        ],
        mode="lines",
        name="Perfect Prediction",
    )
)

style_plotly(
    fig_actual_predicted,
    height=500,
)

st.plotly_chart(
    fig_actual_predicted,
    width="stretch",
)

concept(
    "Reading this figure",
    (
        "Points close to the diagonal line have predicted scores "
        "close to the actual score."
    ),
)


# ============================================================
# STEP 25
# ============================================================

step(
    25,
    "Understand Residuals",
)

code_block(
    """
residual = (
    actual_score
    - predicted_score
)
""",
    "python",
)

residuals = (
    results[
        "y_test_reg"
    ].values
    - results[
        "y_pred_reg"
    ]
)

residual_df = pd.DataFrame(
    {
        "Predicted Score":
            results[
                "y_pred_reg"
            ][:4000],

        "Residual":
            residuals[:4000],
    }
)

fig_residual = px.scatter(
    residual_df,
    x="Predicted Score",
    y="Residual",
    opacity=0.35,
    title="Regression Residual Plot",
)

fig_residual.add_hline(
    y=0,
    line_dash="dash",
)

style_plotly(
    fig_residual,
    height=470,
)

st.plotly_chart(
    fig_residual,
    width="stretch",
)

concept(
    "Residual",
    (
        "A residual is the difference between the actual score "
        "and the predicted score."
    ),
)


# ============================================================
# STEP 26
# ============================================================

step(
    26,
    "Evaluate Linear Regression",
)

code_block(
    """
mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)
""",
    "python",
)

reg_metric_cols = st.columns(3)

with reg_metric_cols[0]:
    output_card(
        "MAE",
        f"{results['mae']:.2f}",
    )

with reg_metric_cols[1]:
    output_card(
        "RMSE",
        f"{results['rmse']:.2f}",
    )

with reg_metric_cols[2]:
    output_card(
        "R²",
        f"{results['r2']:.3f}",
    )

regression_metric_df = pd.DataFrame(
    {
        "Metric": [
            "MAE",
            "RMSE",
            "R²",
        ],
        "Simple Meaning": [
            "Average absolute prediction error",
            "Error metric that penalizes larger mistakes more strongly",
            "Fraction of target variation explained by the model",
        ],
    }
)

show_table(
    regression_metric_df
)

concept(
    "Project interpretation",
    (
        f"The model has MAE = {results['mae']:.2f}, "
        f"RMSE = {results['rmse']:.2f}, and "
        f"R² = {results['r2']:.3f}."
    ),
)


# ============================================================
# R2 VISUAL
# ============================================================

explained_percent = max(
    0,
    min(
        100,
        results[
            "r2"
        ] * 100,
    ),
)

r2_visual_df = pd.DataFrame(
    {
        "Component": [
            "Explained by Model",
            "Unexplained",
        ],
        "Percent": [
            explained_percent,
            100 - explained_percent,
        ],
    }
)

fig_r2 = px.bar(
    r2_visual_df,
    x="Percent",
    y="Component",
    orientation="h",
    range_x=[0, 100],
    title="Explained vs Unexplained Variation",
)

style_plotly(
    fig_r2,
    height=360,
)

st.plotly_chart(
    fig_r2,
    width="stretch",
)


# ============================================================
# STEP 27
# ============================================================

step(
    27,
    "Inspect Regression Coefficients",
)

coefficients = (
    results[
        "linear_pipeline"
    ]
    .named_steps["model"]
    .coef_
)

coefficient_df = pd.DataFrame(
    {
        "Feature": [
            FEATURE_LABELS[f]
            for f in FEATURES
        ],
        "Coefficient":
            coefficients,
    }
).sort_values(
    "Coefficient"
)

fig_coefficients = px.bar(
    coefficient_df,
    x="Coefficient",
    y="Feature",
    orientation="h",
    title="Standardized Linear Regression Coefficients",
)

style_plotly(
    fig_coefficients,
    height=460,
)

st.plotly_chart(
    fig_coefficients,
    width="stretch",
)

st.warning(
    "Coefficients represent associations inside this fitted model. "
    "They should not be interpreted as proof of causation."
)


# ============================================================
# CHAPTER 4
# ============================================================

chapter(
    4,
    "Logistic Regression",
    (
        "Logistic Regression is used to estimate the probability "
        "that a student belongs to the Pass or Fail class."
    ),
)


# ============================================================
# STEP 28
# ============================================================

step(
    28,
    "Regression vs Classification",
)

comparison_df = pd.DataFrame(
    {
        "Property": [
            "Prediction Type",
            "Project Target",
            "Example Output",
            "Main Metrics",
        ],
        "Linear Regression": [
            "Continuous number",
            "exam_score",
            "72.4",
            "MAE, RMSE, R²",
        ],
        "Logistic Regression": [
            "Class / probability",
            "pass_status",
            "Pass, 78%",
            "Accuracy, Precision, Recall, F1, Balanced Accuracy, AUC",
        ],
    }
)

show_table(
    comparison_df
)


# ============================================================
# STEP 29
# ============================================================

step(
    29,
    "Understand Classification Probability",
)

example_probability = st.slider(
    "Example Pass Probability",
    min_value=0,
    max_value=100,
    value=78,
    key="tutorial_probability_slider",
)

example_status = (
    "PASS"
    if example_probability >= 50
    else "FAIL"
)

st.progress(
    example_probability / 100
)

prob_cols = st.columns(2)

with prob_cols[0]:
    output_card(
        "Pass Probability",
        f"{example_probability}%",
    )

with prob_cols[1]:
    output_card(
        "Predicted Class",
        example_status,
    )

concept(
    "Classification threshold",
    (
        "A probability threshold converts a probability into a class. "
        "With a 50% threshold, values at or above 50% are predicted as Pass."
    ),
)


# ============================================================
# STEP 30
# ============================================================

step(
    30,
    "Build Balanced Logistic Regression",
)

code_block(
    """
logistic_model = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )
    )
])
""",
    "python",
)

concept(
    "Why class_weight='balanced'?",
    (
        "The dataset contains more Pass examples than Fail examples. "
        "Balanced class weighting gives the minority class more importance during training."
    ),
)


# ============================================================
# STEP 31
# ============================================================

step(
    31,
    "Train the Logistic Regression Model",
)

code_block(
    """
logistic_model.fit(
    X_train,
    y_train
)
""",
    "python",
)


# ============================================================
# STEP 32
# ============================================================

step(
    32,
    "Generate Classes and Probabilities",
)

code_block(
    """
y_pred = logistic_model.predict(
    X_test
)

y_probability = logistic_model.predict_proba(
    X_test
)
""",
    "python",
)

classification_output_df = pd.DataFrame(
    {
        "Method": [
            "predict()",
            "predict_proba()",
        ],
        "Output": [
            "Pass or Fail",
            "Probability for each class",
        ],
    }
)

show_table(
    classification_output_df
)


# ============================================================
# STEP 33
# ============================================================

step(
    33,
    "Understand the Confusion Matrix",
)

cm = results["cm"]

fig_cm = go.Figure(
    go.Heatmap(
        z=cm,
        x=[
            "Predicted Fail",
            "Predicted Pass",
        ],
        y=[
            "Actual Fail",
            "Actual Pass",
        ],
        text=cm,
        texttemplate="%{text}",
        showscale=False,
    )
)

fig_cm.update_layout(
    title="Confusion Matrix"
)

style_plotly(
    fig_cm,
    height=450,
)

st.plotly_chart(
    fig_cm,
    width="stretch",
)

confusion_df = pd.DataFrame(
    {
        "Term": [
            "True Positive",
            "True Negative",
            "False Positive",
            "False Negative",
        ],
        "Meaning": [
            "Actual Pass predicted as Pass",
            "Actual Fail predicted as Fail",
            "Actual Fail predicted as Pass",
            "Actual Pass predicted as Fail",
        ],
    }
)

show_table(
    confusion_df
)


# ============================================================
# STEP 34
# ============================================================

step(
    34,
    "Evaluate Logistic Regression",
)

classification_metric_cols = st.columns(3)

with classification_metric_cols[0]:
    output_card(
        "Accuracy",
        f"{results['accuracy'] * 100:.1f}%",
    )

with classification_metric_cols[1]:
    output_card(
        "Precision",
        f"{results['precision'] * 100:.1f}%",
    )

with classification_metric_cols[2]:
    output_card(
        "Recall",
        f"{results['recall'] * 100:.1f}%",
    )

classification_metric_cols2 = st.columns(3)

with classification_metric_cols2[0]:
    output_card(
        "F1 Score",
        f"{results['f1'] * 100:.1f}%",
    )

with classification_metric_cols2[1]:
    output_card(
        "Balanced Accuracy",
        f"{results['balanced_accuracy'] * 100:.1f}%",
    )

with classification_metric_cols2[2]:
    output_card(
        "AUC",
        f"{results['auc']:.3f}",
    )

classification_metrics_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy",
            "AUC",
        ],
        "Value": [
            results["accuracy"] * 100,
            results["precision"] * 100,
            results["recall"] * 100,
            results["f1"] * 100,
            results["balanced_accuracy"] * 100,
            results["auc"] * 100,
        ],
    }
)

fig_class_metrics = px.bar(
    classification_metrics_df,
    x="Value",
    y="Metric",
    orientation="h",
    range_x=[0, 100],
    title="Balanced Logistic Regression Metrics",
)

style_plotly(
    fig_class_metrics,
    height=460,
)

st.plotly_chart(
    fig_class_metrics,
    width="stretch",
)


# ============================================================
# METRIC MEANINGS
# ============================================================

metric_meaning_df = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Balanced Accuracy",
            "AUC",
        ],
        "Beginner Meaning": [
            "How often all predictions are correct",
            "When the model predicts Pass, how often it is right",
            "How many actual Pass students are identified",
            "Balance between precision and recall",
            "Performance that gives both classes equal importance",
            "How well the model ranks Pass above Fail",
        ],
    }
)

show_table(
    metric_meaning_df
)


# ============================================================
# STEP 35
# ============================================================

step(
    35,
    "Why Accuracy Alone Can Be Misleading",
)

concept(
    "Majority-class problem",
    (
        "Because most students are Pass, a very simple classifier "
        "could predict Pass almost every time and still achieve high accuracy."
    ),
)

model_comparison = pd.DataFrame(
    {
        "Model": [
            "Dummy",
            "Standard Logistic",
            "Balanced Logistic",
        ],
        "Accuracy": [
            results["dummy_accuracy"] * 100,
            results["standard_accuracy"] * 100,
            results["accuracy"] * 100,
        ],
        "Balanced Accuracy": [
            results["dummy_balanced"] * 100,
            results["standard_balanced"] * 100,
            results["balanced_accuracy"] * 100,
        ],
    }
)

comparison_long = (
    model_comparison.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Percent",
    )
)

fig_compare = px.bar(
    comparison_long,
    x="Model",
    y="Percent",
    color="Metric",
    barmode="group",
    range_y=[0, 100],
    title="Dummy vs Standard vs Balanced Logistic Regression",
)

style_plotly(
    fig_compare,
    height=480,
)

st.plotly_chart(
    fig_compare,
    width="stretch",
)

show_table(
    model_comparison.round(1)
)

concept(
    "Why balanced accuracy matters",
    (
        "Balanced accuracy treats both classes more equally, "
        "so it is useful when Pass and Fail are not equally represented."
    ),
)


# ============================================================
# STEP 36
# ============================================================

step(
    36,
    "Understand the ROC Curve and AUC",
)

fig_roc = go.Figure()

fig_roc.add_trace(
    go.Scatter(
        x=results["fpr"],
        y=results["tpr"],
        mode="lines",
        name=(
            f"Balanced Logistic "
            f"(AUC = {results['auc']:.3f})"
        ),
    )
)

fig_roc.add_trace(
    go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(
            dash="dash"
        ),
    )
)

fig_roc.update_layout(
    title="ROC Curve",
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
)

style_plotly(
    fig_roc,
    height=500,
)

st.plotly_chart(
    fig_roc,
    width="stretch",
)

concept(
    "AUC",
    (
        "AUC summarizes how well the model separates Pass and Fail "
        "across many possible probability thresholds."
    ),
)


# ============================================================
# CHAPTER 5
# ============================================================

chapter(
    5,
    "Making a Prediction for a New Student",
    (
        "The final step is to use the fitted models on a new student profile."
    ),
)


# ============================================================
# STEP 37
# ============================================================

step(
    37,
    "Create One New Student",
)

code_block(
    """
student = pd.DataFrame([{
    "previous_exam_score": 70,
    "previous_gpa": 3.2,
    "attendance_percentage": 85,
    "assignment_completion_rate": 90,
    "study_hours_per_day": 3,
    "practice_tests_completed": 5
}])
""",
    "python",
)


# ============================================================
# STEP 38
# ============================================================

step(
    38,
    "Run Both Models",
)

code_block(
    """
predicted_score = (
    linear_model.predict(
        student
    )[0]
)

predicted_status = (
    logistic_model.predict(
        student
    )[0]
)

probability = (
    logistic_model.predict_proba(
        student
    )
)
""",
    "python",
)

prediction_flow_cols = st.columns(
    [
        1,
        0.18,
        1,
        0.18,
        1,
    ]
)

prediction_flow_items = [
    "Student Inputs",
    "arrow",
    "Preprocessing Pipeline",
    "arrow",
    "ML Prediction",
]

for col, item in zip(
    prediction_flow_cols,
    prediction_flow_items,
):

    with col:

        if item == "arrow":
            flow_arrow()

        else:
            flow_box(item)


# ============================================================
# INTERACTIVE PREDICTION LAB
# ============================================================

section_title(
    "sliders",
    "Interactive Student Prediction Lab",
)

st.caption(
    "Change the six feature values below. "
    "Both trained models will immediately evaluate the profile."
)

input_col1, input_col2 = st.columns(2)

with input_col1:

    previous_exam = st.number_input(
        "Previous Exam Score",
        min_value=0.0,
        max_value=100.0,
        value=70.0,
        step=1.0,
        key="learn_previous_exam",
    )

    previous_gpa = st.number_input(
        "Previous GPA",
        min_value=0.0,
        max_value=4.0,
        value=3.2,
        step=0.1,
        key="learn_previous_gpa",
    )

    attendance = st.number_input(
        "Attendance Percentage",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=1.0,
        key="learn_attendance",
    )


with input_col2:

    assignment_completion = st.number_input(
        "Assignment Completion Rate",
        min_value=0.0,
        max_value=100.0,
        value=90.0,
        step=1.0,
        key="learn_assignment",
    )

    study_hours = st.number_input(
        "Study Hours per Day",
        min_value=0.0,
        max_value=12.0,
        value=3.0,
        step=0.5,
        key="learn_study_hours",
    )

    practice_tests = st.number_input(
        "Practice Tests Completed",
        min_value=0,
        max_value=50,
        value=5,
        step=1,
        key="learn_practice_tests",
    )


new_student = pd.DataFrame(
    [
        {
            "previous_exam_score":
                previous_exam,

            "previous_gpa":
                previous_gpa,

            "attendance_percentage":
                attendance,

            "assignment_completion_rate":
                assignment_completion,

            "study_hours_per_day":
                study_hours,

            "practice_tests_completed":
                practice_tests,
        }
    ]
)


new_predicted_score = (
    results[
        "linear_pipeline"
    ]
    .predict(
        new_student
    )[0]
)


new_predicted_status = str(
    results[
        "balanced_logistic"
    ]
    .predict(
        new_student
    )[0]
)


classes = list(
    results[
        "balanced_logistic"
    ]
    .named_steps["model"]
    .classes_
)

pass_index = classes.index("Pass")

new_pass_probability = (
    results[
        "balanced_logistic"
    ]
    .predict_proba(
        new_student
    )[0][pass_index]
)


prediction_cols = st.columns(3)

with prediction_cols[0]:
    output_card(
        "Predicted Exam Score",
        f"{new_predicted_score:.1f}",
    )

with prediction_cols[1]:
    output_card(
        "Predicted Status",
        new_predicted_status,
    )

with prediction_cols[2]:
    output_card(
        "Pass Probability",
        f"{new_pass_probability * 100:.1f}%",
    )

st.progress(
    float(
        np.clip(
            new_pass_probability,
            0,
            1,
        )
    )
)

concept(
    "Important interpretation",
    (
        "The exam-score prediction comes from Linear Regression. "
        "The Pass/Fail class and probability come from Logistic Regression. "
        "They are separate models."
    ),
)


# ============================================================
# COMPLETE ML SUMMARY
# ============================================================

chapter(
    "Final",
    "Complete Machine-Learning Workflow",
    (
        "You have now followed the entire ML process used in "
        "Student Performance Analytics."
    ),
)

final_summary = pd.DataFrame(
    {
        "Stage": [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
        ],
        "Machine-Learning Process": [
            "Load the dataset",
            "Explore distributions and data quality",
            "Handle missing values",
            "Select six predictors",
            "Define regression and classification targets",
            "Split training and testing data",
            "Standardize predictors",
            "Train Linear Regression",
            "Train and evaluate Logistic Regression",
            "Predict a new student",
        ],
    }
)

show_table(
    final_summary
)


# ============================================================
# KNOWLEDGE CHECK
# ============================================================

section_title(
    "check",
    "Knowledge Check",
)

with st.expander(
    "1. Why should testing data not be used during training?"
):

    st.write(
        "Because testing data should represent unseen examples. "
        "It is used to measure generalization."
    )


with st.expander(
    "2. What is target leakage?"
):

    st.write(
        "Target leakage occurs when information that reveals the target "
        "is accidentally included in the input features."
    )


with st.expander(
    "3. Why do we use median imputation?"
):

    st.write(
        "Median imputation keeps records usable when a numeric predictor is missing "
        "and is relatively resistant to extreme values."
    )


with st.expander(
    "4. Why use a preprocessing pipeline?"
):

    st.write(
        "A pipeline ensures the same transformations are applied during training "
        "and prediction."
    )


with st.expander(
    "5. What is the difference between Linear and Logistic Regression?"
):

    st.write(
        "Linear Regression predicts a continuous number. "
        "Logistic Regression predicts class probabilities and categories."
    )


with st.expander(
    "6. Why can accuracy be misleading?"
):

    st.write(
        "Because the dataset is imbalanced. "
        "A model can obtain good accuracy by favoring the majority class."
    )


with st.expander(
    "7. Why is balanced accuracy useful?"
):

    st.write(
        "Balanced accuracy gives equal importance to performance on both classes."
    )


# ============================================================
# FINAL NOTE
# ============================================================

st.warning(
    "These models demonstrate machine-learning concepts. "
    "Their outputs are statistical predictions, not guaranteed student outcomes "
    "and not evidence of causation."
)


# ============================================================
# FOOTER
# ============================================================

show_footer()