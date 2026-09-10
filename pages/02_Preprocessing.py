# ============================================================
# DATA PREPROCESSING PAGE
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

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


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Data Preprocessing",
    page_icon="⚙️",
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
        classes="preprocess-table"
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

    .preprocess-table-container {{
        {container_style}
        border: 1px solid {chart_border};
        border-radius: 10px;
        background: {surface};
        margin-bottom: 14px;
    }}

    .preprocess-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 13px;
        color: {chart_text};
        background: {surface};
    }}

    .preprocess-table thead th {{
        text-align: left;
        padding: 10px 11px;
        font-weight: 600;
        color: {chart_muted};
        background: {surface};
        border-bottom: 1px solid {chart_border};
        white-space: nowrap;
        position: sticky;
        top: 0;
        z-index: 1;
    }}

    .preprocess-table tbody td {{
        padding: 9px 11px;
        color: {chart_text};
        background: {surface};
        border-bottom: 1px solid {chart_border};
        white-space: nowrap;
    }}

    .preprocess-table tbody tr:last-child td {{
        border-bottom: none;
    }}

    </style>

    <div class="preprocess-table-container">
        {table_html}
    </div>
    """

    st.html(
        html
    )


# ------------------------------------------------------------
# MODEL CONFIGURATION
# ------------------------------------------------------------

features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed"
]

regression_target = "exam_score"
classification_target = "pass_status"

model_columns = features + [
    regression_target,
    classification_target
]


# ------------------------------------------------------------
# DISPLAY LABELS
# ------------------------------------------------------------

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
        "Pass / Fail"
}


def friendly_name(column):

    return feature_labels.get(
        column,
        column
        .replace("_", " ")
        .title()
    )


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

page_title(
    "settings",
    "Data Preprocessing",
    "Transform the raw dataset into model-ready training "
    "and testing data."
)


# ------------------------------------------------------------
# PREPROCESSING WORKFLOW
# ------------------------------------------------------------

section_title(
    "settings",
    "Preprocessing Workflow"
)


workflow = [
    (
        "database",
        "Raw Data",
        "Original student dataset"
    ),
    (
        "filter",
        "Select Features",
        "Six prediction variables"
    ),
    (
        "wrench",
        "Impute Missing",
        "Median imputation"
    ),
    (
        "split",
        "Train / Test",
        "80% / 20% split"
    ),
    (
        "sliders",
        "Standardize",
        "Scale numerical inputs"
    ),
    (
        "brain",
        "Model Ready",
        "Ready for training"
    )
]


workflow_cols = st.columns(6)


for col, (
    icon_name,
    title,
    description
) in zip(
    workflow_cols,
    workflow
):

    with col:

        icon_card(
            icon_name,
            title,
            description
        )


# ------------------------------------------------------------
# DATASET STATUS
# ------------------------------------------------------------

section_title(
    "database",
    "Dataset Status"
)


c1, c2, c3, c4 = st.columns(4)


c1.metric(
    "Records",
    f"{len(df):,}"
)

c2.metric(
    "Variables",
    df.shape[1]
)

c3.metric(
    "Missing Values",
    f"{int(df.isna().sum().sum()):,}"
)

c4.metric(
    "Duplicates",
    f"{int(df.duplicated().sum()):,}"
)


# ------------------------------------------------------------
# MODEL VARIABLES
# ------------------------------------------------------------

section_title(
    "database",
    "Model Variables"
)


feature_col, target_col = st.columns(
    [1.4, 1]
)


# ------------------------------------------------------------
# INPUT FEATURES
# ------------------------------------------------------------

with feature_col:

    with st.container(
        border=True
    ):

        st.markdown(
            "#### Prediction Features"
        )

        f1, f2 = st.columns(2)


        with f1:

            st.markdown(
                """
                **Previous Exam Score**  
                **Previous GPA**  
                **Attendance**
                """
            )


        with f2:

            st.markdown(
                """
                **Assignment Completion**  
                **Study Hours**  
                **Practice Tests**
                """
            )


        st.caption(
            "All six inputs represent information "
            "available before the predicted exam outcome."
        )


# ------------------------------------------------------------
# TARGET VARIABLES
# ------------------------------------------------------------

with target_col:

    with st.container(
        border=True
    ):

        st.markdown(
            "#### Prediction Targets"
        )


        st.markdown(
            "**Linear Regression**"
        )

        st.caption(
            "Exam Score · numerical prediction"
        )


        st.markdown(
            "**Balanced Logistic Regression**"
        )

        st.caption(
            "Pass / Fail · classification"
        )


# ------------------------------------------------------------
# MISSING VALUES
# ------------------------------------------------------------

section_title(
    "wrench",
    "Missing-Value Handling"
)


selected_missing = (
    df[model_columns]
    .isna()
    .sum()
    .reset_index()
)


selected_missing.columns = [
    "Variable",
    "Missing Values"
]


selected_missing["Variable"] = (
    selected_missing["Variable"]
    .map(
        friendly_name
    )
)


missing_plot = (
    selected_missing[
        selected_missing[
            "Missing Values"
        ] > 0
    ]
    .copy()
)


if not missing_plot.empty:

    fig_missing = px.bar(

        missing_plot.sort_values(
            "Missing Values",
            ascending=True
        ),

        x="Missing Values",
        y="Variable",
        orientation="h",
        text="Missing Values"
    )


    fig_missing.update_traces(
        marker_color=accent,
        textposition="outside"
    )


    fig_missing.update_layout(

        height=280,

        margin=dict(
            l=20,
            r=60,
            t=10,
            b=20
        ),

        xaxis_title="Missing Values",
        yaxis_title="",

        showlegend=False
    )


    style_chart(
        fig_missing
    )


    st.plotly_chart(
        fig_missing,
        width="stretch",
        theme=None,

        config={
            "displayModeBar": False
        }
    )


else:

    st.success(
        "No missing values in the selected model variables."
    )


st.info(
    "Missing numerical values are handled using median "
    "imputation inside the model pipeline."
)


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

section_title(
    "split",
    "Train / Test Split"
)


train_size = int(
    len(df) * 0.80
)

test_size = (
    len(df)
    - train_size
)


split_col1, split_col2 = st.columns(
    [1, 1.3]
)


# ------------------------------------------------------------
# SPLIT METRICS
# ------------------------------------------------------------

with split_col1:

    train_metric, test_metric = (
        st.columns(2)
    )


    train_metric.metric(
        "Training",
        f"{train_size:,}",
        "80%"
    )


    test_metric.metric(
        "Testing",
        f"{test_size:,}",
        "20%"
    )


    st.metric(
        "Random State",
        "42"
    )


# ------------------------------------------------------------
# SPLIT DONUT
# ------------------------------------------------------------

with split_col2:

    split_df = pd.DataFrame(
        {
            "Dataset": [
                "Training",
                "Testing"
            ],
            "Records": [
                train_size,
                test_size
            ]
        }
    )


    fig_split = px.pie(
        split_df,
        names="Dataset",
        values="Records",
        hole=0.62
    )


    fig_split.update_traces(
        textinfo="percent+label",
        textposition="inside"
    )


    fig_split.update_layout(

        template=plotly_template,

        height=260,

        margin=dict(
            l=10,
            r=10,
            t=5,
            b=5
        ),

        showlegend=False,

        paper_bgcolor=chart_background,
        plot_bgcolor=chart_background,

        font=dict(
            color=chart_text
        )
    )


    st.plotly_chart(
        fig_split,
        width="stretch",
        theme=None,

        config={
            "displayModeBar": False
        }
    )


# ------------------------------------------------------------
# FEATURE TRANSFORMATION
# ------------------------------------------------------------

section_title(
    "sliders",
    "Feature Transformation"
)


transform1, arrow, transform2 = (
    st.columns(
        [1, 0.2, 1]
    )
)


# ------------------------------------------------------------
# MEDIAN IMPUTATION
# ------------------------------------------------------------

with transform1:

    icon_card(
        "wrench",
        "Median Imputation",
        "Missing numerical values are replaced using "
        "the median learned from the training data."
    )


# ------------------------------------------------------------
# ARROW
# ------------------------------------------------------------

with arrow:

    st.markdown(
        f"""
        <div style="
            text-align:center;
            font-size:30px;
            padding-top:45px;
            color:{chart_muted};
        ">
            →
        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# STANDARDIZATION
# ------------------------------------------------------------

with transform2:

    icon_card(
        "sliders",
        "Standardization",
        "Numerical inputs are centered and scaled before "
        "being passed to the models."
    )


# ------------------------------------------------------------
# STANDARDIZATION FORMULA
# ------------------------------------------------------------

st.markdown(
    f"""
    <div style="
        text-align:center;
        padding:14px;
        margin-top:10px;
        margin-bottom:8px;
        border:1px solid {chart_border};
        border-radius:10px;
        background:{surface};
        font-size:17px;
        color:{chart_text};
    ">
        Standardized value =
        (value − training mean) ÷ training standard deviation
    </div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# MODEL PIPELINES
# ------------------------------------------------------------

section_title(
    "brain",
    "Model Pipelines"
)


pipeline1, pipeline2 = st.columns(2)


# ------------------------------------------------------------
# REGRESSION PIPELINE
# ------------------------------------------------------------

with pipeline1:

    with st.container(
        border=True
    ):

        st.markdown(
            "### Linear Regression"
        )

        st.markdown(
            """
            **6 Inputs**  
            ↓  
            **Median Imputation**  
            ↓  
            **Standardization**  
            ↓  
            **Linear Regression**  
            ↓  
            **Exam Score**
            """
        )


# ------------------------------------------------------------
# CLASSIFICATION PIPELINE
# ------------------------------------------------------------

with pipeline2:

    with st.container(
        border=True
    ):

        st.markdown(
            "### Balanced Logistic Regression"
        )

        st.markdown(
            """
            **6 Inputs**  
            ↓  
            **Median Imputation**  
            ↓  
            **Standardization**  
            ↓  
            **Balanced Logistic Regression**  
            ↓  
            **Pass / Fail Probability**
            """
        )


# ------------------------------------------------------------
# DATA LEAKAGE
# ------------------------------------------------------------

section_title(
    "check",
    "Data Leakage Protection"
)


leak1, leak2 = st.columns(2)


with leak1:

    icon_card(
        "check",
        "Pre-Exam Inputs",
        "The exam score being predicted is not included "
        "among the six predictor variables."
    )


with leak2:

    icon_card(
        "check",
        "Training-Only Fitting",
        "Imputation and scaling parameters are learned "
        "from the training set before being applied "
        "to the test set."
    )


# ------------------------------------------------------------
# MODEL READY DATA
# ------------------------------------------------------------

section_title(
    "database",
    "Model-Ready Structure"
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
    "The live prediction page uses the same six-feature "
    "structure as the trained models."
)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

show_footer()