# ============================================================
# DATASET & EDA PAGE
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from components.styles import apply_global_styles
from components.footer import show_footer


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Dataset & EDA",
    page_icon="📊",
    layout="wide"
)

apply_global_styles()


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()


# ------------------------------------------------------------
# FINAL MODEL VARIABLES
# ------------------------------------------------------------

model_features = [
    "previous_exam_score",
    "previous_gpa",
    "attendance_percentage",
    "assignment_completion_rate",
    "study_hours_per_day",
    "practice_tests_completed"
]

analysis_columns = model_features + [
    "exam_score"
]


# ------------------------------------------------------------
# DISPLAY LABELS
# ------------------------------------------------------------

feature_labels = {
    "previous_exam_score": "Previous Exam Score",
    "previous_gpa": "Previous GPA",
    "attendance_percentage": "Attendance",
    "assignment_completion_rate": "Assignment Completion",
    "study_hours_per_day": "Study Hours per Day",
    "practice_tests_completed": "Practice Tests",
    "exam_score": "Exam Score",
    "pass_status": "Pass Status"
}


def friendly_name(column_name):
    return feature_labels.get(
        column_name,
        column_name.replace("_", " ").title()
    )


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.title("📊 Dataset & EDA")

st.caption(
    "Explore the dataset, selected prediction features, distributions, "
    "relationships, and correlations."
)


# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Distributions",
    "Relationships",
    "Correlations"
])


# ============================================================
# OVERVIEW
# ============================================================

with tab1:

    # --------------------------------------------------------
    # SUMMARY METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{len(df):,}"
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing Values",
        f"{int(df.isnull().sum().sum()):,}"
    )

    col4.metric(
        "Duplicates",
        int(df.duplicated().sum())
    )


    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(12),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MODEL VARIABLES
    # --------------------------------------------------------

    st.subheader("Selected Model Variables")

    selected_df = df[
        analysis_columns + ["pass_status"]
    ]

    summary_df = selected_df[
        analysis_columns
    ].describe().T.reset_index()

    summary_df = summary_df[
        [
            "index",
            "count",
            "mean",
            "std",
            "min",
            "50%",
            "max"
        ]
    ]

    summary_df.columns = [
        "Variable",
        "Count",
        "Mean",
        "Std Dev",
        "Min",
        "Median",
        "Max"
    ]

    summary_df["Variable"] = (
        summary_df["Variable"]
        .map(friendly_name)
    )

    st.dataframe(
        summary_df.round(2),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------------

    st.subheader("Missing Values in Model Variables")

    missing_model = (
        selected_df
        .isnull()
        .sum()
        .sort_values(ascending=True)
    )

    missing_model = missing_model[
        missing_model > 0
    ]

    if len(missing_model) > 0:

        missing_df = pd.DataFrame({
            "Variable": missing_model.index,
            "Missing Values": missing_model.values
        })

        missing_df["Variable"] = (
            missing_df["Variable"]
            .map(friendly_name)
        )

        fig_missing = px.bar(
            missing_df,
            x="Missing Values",
            y="Variable",
            orientation="h",
            text="Missing Values"
        )

        fig_missing.update_traces(
            textposition="outside"
        )

        fig_missing.update_layout(
            height=280,
            margin=dict(
                l=20,
                r=50,
                t=10,
                b=20
            ),
            xaxis_title="Missing Values",
            yaxis_title="",
            showlegend=False
        )

        st.plotly_chart(
            fig_missing,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )

    else:

        st.success(
            "No missing values in the selected model variables."
        )


# ============================================================
# DISTRIBUTIONS
# ============================================================

with tab2:

    st.subheader("Variable Distribution")

    variable = st.selectbox(
        "Select a variable",
        analysis_columns,
        format_func=friendly_name
    )


    fig = px.histogram(
        df,
        x=variable,
        nbins=35,
        marginal="box"
    )

    fig.update_layout(
        height=380,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        xaxis_title=friendly_name(variable),
        yaxis_title="Students",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # --------------------------------------------------------
    # PASS FAIL DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("Pass / Fail Distribution")

    class_counts = (
        df["pass_status"]
        .value_counts()
        .reset_index()
    )

    class_counts.columns = [
        "Status",
        "Students"
    ]


    dist_col1, dist_col2 = st.columns(
        [1, 1]
    )


    with dist_col1:

        fig_class = px.pie(
            class_counts,
            names="Status",
            values="Students",
            hole=0.60
        )

        fig_class.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig_class.update_layout(
            height=300,
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10
            ),
            showlegend=False
        )

        st.plotly_chart(
            fig_class,
            use_container_width=True,
            config={
                "displayModeBar": False
            }
        )


    with dist_col2:

        pass_count = int(
            class_counts.loc[
                class_counts["Status"] == "Pass",
                "Students"
            ].sum()
        )

        fail_count = int(
            class_counts.loc[
                class_counts["Status"] == "Fail",
                "Students"
            ].sum()
        )

        pass_pct = (
            pass_count / len(df) * 100
        )

        fail_pct = (
            fail_count / len(df) * 100
        )

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Pass",
            f"{pass_count:,}",
            f"{pass_pct:.1f}%"
        )

        metric2.metric(
            "Fail",
            f"{fail_count:,}",
            f"{fail_pct:.1f}%"
        )

        st.caption(
            "The target is imbalanced, with substantially more "
            "Pass observations than Fail observations."
        )


# ============================================================
# RELATIONSHIPS
# ============================================================

with tab3:

    st.subheader(
        "Relationship Between Variables"
    )


    col1, col2 = st.columns(2)


    with col1:

        x_var = st.selectbox(
            "X variable",
            model_features,
            index=0,
            format_func=friendly_name
        )


    with col2:

        y_options = [
            "exam_score"
        ] + model_features

        y_var = st.selectbox(
            "Y variable",
            y_options,
            index=0,
            format_func=friendly_name
        )


    plot_df = df[
        [x_var, y_var]
    ].dropna()


    # Sample for performance and readability
    if len(plot_df) > 15000:

        plot_df = plot_df.sample(
            15000,
            random_state=42
        )


    fig_scatter = px.scatter(
        plot_df,
        x=x_var,
        y=y_var,
        opacity=0.30,
        trendline="ols"
    )


    fig_scatter.update_layout(
        height=420,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        xaxis_title=friendly_name(x_var),
        yaxis_title=friendly_name(y_var)
    )


    st.plotly_chart(
        fig_scatter,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    relationship_corr = (
        df[
            [x_var, y_var]
        ]
        .corr()
        .iloc[0, 1]
    )


    metric_col1, metric_col2, metric_col3 = (
        st.columns([1, 1, 2])
    )


    with metric_col1:

        st.metric(
            "Pearson r",
            f"{relationship_corr:.3f}"
        )


    with metric_col2:

        strength = abs(
            relationship_corr
        )

        if strength >= 0.7:
            interpretation = "Strong"
        elif strength >= 0.4:
            interpretation = "Moderate"
        elif strength >= 0.2:
            interpretation = "Weak"
        else:
            interpretation = "Very Weak"

        st.metric(
            "Relationship",
            interpretation
        )


    with metric_col3:

        direction = (
            "positive"
            if relationship_corr > 0
            else "negative"
            if relationship_corr < 0
            else "no linear"
        )

        st.caption(
            f"The selected variables show a "
            f"{interpretation.lower()} {direction} linear relationship."
        )


# ============================================================
# CORRELATIONS
# ============================================================

with tab4:

    st.subheader(
        "Correlation Matrix"
    )


    corr_matrix = (
        df[analysis_columns]
        .corr()
        .round(2)
    )


    display_corr = corr_matrix.copy()

    display_corr.index = [
        friendly_name(col)
        for col in display_corr.index
    ]

    display_corr.columns = [
        friendly_name(col)
        for col in display_corr.columns
    ]


    fig_heatmap = px.imshow(
        display_corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1
    )


    fig_heatmap.update_layout(
        height=520,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),
        coloraxis_colorbar=dict(
            title="Correlation"
        )
    )


    st.plotly_chart(
        fig_heatmap,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


    # --------------------------------------------------------
    # CORRELATION WITH EXAM SCORE
    # --------------------------------------------------------

    st.subheader(
        "Selected Features vs. Exam Score"
    )

    st.caption(
        "Pearson correlation with the final exam score."
    )


    exam_corr = (
        corr_matrix["exam_score"]
        .drop("exam_score")
        .sort_values(
            ascending=True
        )
        .reset_index()
    )


    exam_corr.columns = [
        "Feature",
        "Correlation"
    ]


    exam_corr["Feature"] = (
        exam_corr["Feature"]
        .map(friendly_name)
    )


    fig_corr = px.bar(
        exam_corr,
        x="Correlation",
        y="Feature",
        orientation="h",
        text="Correlation"
    )


    fig_corr.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside"
    )


    fig_corr.update_layout(
        height=320,
        margin=dict(
            l=20,
            r=60,
            t=10,
            b=20
        ),
        xaxis_title="Correlation",
        yaxis_title="",
        showlegend=False
    )


    st.plotly_chart(
        fig_corr,
        use_container_width=True,
        config={
            "displayModeBar": False
        }
    )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

show_footer()