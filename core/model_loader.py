# ============================================================
# MODEL LOADER
# ============================================================

from pathlib import Path
import joblib
import streamlit as st


# Find the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model file locations
LINEAR_MODEL_PATH = BASE_DIR / "models" / "linear_pipeline.pkl"
LOGISTIC_MODEL_PATH = BASE_DIR / "models" / "logistic_pipeline.pkl"


@st.cache_resource
def load_linear_model():
    """
    Load the trained Linear Regression pipeline.
    """
    return joblib.load(LINEAR_MODEL_PATH)


@st.cache_resource
def load_logistic_model():
    """
    Load the trained Balanced Logistic Regression pipeline.
    """
    return joblib.load(LOGISTIC_MODEL_PATH)