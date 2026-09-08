from pathlib import Path
import pandas as pd
import streamlit as st

DATA_PATH = Path("data/student_performance.csv")

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)
