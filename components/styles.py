import streamlit as st

def apply_global_styles():
    st.markdown("""
    <style>
    .stApp { background: #F7F8FA; color: #1F2937; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #E5E7EB; }
    .block-container { max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem; }
    .metric-card { background:#FFF; border:1px solid #E5E7EB; border-radius:14px; padding:18px 20px; min-height:118px; }
    .metric-label { color:#6B7280; font-size:.88rem; margin-bottom:7px; }
    .metric-value { color:#1F2937; font-size:1.65rem; font-weight:700; line-height:1.1; }
    .metric-note { color:#6B7280; font-size:.78rem; margin-top:8px; }
    .feature-chip { background:#FFF; border:1px solid #E5E7EB; border-radius:10px; padding:12px 14px; margin:5px 0; font-size:.92rem; }
    .workflow { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:12px 0 24px 0; }
    .workflow-step { background:#FFF; border:1px solid #E5E7EB; border-radius:10px; padding:10px 14px; font-size:.88rem; font-weight:600; }
    .workflow-arrow { color:#6B7280; }
    .stButton > button { border-radius:10px; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)
