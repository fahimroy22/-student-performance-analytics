import streamlit as st


def apply_global_styles():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    with st.sidebar:
        st.markdown("### Appearance")
        dark_mode = st.toggle(
            "Dark mode",
            value=st.session_state.dark_mode,
            key="global_dark_mode_toggle"
        )
        st.session_state.dark_mode = dark_mode

    if st.session_state.dark_mode:
        colors = {
            "app_bg": "#0F172A",
            "sidebar_bg": "#111827",
            "card_bg": "#1F2937",
            "text": "#F3F4F6",
            "muted": "#9CA3AF",
            "border": "#374151",
            "input_bg": "#1F2937",
            "accent": "#5B8DB8",
            "hover": "#253247",
        }
    else:
        colors = {
            "app_bg": "#F7F8FA",
            "sidebar_bg": "#FFFFFF",
            "card_bg": "#FFFFFF",
            "text": "#1F2937",
            "muted": "#6B7280",
            "border": "#E5E7EB",
            "input_bg": "#FFFFFF",
            "accent": "#355C7D",
            "hover": "#F3F4F6",
        }

    st.markdown(
        f"""
        <style>

        /* ===================================================
           APP + HEADER
        =================================================== */

        .stApp {{
            background: {colors["app_bg"]};
            color: {colors["text"]};
        }}

        .block-container {{
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}

        [data-testid="stHeader"] {{
            background: {colors["app_bg"]} !important;
        }}

        [data-testid="stToolbar"] {{
            background: transparent !important;
        }}

        [data-testid="stDecoration"] {{
            background: {colors["app_bg"]} !important;
        }}

        h1,
        h2,
        h3,
        h4,
        h5,
        h6,
        p,
        label {{
            color: {colors["text"]};
        }}

        .stCaption,
        [data-testid="stCaptionContainer"] {{
            color: {colors["muted"]} !important;
        }}


        /* ===================================================
           SIDEBAR
        =================================================== */

        [data-testid="stSidebar"] {{
            background: {colors["sidebar_bg"]};
            border-right: 1px solid {colors["border"]};
        }}

        [data-testid="stSidebarNav"] {{
            background: transparent;
        }}

        [data-testid="stSidebarNav"] a {{
            border-radius: 8px;
        }}

        [data-testid="stSidebarNav"] a:hover {{
            background: {colors["hover"]};
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4,
        [data-testid="stSidebar"] h5,
        [data-testid="stSidebar"] h6,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {{
            color: {colors["text"]} !important;
        }}


        /* ===================================================
           SIDEBAR COLLAPSE / EXPAND BUTTONS
        =================================================== */

        [data-testid="stSidebarCollapseButton"] button {{
            background: {colors["hover"]} !important;
            color: {colors["text"]} !important;
            border-radius: 10px !important;
        }}

        [data-testid="stSidebarCollapseButton"] span,
        [data-testid="stSidebarCollapseButton"]
        [data-testid="stIconMaterial"] {{
            color: {colors["text"]} !important;
        }}

        [data-testid="stSidebarCollapseButton"] svg {{
            color: {colors["text"]} !important;
            fill: {colors["text"]} !important;
            stroke: {colors["text"]} !important;
        }}

        [data-testid="stSidebarCollapseButton"] button:hover {{
            background: {colors["accent"]} !important;
        }}

        [data-testid="stExpandSidebarButton"] {{
            background: {colors["hover"]} !important;
            color: {colors["text"]} !important;
            border-radius: 10px !important;
        }}

        [data-testid="stExpandSidebarButton"] span {{
            color: {colors["text"]} !important;
        }}

        [data-testid="stExpandSidebarButton"]
        [data-testid="stIconMaterial"] {{
            color: {colors["text"]} !important;
            -webkit-text-fill-color: {colors["text"]} !important;
        }}

        [data-testid="stExpandSidebarButton"] svg {{
            color: {colors["text"]} !important;
            fill: {colors["text"]} !important;
            stroke: {colors["text"]} !important;
        }}

        [data-testid="stExpandSidebarButton"]:hover {{
            background: {colors["accent"]} !important;
        }}


        /* ===================================================
           CUSTOM CARDS
        =================================================== */

        .metric-card {{
            background: {colors["card_bg"]};
            border: 1px solid {colors["border"]};
            border-radius: 14px;
            padding: 18px 20px;
            min-height: 118px;
        }}

        .metric-label {{
            color: {colors["muted"]};
            font-size: 0.88rem;
            margin-bottom: 7px;
        }}

        .metric-value {{
            color: {colors["text"]};
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.1;
        }}

        .metric-note {{
            color: {colors["muted"]};
            font-size: 0.78rem;
            margin-top: 8px;
        }}


        /* ===================================================
           STREAMLIT METRICS
        =================================================== */

        [data-testid="stMetric"] {{
            background: {colors["card_bg"]};
            border: 1px solid {colors["border"]};
            border-radius: 12px;
            padding: 14px;
        }}

        [data-testid="stMetricLabel"] {{
            color: {colors["muted"]} !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {colors["text"]} !important;
        }}


        /* ===================================================
           FEATURE CHIPS
        =================================================== */

        .feature-chip {{
            background: {colors["card_bg"]};
            border: 1px solid {colors["border"]};
            border-radius: 10px;
            padding: 12px 14px;
            margin: 5px 0;
            font-size: 0.92rem;
            color: {colors["text"]};
        }}


        /* ===================================================
           WORKFLOW
        =================================================== */

        .workflow {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 8px;
            margin: 12px 0 24px 0;
        }}

        .workflow-step {{
            background: {colors["card_bg"]};
            border: 1px solid {colors["border"]};
            border-radius: 10px;
            padding: 10px 14px;
            font-size: 0.88rem;
            font-weight: 600;
            color: {colors["text"]};
        }}

        .workflow-arrow {{
            color: {colors["muted"]};
        }}


        /* ===================================================
           NORMAL BUTTONS
        =================================================== */

        .stButton > button {{
            border-radius: 10px;
            font-weight: 600;
        }}

        .stButton > button:not([kind="primary"]) {{
            background: {colors["card_bg"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
        }}

        .stButton > button:not([kind="primary"]):hover {{
            background: {colors["hover"]};
            border-color: {colors["accent"]};
        }}

        .stButton > button[kind="primary"] {{
            background: {colors["accent"]} !important;
            border-color: {colors["accent"]} !important;
            color: #FFFFFF !important;
        }}

        .stButton > button[kind="primary"] p,
        .stButton > button[kind="primary"] span,
        .stButton > button[kind="primary"] div {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        .stButton > button[kind="primary"]:hover {{
            opacity: 0.92;
        }}


        /* ===================================================
           FORM SUBMIT BUTTONS
        =================================================== */

        [data-testid="stFormSubmitButton"] button {{
            background: {colors["accent"]} !important;
            border: 1px solid {colors["accent"]} !important;
            border-radius: 10px !important;
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
            font-weight: 600 !important;
        }}

        [data-testid="stFormSubmitButton"] button * {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        [data-testid="stFormSubmitButton"] button p {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        [data-testid="stFormSubmitButton"] button span {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        [data-testid="stFormSubmitButton"] button div {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}

        [data-testid="stFormSubmitButton"] button:hover {{
            background: {colors["accent"]} !important;
            border-color: {colors["accent"]} !important;
            color: #FFFFFF !important;
            opacity: 0.92;
        }}

        [data-testid="stFormSubmitButton"] button:hover * {{
            color: #FFFFFF !important;
            -webkit-text-fill-color: #FFFFFF !important;
        }}


        /* ===================================================
           NUMBER / TEXT INPUTS
        =================================================== */

        [data-baseweb="input"] > div {{
            background: {colors["input_bg"]} !important;
            border-color: {colors["border"]} !important;
        }}

        [data-baseweb="input"] input {{
            background: {colors["input_bg"]} !important;
            color: {colors["text"]} !important;
            -webkit-text-fill-color: {colors["text"]} !important;
        }}

        [data-testid="stNumberInput"] input {{
            background: {colors["input_bg"]} !important;
            color: {colors["text"]} !important;
            -webkit-text-fill-color: {colors["text"]} !important;
        }}


        /* ===================================================
           SELECTBOX
        =================================================== */

        [data-baseweb="select"] {{
            background: transparent !important;
            color: {colors["text"]} !important;
        }}

        [data-baseweb="select"] > div {{
            background: {colors["input_bg"]} !important;
            color: {colors["text"]} !important;
            border-color: {colors["border"]} !important;
        }}

        [data-baseweb="select"] span {{
            color: {colors["text"]} !important;
        }}

        [data-baseweb="select"] svg {{
            color: {colors["text"]} !important;
            fill: {colors["text"]} !important;
        }}

        input[role="combobox"] {{
            background: {colors["input_bg"]} !important;
            color: {colors["text"]} !important;
            border-color: {colors["border"]} !important;
            -webkit-text-fill-color: {colors["text"]} !important;
            caret-color: {colors["text"]} !important;
        }}

        input[role="combobox"]::placeholder {{
            color: {colors["muted"]} !important;
            -webkit-text-fill-color: {colors["muted"]} !important;
        }}

        input[role="combobox"]:focus,
        input[role="combobox"]:read-only {{
            background: {colors["input_bg"]} !important;
            color: {colors["text"]} !important;
            -webkit-text-fill-color: {colors["text"]} !important;
        }}

        [data-testid="stSelectbox"]
        [data-baseweb="select"] > div {{
            background-color: {colors["input_bg"]} !important;
            border-color: {colors["border"]} !important;
        }}

        [data-testid="stSelectbox"]
        input[role="combobox"] {{
            background-color: {colors["input_bg"]} !important;
            color: {colors["text"]} !important;
            -webkit-text-fill-color: {colors["text"]} !important;
        }}

        [data-testid="stSelectbox"] label {{
            color: {colors["text"]} !important;
        }}


        /* ===================================================
           SELECTBOX POPOVER
        =================================================== */

        [data-baseweb="popover"],
        [data-baseweb="popover"] > div {{
            background: {colors["card_bg"]} !important;
        }}

        [role="listbox"] {{
            background: {colors["card_bg"]} !important;
            border: 1px solid {colors["border"]} !important;
            color: {colors["text"]} !important;
        }}

        [role="option"] {{
            background: {colors["card_bg"]} !important;
            color: {colors["text"]} !important;
        }}

        [role="option"] * {{
            color: {colors["text"]} !important;
        }}

        [role="option"]:hover,
        [role="option"][aria-selected="true"] {{
            background: {colors["hover"]} !important;
            color: {colors["text"]} !important;
        }}


        /* ===================================================
           EXPANDERS
        =================================================== */

        [data-testid="stExpander"] {{
            background: {colors["card_bg"]};
            border: 1px solid {colors["border"]};
            border-radius: 10px;
        }}


        /* ===================================================
           DATAFRAME
        =================================================== */

        [data-testid="stDataFrame"] {{
            border: 1px solid {colors["border"]};
            border-radius: 10px;
            overflow: hidden;
        }}


        /* ===================================================
           TABS
        =================================================== */

        button[data-baseweb="tab"] {{
            color: {colors["muted"]};
        }}

        button[data-baseweb="tab"] p {{
            color: {colors["muted"]} !important;
        }}

        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {colors["accent"]};
        }}

        button[data-baseweb="tab"][aria-selected="true"] p {{
            color: {colors["text"]} !important;
        }}


        /* ===================================================
           OTHER
        =================================================== */

        hr {{
            border-color: {colors["border"]} !important;
        }}

        a {{
            color: {colors["accent"]};
        }}

        [data-testid="stToggle"] {{
            margin-bottom: 0.75rem;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )


def get_plotly_template():
    if st.session_state.get("dark_mode", False):
        return "plotly_dark"

    return "plotly_white"


def is_dark_mode():
    return st.session_state.get("dark_mode", False)


def get_theme_colors():
    if is_dark_mode():
        return {
            "background": "#0F172A",
            "surface": "#1F2937",
            "text": "#F3F4F6",
            "muted": "#9CA3AF",
            "border": "#374151",
            "accent": "#5B8DB8",
        }

    return {
        "background": "#F7F8FA",
        "surface": "#FFFFFF",
        "text": "#1F2937",
        "muted": "#6B7280",
        "border": "#E5E7EB",
        "accent": "#355C7D",
    }