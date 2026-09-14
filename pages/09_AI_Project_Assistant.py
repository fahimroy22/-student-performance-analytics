import base64
import html

import streamlit as st
from google import genai

from components.styles import (
    apply_global_styles,
    get_theme_colors,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Project Assistant",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# GLOBAL THEME
# ============================================================

apply_global_styles()

theme = get_theme_colors()

IS_DARK = st.session_state.get(
    "dark_mode",
    False,
)

APP_BG = theme["background"]
SURFACE = theme["surface"]
TEXT = theme["text"]
MUTED = theme["muted"]
BORDER = theme["border"]
ACCENT = theme["accent"]

USER_BG = (
    "#253247"
    if IS_DARK
    else "#E9EEF5"
)

INPUT_BG = (
    "#1F2937"
    if IS_DARK
    else "#FFFFFF"
)

CODE_BG = (
    "#111827"
    if IS_DARK
    else "#F6F8FA"
)

INLINE_CODE_BG = (
    "#253247"
    if IS_DARK
    else "#EEF2F7"
)

BOT_BG = (
    "#1F2937"
    if IS_DARK
    else "#F3F6FA"
)

BOT_ICON = (
    "#69A7D3"
    if IS_DARK
    else "#355C7D"
)


# ============================================================
# PAGE CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* ===================================================
       PAGE WIDTH
    =================================================== */

    .block-container {{
        max-width: 1040px !important;
        padding-top: 2rem !important;
        padding-bottom: 8rem !important;
    }}


    /* ===================================================
       PAGE HEADER
    =================================================== */

    .ai-header {{
        text-align: center;
        margin: 10px auto 20px auto;
        max-width: 760px;
    }}

    .ai-title {{
        color: {TEXT} !important;
        font-size: 2rem;
        font-weight: 750;
        line-height: 1.2;
        margin-bottom: 8px;
    }}

    .ai-subtitle {{
        color: {MUTED} !important;
        font-size: 0.96rem;
        line-height: 1.6;
    }}


    /* ===================================================
       TOP BAR
    =================================================== */

    .ai-top-note {{
        color: {MUTED};
        font-size: 0.82rem;
        padding-top: 11px;
    }}


    /* ===================================================
       BUTTONS
    =================================================== */

    .stButton > button {{
        min-height: 44px;
        border-radius: 12px !important;
        background: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        font-weight: 500 !important;
        transition:
            background 0.15s ease,
            border-color 0.15s ease;
    }}

    .stButton > button:hover {{
        background: {USER_BG} !important;
        border-color: {ACCENT} !important;
    }}

    .stButton > button p,
    .stButton > button span {{
        color: {TEXT} !important;
    }}


    /* ===================================================
       USER MESSAGE — RIGHT
    =================================================== */

    .user-message-wrap {{
        width: 100%;
        display: flex;
        justify-content: flex-end;
        margin: 19px 0 18px 0;
    }}

    .user-message {{
        max-width: 72%;
        background: {USER_BG};
        color: {TEXT};
        border: 1px solid {BORDER};
        border-radius: 18px 18px 5px 18px;
        padding: 11px 15px;
        font-size: 0.94rem;
        line-height: 1.55;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
    }}


    /* ===================================================
       AI BOT ICON
    =================================================== */

    .ai-bot-wrap {{
        width: 42px;
        height: 42px;
        border-radius: 12px;
        background: {BOT_BG};
        border: 1px solid {BORDER};
        display: flex;
        align-items: center;
        justify-content: center;
        margin-top: 3px;
        flex-shrink: 0;
    }}

    .ai-bot-wrap img {{
        width: 28px;
        height: 28px;
        display: block;
    }}


    /* ===================================================
       THINKING / TYPING ANIMATION
    =================================================== */

    .typing-indicator {{
        display: flex;
        align-items: center;
        gap: 6px;
        height: 42px;
        padding-left: 4px;
    }}

    .typing-dot {{
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: {MUTED};
        animation: typingPulse 1.2s infinite ease-in-out;
    }}

    .typing-dot:nth-child(2) {{
        animation-delay: 0.15s;
    }}

    .typing-dot:nth-child(3) {{
        animation-delay: 0.30s;
    }}

    @keyframes typingPulse {{
        0%,
        60%,
        100% {{
            opacity: 0.32;
            transform: translateY(0);
        }}

        30% {{
            opacity: 1;
            transform: translateY(-4px);
        }}
    }}


    /* ===================================================
       MARKDOWN TEXT
    =================================================== */

    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3,
    [data-testid="stMarkdownContainer"] h4 {{
        color: {TEXT} !important;
    }}

    [data-testid="stMarkdownContainer"] p {{
        color: {TEXT} !important;
        line-height: 1.65 !important;
    }}

    [data-testid="stMarkdownContainer"] li {{
        color: {TEXT} !important;
        line-height: 1.6 !important;
    }}

    [data-testid="stMarkdownContainer"] strong,
    [data-testid="stMarkdownContainer"] em {{
        color: {TEXT} !important;
    }}

    [data-testid="stMarkdownContainer"] hr {{
        border-color: {BORDER} !important;
    }}


    /* ===================================================
       INLINE CODE
    =================================================== */

    [data-testid="stMarkdownContainer"] p code,
    [data-testid="stMarkdownContainer"] li code {{
        background: {INLINE_CODE_BG} !important;
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 5px !important;
        padding: 2px 5px !important;
    }}


    /* ===================================================
       AI-GENERATED CODE BLOCKS — DARK MODE FIX
    =================================================== */

    [data-testid="stMarkdownContainer"] pre,
    [data-testid="stMarkdownContainer"] pre *,
    [data-testid="stCodeBlock"],
    [data-testid="stCodeBlock"] *,
    div[data-testid="stCodeBlock"],
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] code {{
        background: {CODE_BG} !important;
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        border-color: {BORDER} !important;
    }}

    [data-testid="stMarkdownContainer"] pre {{
        background: {CODE_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 16px !important;
        overflow-x: auto !important;
    }}

    [data-testid="stMarkdownContainer"] pre code {{
        background: transparent !important;
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
    }}

    [data-testid="stMarkdownContainer"] pre span {{
        background: transparent !important;
    }}

    [data-testid="stCodeBlock"] {{
        background: {CODE_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}

    [data-testid="stCodeBlock"] pre {{
        background: {CODE_BG} !important;
        color: {TEXT} !important;
    }}

    [data-testid="stCodeBlock"] code {{
        background: transparent !important;
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
    }}

    [data-testid="stCodeBlock"] span {{
        background: transparent !important;
    }}


    /* ===================================================
       KEEP COPY BUTTON VISIBLE + THEMED
    =================================================== */

    [data-testid="stCodeBlock"] button {{
        background: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 8px !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    [data-testid="stCodeBlock"] button:hover {{
        background: {USER_BG} !important;
        border-color: {ACCENT} !important;
    }}

    [data-testid="stCodeBlock"] button span {{
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
    }}

    [data-testid="stCodeBlock"] button svg {{
        color: {TEXT} !important;
        fill: {TEXT} !important;
        stroke: {TEXT} !important;
    }}

    [data-testid="stCodeBlock"] button * {{
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
    }}


    /* ===================================================
       SYNTAX TOKEN FALLBACKS
    =================================================== */

    [data-testid="stMarkdownContainer"] pre .token,
    [data-testid="stCodeBlock"] .token {{
        background: transparent !important;
    }}

    [data-testid="stMarkdownContainer"] pre code span,
    [data-testid="stCodeBlock"] code span {{
        -webkit-text-fill-color: currentColor !important;
    }}


    /* ===================================================
       LINKS
    =================================================== */

    [data-testid="stMarkdownContainer"] a {{
        color: {ACCENT} !important;
        font-weight: 600;
        text-decoration: none !important;
    }}

    [data-testid="stMarkdownContainer"] a:hover {{
        text-decoration: underline !important;
    }}


    /* ===================================================
       TABLES
    =================================================== */

    [data-testid="stMarkdownContainer"] table {{
        width: 100%;
        background: {SURFACE} !important;
        color: {TEXT} !important;
        border-collapse: collapse;
    }}

    [data-testid="stMarkdownContainer"] th {{
        background: {USER_BG} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        padding: 8px 10px !important;
    }}

    [data-testid="stMarkdownContainer"] td {{
        background: {SURFACE} !important;
        color: {TEXT} !important;
        border: 1px solid {BORDER} !important;
        padding: 8px 10px !important;
    }}


    /* ===================================================
       STARTER AREA
    =================================================== */

    .starter-title {{
        text-align: center;
        color: {TEXT};
        font-size: 1.3rem;
        font-weight: 700;
        margin-top: 42px;
        margin-bottom: 5px;
    }}

    .starter-subtitle {{
        text-align: center;
        color: {MUTED};
        font-size: 0.88rem;
        margin-bottom: 18px;
    }}


    /* ===================================================
       BOTTOM CHAT AREA
    =================================================== */

    [data-testid="stBottomBlockContainer"] {{
        background: {APP_BG} !important;
        border-top: 1px solid {BORDER} !important;
        padding-top: 12px !important;
    }}

    [data-testid="stBottomBlockContainer"] > div {{
        background: {APP_BG} !important;
    }}

    div[class*="stBottom"] {{
        background: {APP_BG} !important;
    }}


    /* ===================================================
       CHAT INPUT
    =================================================== */

    [data-testid="stChatInput"] {{
        background: {INPUT_BG} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 16px !important;
        box-shadow: none !important;
    }}

    [data-testid="stChatInput"]:focus-within {{
        border-color: {ACCENT} !important;
    }}

    [data-testid="stChatInput"] > div {{
        background: {INPUT_BG} !important;
    }}

    [data-baseweb="textarea"],
    [data-baseweb="textarea"] > div {{
        background: {INPUT_BG} !important;
    }}

    [data-testid="stChatInput"] textarea,
    [data-baseweb="textarea"] textarea {{
        background: {INPUT_BG} !important;
        color: {TEXT} !important;
        -webkit-text-fill-color: {TEXT} !important;
        caret-color: {TEXT} !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: {MUTED} !important;
        -webkit-text-fill-color: {MUTED} !important;
    }}

    [data-testid="stChatInput"] button {{
        background: transparent !important;
        color: {TEXT} !important;
    }}

    [data-testid="stChatInput"] button svg {{
        color: {TEXT} !important;
        fill: {TEXT} !important;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT KNOWLEDGE
# ============================================================

PROJECT_CONTEXT = """
You are the AI Project Assistant for a machine-learning project
called Student Performance Analytics.

You help users understand the project, machine-learning concepts,
Python code, model results, graphs, predictions, and errors.

PROJECT DATASET
- 100,000 student records
- 44 variables
- 52,161 missing values
- 0 duplicate rows

CLASS DISTRIBUTION
- Pass: 77,389 students = 77.4%
- Fail: 22,611 students = 22.6%

SELECTED FEATURES
1. previous_exam_score
2. previous_gpa
3. attendance_percentage
4. assignment_completion_rate
5. study_hours_per_day
6. practice_tests_completed

REGRESSION
Model:
Linear Regression

Target:
exam_score

Metrics:
- MAE: 9.38
- RMSE: 11.76
- R²: 0.444

CLASSIFICATION
Preferred model:
Balanced Logistic Regression

Target:
pass_status

Metrics:
- Accuracy: 72.5%
- Precision: 90.6%
- Recall: 71.9%
- F1: 80.1%
- Balanced Accuracy: 73.2%
- AUC: 0.814

MODEL COMPARISON
Dummy:
- Accuracy: 77.4%
- Balanced Accuracy: 50.0%

Standard Logistic:
- Accuracy: 80.9%
- Balanced Accuracy: 64.8%

Balanced Logistic:
- Accuracy: 72.5%
- Balanced Accuracy: 73.2%

PREPROCESSING
- Median imputation
- StandardScaler
- Six selected predictors
- 80/20 train-test split
- random_state = 42
- Classification split is stratified
- scikit-learn Pipeline

IMPORTANT PROJECT FACTS
- Linear Regression and Logistic Regression are separate models.
- Predicted exam score does not directly determine Pass probability.
- Regression coefficients represent associations, not causation.
- Predictions are estimates, not guaranteed outcomes.
- Do not invent project metrics or implementation details.
"""


# ============================================================
# PAGE GUIDE
# ============================================================

PAGE_GUIDE = """
Dashboard
URL: /
Used for:
- project overview
- dataset summary
- missing values
- duplicates
- exam-score distribution
- Pass/Fail overview

Dataset & EDA
URL: /Dataset_EDA
Used for:
- exploratory data analysis
- distributions
- missing values
- feature relationships
- correlations
- Pass/Fail distribution

Preprocessing
URL: /Preprocessing
Used for:
- feature selection
- median imputation
- StandardScaler
- train/test split
- preprocessing pipeline
- model-ready data

Linear Regression
URL: /Linear_Regression
Used for:
- regression training
- fit()
- exam-score prediction
- actual vs predicted
- residuals
- MAE
- RMSE
- R²
- coefficients

Logistic Regression
URL: /Logistic_Regression
Used for:
- classification training
- Pass/Fail prediction
- class_weight="balanced"
- predict()
- predict_proba()
- confusion matrix
- Accuracy
- Precision
- Recall
- F1
- Balanced Accuracy
- ROC
- AUC

Model Comparison
URL: /Model_Comparison
Used for:
- comparing classifiers
- Dummy classifier
- Standard Logistic Regression
- Balanced Logistic Regression
- Accuracy
- Balanced Accuracy
- model selection

Predict Performance
URL: /Predict_Performance
Used for:
- student feature inputs
- exam-score prediction
- Pass/Fail prediction
- Pass probability
- model-based scenario simulation

Threshold Explorer
URL: /Threshold_Explorer
Used for:
- classification threshold exploration
- threshold tradeoffs

Learn ML Pipeline
URL: /Learn_ML_Pipeline
Used for:
- beginner ML tutorial
- preprocessing explanation
- regression explanation
- classification explanation
- model evaluation
- prediction examples

AI Project Assistant
URL: /AI_Project_Assistant
Used for:
- project questions
- code explanations
- ML concepts
- result interpretation
- troubleshooting
"""


# ============================================================
# PAGE REFERENCE RULE
# ============================================================

PAGE_REFERENCE_RULE = """
When the answer discusses something that appears in the application,
finish with this section when relevant:

### Where this is used in the project

Use clickable Markdown page links.

Examples:

- [Preprocessing](/Preprocessing) — median imputation and scaling are performed here.
- [Linear Regression](/Linear_Regression) — this metric is used to evaluate the regression model.

Only include genuinely relevant pages.

Usually include 1 to 3 pages.

Do not list unrelated pages.

Use the exact URL paths from PAGE_GUIDE.

If there is no meaningful page reference, omit this section.
"""


# ============================================================
# RESPONSE STYLE
# ============================================================

RESPONSE_STYLE = """
Answer like a helpful beginner-friendly machine-learning tutor.

Rules:
- Use simple language.
- Be concise by default.
- Explain why something is used, not only what it is.
- Use project examples when helpful.
- Use fenced Python Markdown code blocks when code is useful.
- Never invent project facts.
- Do not treat association as causation.
- Use short headings only when helpful.
- Add clickable project page references when relevant.
"""


# ============================================================
# GEMINI CLIENT
# ============================================================

def get_gemini_client():

    try:

        api_key = st.secrets[
            "GEMINI_API_KEY"
        ]

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


client = get_gemini_client()


if client is None:

    st.error(
        "Gemini is not configured. "
        "Add GEMINI_API_KEY to .streamlit/secrets.toml."
    )

    st.stop()


# ============================================================
# SESSION STATE
# ============================================================

if "gemini_project_chat" not in st.session_state:

    st.session_state.gemini_project_chat = []


# ============================================================
# HEADER
# ============================================================

st.html(
    (
        '<div class="ai-header">'
        '<div class="ai-title">'
        'AI Project Assistant'
        '</div>'
        '<div class="ai-subtitle">'
        'Ask about machine learning, project code, '
        'model metrics, predictions, or errors.'
        '</div>'
        '</div>'
    )
)


# ============================================================
# TOP BAR
# ============================================================

top_left, top_right = st.columns(
    [5, 1]
)


with top_left:

    st.html(
        (
            '<div class="ai-top-note">'
            'Project-aware machine-learning tutor'
            '</div>'
        )
    )


with top_right:

    new_chat = st.button(
        "New chat",
        width="stretch",
        key="new_ai_chat",
    )


if new_chat:

    st.session_state.gemini_project_chat = []

    st.rerun()


# ============================================================
# STARTER PROMPTS
# ============================================================

starter_question = None


if not st.session_state.gemini_project_chat:

    st.html(
        (
            '<div class="starter-title">'
            'How can I help?'
            '</div>'
            '<div class="starter-subtitle">'
            'Choose a suggestion or ask your own question below.'
            '</div>'
        )
    )


    starter_row1 = st.columns(
        2
    )

    with starter_row1[0]:

        if st.button(
            "Explain R² = 0.444",
            width="stretch",
            key="starter_r2",
        ):

            starter_question = (
                "Explain what R² = 0.444 means in this project."
            )


    with starter_row1[1]:

        if st.button(
            "Why balanced accuracy?",
            width="stretch",
            key="starter_balanced_accuracy",
        ):

            starter_question = (
                "Why is balanced accuracy important "
                "for this project?"
            )


    starter_row2 = st.columns(
        2
    )

    with starter_row2[0]:

        if st.button(
            "Linear vs Logistic Regression",
            width="stretch",
            key="starter_models",
        ):

            starter_question = (
                "Explain the difference between Linear Regression "
                "and Logistic Regression in this project."
            )


    with starter_row2[1]:

        if st.button(
            "Why use an ML pipeline?",
            width="stretch",
            key="starter_pipeline",
        ):

            starter_question = (
                "Why does this project use "
                "a scikit-learn Pipeline?"
            )


# ============================================================
# BOT SVG AS BASE64 IMAGE
# ============================================================

def bot_svg():

    svg = f"""
    <svg
        xmlns="http://www.w3.org/2000/svg"
        width="28"
        height="28"
        viewBox="0 0 24 24"
        fill="none"
    >

        <path
            d="M12 2.5V5"
            stroke="{BOT_ICON}"
            stroke-width="1.8"
            stroke-linecap="round"
        />

        <path
            d="M9.5 2.5H14.5"
            stroke="{BOT_ICON}"
            stroke-width="1.8"
            stroke-linecap="round"
        />

        <rect
            x="4"
            y="6"
            width="16"
            height="13"
            rx="4"
            stroke="{BOT_ICON}"
            stroke-width="1.8"
        />

        <circle
            cx="9"
            cy="12"
            r="1.3"
            fill="{BOT_ICON}"
        />

        <circle
            cx="15"
            cy="12"
            r="1.3"
            fill="{BOT_ICON}"
        />

        <path
            d="M9 16H15"
            stroke="{BOT_ICON}"
            stroke-width="1.8"
            stroke-linecap="round"
        />

        <path
            d="M4 11H2.5"
            stroke="{BOT_ICON}"
            stroke-width="1.8"
            stroke-linecap="round"
        />

        <path
            d="M21.5 11H20"
            stroke="{BOT_ICON}"
            stroke-width="1.8"
            stroke-linecap="round"
        />

    </svg>
    """

    encoded = base64.b64encode(
        svg.encode("utf-8")
    ).decode("utf-8")

    return (
        '<div class="ai-bot-wrap">'
        '<img '
        f'src="data:image/svg+xml;base64,{encoded}" '
        'alt="AI Assistant" '
        'width="28" '
        'height="28">'
        '</div>'
    )


# ============================================================
# THINKING INDICATOR
# ============================================================

def thinking_indicator():

    return (
        '<div class="typing-indicator">'
        '<div class="typing-dot"></div>'
        '<div class="typing-dot"></div>'
        '<div class="typing-dot"></div>'
        '</div>'
    )


# ============================================================
# USER MESSAGE
# ============================================================

def show_user_message(message):

    escaped_message = html.escape(
        str(message)
    )

    st.html(
        (
            '<div class="user-message-wrap">'
            '<div class="user-message">'
            f'{escaped_message}'
            '</div>'
            '</div>'
        )
    )


# ============================================================
# ASSISTANT MESSAGE
# ============================================================

def show_assistant_message(message):

    icon_col, response_col = st.columns(
        [0.06, 0.94]
    )

    with icon_col:

        st.html(
            bot_svg()
        )

    with response_col:

        st.markdown(
            message
        )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.gemini_project_chat:

    if message["role"] == "user":

        show_user_message(
            message["content"]
        )

    else:

        show_assistant_message(
            message["content"]
        )


# ============================================================
# CHAT INPUT
# ============================================================

typed_question = st.chat_input(
    "Ask about the project, code, models, or an error..."
)


user_question = (
    starter_question
    if starter_question
    else typed_question
)


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_prompt(question):

    recent_messages = (
        st.session_state
        .gemini_project_chat[-8:]
    )

    conversation = ""

    for message in recent_messages:

        conversation += (
            f"\n{message['role'].upper()}: "
            f"{message['content']}\n"
        )


    return f"""
{PROJECT_CONTEXT}

{PAGE_GUIDE}

{PAGE_REFERENCE_RULE}

{RESPONSE_STYLE}

RECENT CONVERSATION
{conversation}

NEW USER QUESTION
{question}

Answer the new user question.

When relevant, finish with the clickable
"Where this is used in the project" section.
"""


# ============================================================
# FRIENDLY ERRORS
# ============================================================

def friendly_error(error):

    error_text = str(
        error
    ).lower()


    if (
        "429" in error_text
        or "quota" in error_text
        or "resource_exhausted" in error_text
    ):

        return (
            "The free Gemini usage limit has been reached temporarily. "
            "Please wait a little and try again."
        )


    if (
        "401" in error_text
        or "403" in error_text
        or "api key" in error_text
    ):

        return (
            "Gemini could not authenticate the API key. "
            "Please check your Streamlit secrets."
        )


    if (
        "404" in error_text
        or "not_found" in error_text
    ):

        return (
            "The configured Gemini model is currently unavailable."
        )


    if (
        "503" in error_text
        or "unavailable" in error_text
    ):

        return (
            "Gemini is temporarily unavailable. "
            "Please try again shortly."
        )


    return (
        "I could not generate a response right now. "
        "Please try again."
    )


# ============================================================
# GEMINI STREAM
# ============================================================

def stream_gemini_response(prompt):

    response_stream = (
        client.models.generate_content_stream(
            model="gemini-3.6-flash",
            contents=prompt,
        )
    )


    for chunk in response_stream:

        try:

            chunk_text = chunk.text

        except Exception:

            chunk_text = None


        if chunk_text:

            yield chunk_text


# ============================================================
# GENERATE RESPONSE
# ============================================================

if user_question:

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.gemini_project_chat.append(
        {
            "role": "user",
            "content": user_question,
        }
    )


    # --------------------------------------------------------
    # USER MESSAGE — RIGHT
    # --------------------------------------------------------

    show_user_message(
        user_question
    )


    # --------------------------------------------------------
    # BUILD PROMPT
    # --------------------------------------------------------

    prompt = build_prompt(
        user_question
    )


    # --------------------------------------------------------
    # AI RESPONSE — LEFT
    # --------------------------------------------------------

    icon_col, response_col = st.columns(
        [0.06, 0.94]
    )


    with icon_col:

        st.html(
            bot_svg()
        )


    with response_col:

        thinking_placeholder = st.empty()

        thinking_placeholder.html(
            thinking_indicator()
        )

        stream_state = {
            "started": False
        }


        try:

            response_stream = (
                stream_gemini_response(
                    prompt
                )
            )


            def animated_stream():

                for chunk in response_stream:

                    if not stream_state[
                        "started"
                    ]:

                        thinking_placeholder.empty()

                        stream_state[
                            "started"
                        ] = True

                    yield chunk


            answer = st.write_stream(
                animated_stream()
            )


            if not stream_state[
                "started"
            ]:

                thinking_placeholder.empty()


            if not answer:

                answer = (
                    "I couldn't generate a text response. "
                    "Please try asking the question another way."
                )

                st.markdown(
                    answer
                )


        except Exception as error:

            thinking_placeholder.empty()

            answer = friendly_error(
                error
            )

            st.markdown(
                answer
            )


    # --------------------------------------------------------
    # SAVE AI RESPONSE
    # --------------------------------------------------------

    st.session_state.gemini_project_chat.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# EMPTY CHAT NOTE
# ============================================================

if not st.session_state.gemini_project_chat:

    st.caption(
        "AI responses are for educational support. "
        "Do not enter passwords, API keys, or private student information."
    )