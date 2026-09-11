import streamlit as st


# ============================================================
# ANIMATED MACHINE-LEARNING PIPELINE
# Real ECG path-following animation
# ============================================================

def render_ml_pipeline_story(theme_colors):

    surface = theme_colors["surface"]
    text = theme_colors["text"]
    muted = theme_colors["muted"]
    border = theme_colors["border"]
    accent = theme_colors["accent"]

    stages = [
        ("01", "Raw Data", "100,000 records"),
        ("02", "Data Quality", "Missing values & duplicates"),
        ("03", "6 Predictors", "Academic features"),
        ("04", "80 / 20 Split", "Train & test sets"),
        ("05", "Preprocessing", "Imputation & scaling"),
        ("06", "ML Models", "Linear + Logistic"),
        ("07", "Evaluation", "Performance metrics"),
        ("08", "Prediction", "Score + Pass / Fail"),
    ]

    # --------------------------------------------------------
    # NODE POSITIONS
    # --------------------------------------------------------

    centers = [
        65,
        245,
        425,
        605,
        785,
        965,
        1145,
        1325,
    ]

    baseline = 52

    # --------------------------------------------------------
    # BUILD ONE CONTINUOUS ECG PATH
    # --------------------------------------------------------

    path = f"M {centers[0]} {baseline}"

    for i in range(len(centers) - 1):

        x1 = centers[i]
        x2 = centers[i + 1]

        distance = x2 - x1

        a = x1 + distance * 0.23
        b = x1 + distance * 0.31
        c = x1 + distance * 0.38
        d = x1 + distance * 0.45
        e = x1 + distance * 0.52
        f = x1 + distance * 0.59
        g = x1 + distance * 0.67

        path += (
            f" L {a:.1f} {baseline}"
            f" L {b:.1f} {baseline - 7}"
            f" L {c:.1f} {baseline + 10}"
            f" L {d:.1f} {baseline - 32}"
            f" L {e:.1f} {baseline + 34}"
            f" L {f:.1f} {baseline - 10}"
            f" L {g:.1f} {baseline}"
            f" L {x2} {baseline}"
        )

    # --------------------------------------------------------
    # SVG NODES
    # --------------------------------------------------------

    nodes = ""

    for i, (number, _, _) in enumerate(stages):

        x = centers[i]

        nodes += f"""
        <g class="node node-{i + 1}">

            <circle
                class="node-halo"
                cx="{x}"
                cy="{baseline}"
                r="31"
            />

            <circle
                class="node-circle"
                cx="{x}"
                cy="{baseline}"
                r="27"
            />

            <text
                class="node-number"
                x="{x}"
                y="{baseline + 4}"
                text-anchor="middle"
            >
                {number}
            </text>

        </g>
        """

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    labels = ""

    for _, title, description in stages:

        labels += f"""
        <div class="stage-label">

            <div class="stage-title">
                {title}
            </div>

            <div class="stage-description">
                {description}
            </div>

        </div>
        """

    # --------------------------------------------------------
    # FULL COMPONENT
    # --------------------------------------------------------

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

    <meta charset="UTF-8">

    <style>

    * {{
        box-sizing: border-box;
    }}

    html,
    body {{
        margin: 0;
        padding: 0;

        background: transparent;

        font-family:
            -apple-system,
            BlinkMacSystemFont,
            "Segoe UI",
            sans-serif;
    }}


    /* ======================================================
       MAIN CARD
    ====================================================== */

    .pipeline-card {{
        width: 100%;

        background: {surface};

        border:
            1px solid {border};

        border-radius:
            16px;

        padding:
            22px 20px 17px 20px;

        overflow:
            hidden;
    }}


    /* ======================================================
       SVG AREA
    ====================================================== */

    .svg-container {{
        width: 100%;
        height: 100px;
    }}


    svg {{
        width: 100%;
        height: 100%;

        overflow: visible;
    }}


    /* ======================================================
       ECG
    ====================================================== */

    .ecg-background {{
        fill: none;

        stroke:
            {border};

        stroke-width:
            2;

        stroke-linecap:
            round;

        stroke-linejoin:
            round;

        vector-effect:
            non-scaling-stroke;
    }}


    .ecg-active {{
        fill: none;

        stroke:
            {accent};

        stroke-width:
            2.25;

        stroke-linecap:
            round;

        stroke-linejoin:
            round;

        vector-effect:
            non-scaling-stroke;

        opacity:
            0.72;
    }}


    /* ======================================================
       NODES
    ====================================================== */

    .node-circle {{
        fill:
            {surface};

        stroke:
            {accent};

        stroke-width:
            2;

        vector-effect:
            non-scaling-stroke;
    }}


    .node-halo {{
        fill:
            none;

        stroke:
            {accent};

        stroke-width:
            7;

        opacity:
            0;
    }}


    .node-number {{
        fill:
            {accent};

        font-size:
            11px;

        font-weight:
            700;

        font-family:
            Arial,
            sans-serif;
    }}


    /* ======================================================
       NODE PULSE
    ====================================================== */

    .node-1 .node-halo {{
        animation: nodePulse 8s linear infinite;
        animation-delay: 0s;
    }}

    .node-2 .node-halo {{
        animation: nodePulse 8s linear infinite;
        animation-delay: 1.08s;
    }}

    .node-3 .node-halo {{
        animation: nodePulse 8s linear infinite;
        animation-delay: 2.16s;
    }}

    .node-4 .node-halo {{
        animation: nodePulse 8s linear infinite;
        animation-delay: 3.24s;
    }}

    .node-5 .node-halo {{
        animation: nodePulse 8s linear infinite;
        animation-delay: 4.32s;
    }}

    .node-6 .node-halo {{
        animation: nodePulse 8s linear infinite;
        animation-delay: 5.40s;
    }}

    .node-7 .node-halo {{
        animation: nodePulse 8s linear infinite;
        animation-delay: 6.48s;
    }}

    .node-8 .node-halo {{
        animation: nodePulse 8s linear infinite;
        animation-delay: 7.56s;
    }}


    @keyframes nodePulse {{

        0% {{
            opacity: 0;
        }}

        4% {{
            opacity: 0.08;
        }}

        8% {{
            opacity: 0.48;
        }}

        12% {{
            opacity: 0.15;
        }}

        18%,
        100% {{
            opacity: 0;
        }}

    }}


    /* ======================================================
       LABELS
    ====================================================== */

    .label-grid {{
        display: grid;

        grid-template-columns:
            repeat(8, 1fr);

        gap:
            6px;

        margin-top:
            -5px;
    }}


    .stage-label {{
        text-align:
            center;

        min-width:
            0;
    }}


    .stage-title {{
        color:
            {text};

        font-size:
            11px;

        line-height:
            1.25;

        font-weight:
            700;

        margin-bottom:
            5px;
    }}


    .stage-description {{
        color:
            {muted};

        font-size:
            9px;

        line-height:
            1.35;

        max-width:
            125px;

        margin:
            auto;
    }}


    /* ======================================================
       INFO BOX
    ====================================================== */

    .info-box {{
        margin-top:
            17px;

        padding:
            11px 16px;

        text-align:
            center;

        border:
            1px solid {border};

        border-radius:
            9px;
    }}


    .info-title {{
        color:
            {accent};

        font-size:
            11px;

        font-weight:
            700;

        margin-bottom:
            3px;
    }}


    .info-text {{
        color:
            {muted};

        font-size:
            9px;

        line-height:
            1.5;
    }}


    /* ======================================================
       RESPONSIVE
    ====================================================== */

    @media (max-width: 900px) {{

        .pipeline-card {{
            overflow-x:
                auto;
        }}

        .svg-container,
        .label-grid {{
            min-width:
                950px;
        }}

    }}


    /* ======================================================
       ACCESSIBILITY
    ====================================================== */

    @media (prefers-reduced-motion: reduce) {{

        .node-halo {{
            animation:
                none !important;
        }}

    }}

    </style>

    </head>


    <body>


    <div class="pipeline-card">


        <div class="svg-container">


        <svg
            viewBox="0 0 1390 105"
            xmlns="http://www.w3.org/2000/svg"
        >


            <!-- ==========================================
                 FILTERS
            =========================================== -->

            <defs>


                <filter
                    id="pulseGlow"
                    x="-400%"
                    y="-400%"
                    width="800%"
                    height="800%"
                >

                    <feGaussianBlur
                        stdDeviation="5"
                        result="blur"
                    />

                    <feMerge>

                        <feMergeNode
                            in="blur"
                        />

                        <feMergeNode
                            in="SourceGraphic"
                        />

                    </feMerge>

                </filter>


            </defs>


            <!-- ==========================================
                 ECG PATH
            =========================================== -->

            <path
                class="ecg-background"
                d="{path}"
            />


            <path
                class="ecg-active"
                d="{path}"
            />


            <!-- ==========================================
                 NODES
            =========================================== -->

            {nodes}


            <!-- ==========================================
                 MAIN MOVING PULSE
                 Follows the ECG path itself.
            =========================================== -->

            <circle
                r="5.5"
                fill="{accent}"
                filter="url(#pulseGlow)"
            >

                <animateMotion
                    dur="8s"
                    repeatCount="indefinite"
                    path="{path}"
                    calcMode="linear"
                />

            </circle>


            <!-- bright center of the pulse -->

            <circle
                r="2"
                fill="white"
            >

                <animateMotion
                    dur="8s"
                    repeatCount="indefinite"
                    path="{path}"
                    calcMode="linear"
                />

            </circle>


        </svg>


        </div>


        <div class="label-grid">

            {labels}

        </div>


        <div class="info-box">

            <div class="info-title">

                Follow the data flow through the
                machine-learning pipeline

            </div>


            <div class="info-text">

                The animated pulse follows the actual
                heartbeat path as student data moves through
                data quality checks, feature selection,
                preprocessing, model training, evaluation,
                and finally the prediction system.

            </div>

        </div>


    </div>


    </body>

    </html>
    """

    # --------------------------------------------------------
    # CURRENT STREAMLIT IFRAME API
    #
    # st.iframe accepts raw HTML directly, so this preserves
    # the SVG animateMotion behavior without using the
    # deprecated st.components.v1.html API.
    # --------------------------------------------------------

    st.iframe(
        html,
        width="stretch",
        height=245,
        tab_index=-1,
    )


# ============================================================
# SYSTEM STATUS
# ============================================================

def render_system_status(
    theme_colors,
    records,
    variables,
    missing_values,
):

    surface = theme_colors["surface"]
    text = theme_colors["text"]
    muted = theme_colors["muted"]
    border = theme_colors["border"]
    accent = theme_colors["accent"]

    status_items = [
        (
            "Dataset",
            f"{records:,} records · {variables} variables",
            "Ready",
        ),
        (
            "Data Quality",
            f"{missing_values:,} missing cells handled in pipeline",
            "Ready",
        ),
        (
            "Regression Model",
            "Linear Regression pipeline",
            "Loaded",
        ),
        (
            "Classification Model",
            "Balanced Logistic Regression pipeline",
            "Loaded",
        ),
        (
            "Deployment",
            "Streamlit web application",
            "Online",
        ),
    ]

    cards = ""

    for title, description, status in status_items:

        cards += f"""
        <div class="system-card">

            <div class="system-dot"></div>

            <div>

                <div class="system-title">
                    {title}
                </div>

                <div class="system-description">
                    {description}
                </div>

                <div class="system-status">
                    {status}
                </div>

            </div>

        </div>
        """

    html = f"""
    <style>

    .system-grid {{
        display: grid;

        grid-template-columns:
            repeat(5, 1fr);

        gap:
            10px;

        margin:
            8px 0 20px 0;
    }}


    .system-card {{
        background:
            {surface};

        border:
            1px solid {border};

        border-radius:
            12px;

        padding:
            14px 13px;

        display:
            grid;

        grid-template-columns:
            10px 1fr;

        column-gap:
            9px;

        min-height:
            84px;

        transition:
            transform .18s ease,
            border-color .18s ease;
    }}


    .system-card:hover {{
        transform:
            translateY(-2px);

        border-color:
            {accent};
    }}


    .system-dot {{
        width:
            8px;

        height:
            8px;

        margin-top:
            5px;

        border-radius:
            50%;

        background:
            {accent};

        box-shadow:
            0 0 0 3px {border};
    }}


    .system-title {{
        color:
            {text};

        font-size:
            12px;

        font-weight:
            650;

        margin-bottom:
            4px;
    }}


    .system-description {{
        color:
            {muted};

        font-size:
            10px;

        line-height:
            1.35;
    }}


    .system-status {{
        color:
            {accent};

        font-size:
            10px;

        font-weight:
            650;

        margin-top:
            8px;
    }}


    @media (max-width: 1000px) {{

        .system-grid {{
            grid-template-columns:
                repeat(2, 1fr);
        }}

    }}


    @media (max-width: 600px) {{

        .system-grid {{
            grid-template-columns:
                1fr;
        }}

    }}

    </style>


    <div class="system-grid">

        {cards}

    </div>
    """

    st.html(
        html
    )