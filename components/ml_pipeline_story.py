import streamlit as st


# ============================================================
# ANIMATED MACHINE-LEARNING PIPELINE
# ECG design with one synchronized dot + smooth trail
# ============================================================

def render_ml_pipeline_story(theme_colors):

    surface = theme_colors["surface"]
    text = theme_colors["text"]
    muted = theme_colors["muted"]
    border = theme_colors["border"]
    accent = theme_colors["accent"]

    positive = "#5FA879"


    # ========================================================
    # PIPELINE STAGES
    # ========================================================

    stages = [
        (
            "01",
            "Raw Data",
            "100,000 records",
        ),
        (
            "02",
            "Data Quality",
            "Missing values + duplicates",
        ),
        (
            "03",
            "6 Predictors",
            "Selected academic features",
        ),
        (
            "04",
            "80 / 20 Split",
            "Training + test sets",
        ),
        (
            "05",
            "Preprocessing",
            "Median imputation + scaling",
        ),
        (
            "06",
            "ML Models",
            "Linear + Balanced Logistic",
        ),
        (
            "07",
            "Evaluation",
            "Performance metrics",
        ),
        (
            "08",
            "Prediction",
            "Score + Class + Probability",
        ),
    ]


    # ========================================================
    # NODE POSITIONS
    # ========================================================

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

    baseline = 56


    # ========================================================
    # BUILD ECG PATH
    # ========================================================

    path = (
        f"M {centers[0]} "
        f"{baseline}"
    )


    for i in range(
        len(centers) - 1
    ):

        x1 = centers[i]
        x2 = centers[i + 1]

        distance = (
            x2 - x1
        )

        a = x1 + distance * 0.23
        b = x1 + distance * 0.31
        c = x1 + distance * 0.38
        d = x1 + distance * 0.45
        e = x1 + distance * 0.52
        f = x1 + distance * 0.59
        g = x1 + distance * 0.67


        path += (
            f" L {a:.1f} {baseline}"
            f" L {b:.1f} {baseline - 5}"
            f" L {c:.1f} {baseline + 7}"
            f" L {d:.1f} {baseline - 23}"
            f" L {e:.1f} {baseline + 25}"
            f" L {f:.1f} {baseline - 8}"
            f" L {g:.1f} {baseline}"
            f" L {x2} {baseline}"
        )


    # ========================================================
    # SVG NODES
    # ========================================================

    nodes = ""


    for i, (
        number,
        _,
        _,
    ) in enumerate(stages):

        x = centers[i]

        nodes += f"""
        <g
            class="node"
            id="pipeline-node-{i}"
        >

            <circle
                class="node-halo"
                cx="{x}"
                cy="{baseline}"
                r="36"
            />

            <circle
                class="node-circle"
                cx="{x}"
                cy="{baseline}"
                r="30"
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


    # ========================================================
    # STAGE CARDS
    # ========================================================

    labels = ""


    for i, (
        _,
        title,
        description,
    ) in enumerate(stages):

        special_class = ""

        if i == 5:
            special_class = (
                " model-stage"
            )

        elif i == 7:
            special_class = (
                " prediction-stage"
            )


        labels += f"""
        <div
            id="pipeline-card-{i}"
            class="stage-card{special_class}"
        >

            <div class="stage-title">
                {title}
            </div>

            <div class="stage-description">
                {description}
            </div>

        </div>
        """


    # ========================================================
    # COMPLETE COMPONENT
    # ========================================================

    html = f"""
    <!DOCTYPE html>

    <html>

    <head>

    <meta charset="UTF-8">

    <style>


    /* ======================================================
       BASE
    ====================================================== */

    * {{
        box-sizing:
            border-box;
    }}


    html,
    body {{
        margin:
            0;

        padding:
            0;

        background:
            transparent;

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
        width:
            100%;

        background:
            {surface};

        border:
            1px solid {border};

        border-radius:
            16px;

        padding:
            17px 16px 15px 16px;

        overflow:
            hidden;
    }}


    /* ======================================================
       ECG AREA
    ====================================================== */

    .svg-container {{
        width:
            100%;

        height:
            118px;
    }}


    svg {{
        width:
            100%;

        height:
            100%;

        overflow:
            visible;
    }}


    /* ======================================================
       ECG BACKGROUND
    ====================================================== */

    .ecg-background {{
        fill:
            none;

        stroke:
            {border};

        stroke-width:
            2.2;

        stroke-linecap:
            round;

        stroke-linejoin:
            round;

        vector-effect:
            non-scaling-stroke;
    }}


    /* ======================================================
       SMOOTH MOVING TRAIL
    ====================================================== */

    .motion-trail-glow {{
        fill:
            none;

        stroke:
            {accent};

        stroke-width:
            8;

        stroke-linecap:
            round;

        stroke-linejoin:
            round;

        opacity:
            0.12;

        vector-effect:
            non-scaling-stroke;

        filter:
            url(#trailBlur);
    }}


    .motion-trail {{
        fill:
            none;

        stroke:
            {accent};

        stroke-width:
            3.2;

        stroke-linecap:
            round;

        stroke-linejoin:
            round;

        opacity:
            0.82;

        vector-effect:
            non-scaling-stroke;
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
            1.9;

        vector-effect:
            non-scaling-stroke;

        transition:
            stroke-width .20s ease,
            filter .20s ease;
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

        transition:
            opacity .20s ease;
    }}


    .node-number {{
        fill:
            {accent};

        font-size:
            12px;

        font-weight:
            750;

        font-family:
            Arial,
            sans-serif;
    }}


    /* ======================================================
       ACTIVE NODE
    ====================================================== */

    .node.active .node-halo {{
        opacity:
            0.42;
    }}


    .node.active .node-circle {{
        stroke-width:
            2.3;

        filter:
            drop-shadow(
                0 0 4px {accent}
            );
    }}


    .node.final-active .node-halo {{
        stroke:
            {positive};

        opacity:
            0.45;
    }}


    .node.final-active .node-circle {{
        stroke:
            {positive};

        stroke-width:
            2.3;

        filter:
            drop-shadow(
                0 0 4px {positive}
            );
    }}


    .node.final-active .node-number {{
        fill:
            {positive};
    }}


    /* ======================================================
       LABEL GRID
    ====================================================== */

    .label-grid {{
        display:
            grid;

        grid-template-columns:
            repeat(8, 1fr);

        gap:
            8px;

        margin-top:
            -3px;
    }}


    /* ======================================================
       STAGE CARDS
    ====================================================== */

    .stage-card {{
        min-width:
            0;

        min-height:
            77px;

        background:
            {surface};

        border:
            1px solid {border};

        border-radius:
            10px;

        padding:
            10px 7px 9px 7px;

        text-align:
            center;

        transition:
            border-color .20s ease,
            transform .20s ease,
            box-shadow .20s ease;
    }}


    .stage-title {{
        color:
            {text};

        font-size:
            13px;

        line-height:
            1.25;

        font-weight:
            700;

        margin-bottom:
            6px;
    }}


    .stage-description {{
        color:
            {muted};

        font-size:
            10.5px;

        line-height:
            1.35;

        max-width:
            150px;

        margin:
            auto;
    }}


    /* ======================================================
       ACTIVE CARD
    ====================================================== */

    .stage-card.active {{
        border-color:
            {accent};

        transform:
            translateY(-2px);

        box-shadow:
            0 0 0 1px {accent};
    }}


    .stage-card.final-active {{
        border-color:
            {positive};

        transform:
            translateY(-2px);

        box-shadow:
            0 0 0 1px {positive};
    }}


    /* ======================================================
       SPECIAL STAGE TITLES
    ====================================================== */

    .model-stage .stage-title {{
        color:
            {accent};
    }}


    .prediction-stage .stage-title {{
        color:
            {positive};
    }}


    /* ======================================================
       RESPONSIVE
    ====================================================== */

    @media (
        max-width:
        950px
    ) {{

        .pipeline-card {{
            overflow-x:
                auto;
        }}


        .svg-container,
        .label-grid {{
            min-width:
                1120px;
        }}

    }}


    /* ======================================================
       REDUCED MOTION
    ====================================================== */

    @media (
        prefers-reduced-motion:
        reduce
    ) {{

        #motionTrail,
        #motionTrailGlow,
        #motionDot,
        #motionCenter {{
            display:
                none;
        }}

    }}


    </style>

    </head>


    <body>


    <div class="pipeline-card">


        <!-- ==================================================
             ECG AREA
        =================================================== -->

        <div class="svg-container">


        <svg
            viewBox="0 0 1390 118"
            xmlns="http://www.w3.org/2000/svg"
        >


            <!-- ==============================================
                 FILTERS
            =============================================== -->

            <defs>


                <filter
                    id="pulseGlow"
                    x="-400%"
                    y="-400%"
                    width="800%"
                    height="800%"
                >

                    <feGaussianBlur
                        stdDeviation="4.8"
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


                <filter
                    id="trailBlur"
                    x="-100%"
                    y="-300%"
                    width="300%"
                    height="700%"
                >

                    <feGaussianBlur
                        stdDeviation="3"
                    />

                </filter>


            </defs>


            <!-- ==============================================
                 HIDDEN MOTION PATH

                 JavaScript uses this exact path for BOTH
                 the dot position and trail position.
            =============================================== -->

            <path
                id="motionPath"
                d="{path}"
                fill="none"
                stroke="none"
            />


            <!-- ==============================================
                 MUTED ECG BASE
            =============================================== -->

            <path
                class="ecg-background"
                d="{path}"
            />


            <!-- ==============================================
                 SMOOTH TRAIL GLOW
            =============================================== -->

            <polyline
                id="motionTrailGlow"
                class="motion-trail-glow"
                points=""
            />


            <!-- ==============================================
                 SMOOTH TRAIL
            =============================================== -->

            <polyline
                id="motionTrail"
                class="motion-trail"
                points=""
            />


            <!-- ==============================================
                 NUMBERED NODES
            =============================================== -->

            {nodes}


            <!-- ==============================================
                 SINGLE MOVING DOT
            =============================================== -->

            <circle
                id="motionDot"
                r="6.2"
                fill="{accent}"
                filter="url(#pulseGlow)"
            />


            <!-- ==============================================
                 WHITE DOT CENTER
            =============================================== -->

            <circle
                id="motionCenter"
                r="2.15"
                fill="white"
            />


        </svg>


        </div>


        <!-- ==================================================
             STAGE CARDS
        =================================================== -->

        <div class="label-grid">

            {labels}

        </div>


    </div>


    <!-- ======================================================
         SYNCHRONIZED ANIMATION
    ======================================================= -->

    <script>

    (function() {{

        const path =
            document.getElementById(
                "motionPath"
            );


        const trail =
            document.getElementById(
                "motionTrail"
            );


        const trailGlow =
            document.getElementById(
                "motionTrailGlow"
            );


        const dot =
            document.getElementById(
                "motionDot"
            );


        const center =
            document.getElementById(
                "motionCenter"
            );


        if (
            !path ||
            !trail ||
            !trailGlow ||
            !dot ||
            !center
        ) {{
            return;
        }}


        /* --------------------------------------------------
           ANIMATION SETTINGS
        -------------------------------------------------- */

        const duration =
            9600;


        /*
        Trail length is measured in SVG path units.

        Increase this number for a longer trail.
        Reduce it for a shorter trail.
        */

        const trailLength =
            105;


        /*
        Smaller sampleStep = smoother trail.

        3 provides a smooth shape without creating
        excessive SVG points.
        */

        const sampleStep =
            3;


        const totalLength =
            path.getTotalLength();


        let startTime =
            null;


        let previousActive =
            -1;


        const nodeFractions = [
            0,
            1 / 7,
            2 / 7,
            3 / 7,
            4 / 7,
            5 / 7,
            6 / 7,
            1
        ];


        /* --------------------------------------------------
           ACTIVE STAGE
        -------------------------------------------------- */

        function updateActiveStage(
            progress
        ) {{

            let closestIndex =
                0;


            let smallestDistance =
                Infinity;


            for (
                let i = 0;
                i < nodeFractions.length;
                i++
            ) {{

                const distance =
                    Math.abs(
                        progress
                        - nodeFractions[i]
                    );


                if (
                    distance
                    < smallestDistance
                ) {{

                    smallestDistance =
                        distance;

                    closestIndex =
                        i;
                }}

            }}


            /*
            Only highlight a stage while the dot
            is reasonably close to its node.
            */

            const activeThreshold =
                0.045;


            let activeIndex =
                -1;


            if (
                smallestDistance
                <= activeThreshold
            ) {{

                activeIndex =
                    closestIndex;
            }}


            if (
                activeIndex
                === previousActive
            ) {{
                return;
            }}


            /* Remove previous state */

            for (
                let i = 0;
                i < 8;
                i++
            ) {{

                const node =
                    document.getElementById(
                        `pipeline-node-${{i}}`
                    );


                const card =
                    document.getElementById(
                        `pipeline-card-${{i}}`
                    );


                if (node) {{

                    node.classList.remove(
                        "active",
                        "final-active"
                    );

                }}


                if (card) {{

                    card.classList.remove(
                        "active",
                        "final-active"
                    );

                }}

            }}


            /* Add current state */

            if (
                activeIndex >= 0
            ) {{

                const node =
                    document.getElementById(
                        `pipeline-node-${{activeIndex}}`
                    );


                const card =
                    document.getElementById(
                        `pipeline-card-${{activeIndex}}`
                    );


                const className =
                    activeIndex === 7
                    ? "final-active"
                    : "active";


                if (node) {{

                    node.classList.add(
                        className
                    );

                }}


                if (card) {{

                    card.classList.add(
                        className
                    );

                }}

            }}


            previousActive =
                activeIndex;
        }}


        /* --------------------------------------------------
           BUILD TRAIL POINTS

           The trail END is always the exact same point
           as the moving dot.
        -------------------------------------------------- */

        function buildTrail(
            currentLength
        ) {{

            const trailStart =
                Math.max(
                    0,
                    currentLength
                    - trailLength
                );


            const points =
                [];


            for (
                let length = trailStart;
                length < currentLength;
                length += sampleStep
            ) {{

                const point =
                    path.getPointAtLength(
                        length
                    );


                points.push(
                    `${{point.x.toFixed(2)}},${{point.y.toFixed(2)}}`
                );

            }}


            /*
            Always append the dot's exact point last.
            This guarantees the trail is physically
            attached to the dot.
            */

            const finalPoint =
                path.getPointAtLength(
                    currentLength
                );


            points.push(
                `${{finalPoint.x.toFixed(2)}},${{finalPoint.y.toFixed(2)}}`
            );


            return points.join(
                " "
            );
        }}


        /* --------------------------------------------------
           ANIMATION LOOP
        -------------------------------------------------- */

        function animate(
            timestamp
        ) {{

            if (
                startTime === null
            ) {{

                startTime =
                    timestamp;

            }}


            const elapsed =
                (
                    timestamp
                    - startTime
                )
                % duration;


            const progress =
                elapsed
                / duration;


            const currentLength =
                progress
                * totalLength;


            /*
            Main dot position
            */

            const point =
                path.getPointAtLength(
                    currentLength
                );


            dot.setAttribute(
                "cx",
                point.x
            );


            dot.setAttribute(
                "cy",
                point.y
            );


            center.setAttribute(
                "cx",
                point.x
            );


            center.setAttribute(
                "cy",
                point.y
            );


            /*
            Trail derived from EXACTLY
            the same currentLength.
            */

            const trailPoints =
                buildTrail(
                    currentLength
                );


            trail.setAttribute(
                "points",
                trailPoints
            );


            trailGlow.setAttribute(
                "points",
                trailPoints
            );


            /*
            Synchronize card/node highlight
            with the same progress variable.
            */

            updateActiveStage(
                progress
            );


            requestAnimationFrame(
                animate
            );

        }}


        /* --------------------------------------------------
           REDUCED MOTION
        -------------------------------------------------- */

        const reducedMotion =
            window.matchMedia(
                "(prefers-reduced-motion: reduce)"
            );


        if (
            !reducedMotion.matches
        ) {{

            requestAnimationFrame(
                animate
            );

        }}


    }})();

    </script>


    </body>

    </html>
    """


    # ========================================================
    # STREAMLIT IFRAME
    # ========================================================

    st.iframe(
        html,
        width="stretch",
        height=265,
        tab_index=-1,
    )


# ============================================================
# SYSTEM STATUS
# Kept for compatibility if referenced elsewhere.
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
            f"{records:,} records · "
            f"{variables} variables",
            "Ready",
        ),
        (
            "Data Quality",
            f"{missing_values:,} missing cells "
            f"handled in pipeline",
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


    for (
        title,
        description,
        status,
    ) in status_items:

        cards += f"""
        <div class="system-card">

            <div class="system-dot">
            </div>

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
        display:
            grid;

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


    @media (
        max-width:
        1000px
    ) {{

        .system-grid {{
            grid-template-columns:
                repeat(2, 1fr);
        }}

    }}


    @media (
        max-width:
        600px
    ) {{

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