# ============================================================
# REUSABLE SVG ICON SYSTEM
# ============================================================

import streamlit as st


# ------------------------------------------------------------
# SVG PATHS
# ------------------------------------------------------------

ICONS = {

    "dashboard":
        '<rect x="3" y="3" width="7" height="7" rx="1"/>'
        '<rect x="14" y="3" width="7" height="7" rx="1"/>'
        '<rect x="3" y="14" width="7" height="7" rx="1"/>'
        '<rect x="14" y="14" width="7" height="7" rx="1"/>',

    "chart":
        '<path d="M3 3v18h18"/>'
        '<path d="M7 16v-5"/>'
        '<path d="M12 16V7"/>'
        '<path d="M17 16v-8"/>',

    "settings":
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.4-2.4 1'
        'a8 8 0 0 0-1.7-1L14.5 3h-5L9 6.1a8 8 0 0 0-1.7 1'
        'l-2.4-1-2 3.4L5 11a7 7 0 0 0 0 2l-2.1 1.5 2 3.4'
        ' 2.4-1a8 8 0 0 0 1.7 1l.5 3.1h5l.5-3.1'
        'a8 8 0 0 0 1.7-1l2.4 1 2-3.4L19 13'
        'a7 7 0 0 0 0-1z"/>',

    "trending":
        '<path d="M3 17l6-6 4 4 8-8"/>'
        '<path d="M15 7h6v6"/>',

    "target":
        '<circle cx="12" cy="12" r="9"/>'
        '<circle cx="12" cy="12" r="5"/>'
        '<circle cx="12" cy="12" r="1"/>',

    "scale":
        '<path d="M12 3v18"/>'
        '<path d="M5 6h14"/>'
        '<path d="M5 6l-3 6h6L5 6z"/>'
        '<path d="M19 6l-3 6h6l-3-6z"/>'
        '<path d="M8 21h8"/>',

    "sparkles":
        '<path d="M12 3l1.4 3.6L17 8l-3.6 1.4L12 13'
        'l-1.4-3.6L7 8l3.6-1.4L12 3z"/>'
        '<path d="M19 14l.8 2.2L22 17l-2.2.8L19 20'
        'l-.8-2.2L16 17l2.2-.8L19 14z"/>',

    "database":
        '<ellipse cx="12" cy="5" rx="8" ry="3"/>'
        '<path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
        '<path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/>',

    "filter":
        '<path d="M4 5h16"/>'
        '<path d="M7 12h10"/>'
        '<path d="M10 19h4"/>',

    "wrench":
        '<path d="M14.7 6.3a4 4 0 0 0-5-5'
        'l2.1 2.1-2.8 2.8-2.1-2.1a4 4 0 0 0 5 5'
        'L19 16.2a2 2 0 1 1-2.8 2.8l-7.1-7.1"/>',

    "split":
        '<path d="M6 3v5c0 3 2 4 6 4h6"/>'
        '<path d="M15 9l3 3-3 3"/>'
        '<path d="M6 21v-5c0-2 1-3 3-4"/>',

    "sliders":
        '<path d="M4 6h10"/>'
        '<path d="M18 6h2"/>'
        '<circle cx="16" cy="6" r="2"/>'
        '<path d="M4 12h2"/>'
        '<path d="M10 12h10"/>'
        '<circle cx="8" cy="12" r="2"/>'
        '<path d="M4 18h8"/>'
        '<path d="M16 18h4"/>'
        '<circle cx="14" cy="18" r="2"/>',

    "brain":
        '<path d="M9 4a4 4 0 0 0-4 4v1'
        'a3 3 0 0 0 0 6v1a4 4 0 0 0 4 4"/>'
        '<path d="M15 4a4 4 0 0 1 4 4v1'
        'a3 3 0 0 1 0 6v1a4 4 0 0 1-4 4"/>'
        '<path d="M9 4v16"/>'
        '<path d="M15 4v16"/>',

    "check":
        '<circle cx="12" cy="12" r="9"/>'
        '<path d="M8 12l3 3 5-6"/>',

    "graduation":
        '<path d="M2 9l10-5 10 5-10 5L2 9z"/>'
        '<path d="M6 11v5c3 2 9 2 12 0v-5"/>'
        '<path d="M22 9v6"/>'
}


# ------------------------------------------------------------
# SVG GENERATOR
# ------------------------------------------------------------

def svg_icon(
    name,
    size=24,
    color="#355C7D",
    stroke_width=2
):

    paths = ICONS.get(
        name,
        ICONS["dashboard"]
    )

    return (
        f'<svg width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" '
        f'fill="none" '
        f'stroke="{color}" '
        f'stroke-width="{stroke_width}" '
        f'stroke-linecap="round" '
        f'stroke-linejoin="round" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'{paths}'
        f'</svg>'
    )


# ------------------------------------------------------------
# PAGE TITLE
# ------------------------------------------------------------

def page_title(
    icon_name,
    title,
    subtitle=None
):

    icon_svg = svg_icon(
        icon_name,
        size=31
    )

    html = (
        '<div style="display:flex;'
        'align-items:center;'
        'gap:11px;'
        'margin-bottom:4px;">'
        f'{icon_svg}'
        '<h1 style="margin:0;padding:0;">'
        f'{title}'
        '</h1>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )

    if subtitle:

        subtitle_html = (
            '<div style="'
            'color:#6B7280;'
            'font-size:14px;'
            'margin-top:3px;'
            'margin-bottom:20px;">'
            f'{subtitle}'
            '</div>'
        )

        st.markdown(
            subtitle_html,
            unsafe_allow_html=True
        )


# ------------------------------------------------------------
# SECTION TITLE
# ------------------------------------------------------------

def section_title(
    icon_name,
    title
):

    icon_svg = svg_icon(
        icon_name,
        size=21
    )

    html = (
        '<div style="'
        'display:flex;'
        'align-items:center;'
        'gap:8px;'
        'margin-top:18px;'
        'margin-bottom:10px;">'
        f'{icon_svg}'
        '<h3 style="margin:0;padding:0;">'
        f'{title}'
        '</h3>'
        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )


# ------------------------------------------------------------
# ICON CARD
# ------------------------------------------------------------

def icon_card(
    icon_name,
    title,
    description="",
    icon_size=25
):

    icon_svg = svg_icon(
        icon_name,
        size=icon_size
    )

    html = (
        '<div style="'
        'border:1px solid #E5E7EB;'
        'border-radius:10px;'
        'background:#FFFFFF;'
        'padding:16px;'
        'min-height:115px;">'

        '<div style="margin-bottom:10px;">'
        f'{icon_svg}'
        '</div>'

        '<div style="'
        'font-weight:600;'
        'color:#1F2937;'
        'margin-bottom:5px;">'
        f'{title}'
        '</div>'

        '<div style="'
        'color:#6B7280;'
        'font-size:13px;'
        'line-height:1.45;">'
        f'{description}'
        '</div>'

        '</div>'
    )

    st.markdown(
        html,
        unsafe_allow_html=True
    )