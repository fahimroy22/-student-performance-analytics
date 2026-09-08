import streamlit as st


def show_footer():

    st.markdown("---")

    st.markdown(
        """
        <div style="
            text-align: center;
            color: #6B7280;
            font-size: 13px;
            padding: 4px 0 12px 0;
        ">
            Student Performance Analytics ·
            <b>MD. Fahim Ahmed</b> ·
            ID: 12411058
        </div>
        """,
        unsafe_allow_html=True
    )