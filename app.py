"""
ClearSAR - frontend.

    streamlit run app.py

Reads the folder produced by notebooks/kaggle_pipeline.py (download clearsar_output.zip from
Kaggle and extract it into ./outputs/).

  Plan v4         Home · Building the model (stages 0-7) · Explore patches
  Phase 1         Model comparison · Experiments
  Expert system   Advisor (explainable decision support) · Knowledge base
  About           How it works
"""
import streamlit as st

st.set_page_config(page_title="ClearSAR", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

from frontend.common import inject_css, sidebar_source  # noqa: E402

pages = {
    "ClearSAR-Net (plan v4)": [
        st.Page("frontend/home.py", title="Home", icon=":material/satellite_alt:", default=True),
        st.Page("frontend/building.py", title="Building the model", icon=":material/construction:"),
        st.Page("frontend/explore.py", title="Explore patches", icon=":material/image_search:"),
    ],
    "Phase 1 · exploration": [
        st.Page("frontend/results.py", title="Model comparison", icon=":material/leaderboard:"),
        st.Page("frontend/experiments.py", title="Experiments", icon=":material/science:"),
    ],
    "Expert system": [
        st.Page("frontend/advisor.py", title="Advisor", icon=":material/psychology:"),
        st.Page("frontend/knowledge.py", title="Knowledge base", icon=":material/account_tree:"),
    ],
    "About": [
        st.Page("frontend/about.py", title="How it works", icon=":material/info:"),
    ],
}
nav = st.navigation(pages)
inject_css()
sidebar_source()
nav.run()
