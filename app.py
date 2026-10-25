"""
ClearSAR - frontend.

    streamlit run app.py

Reads outputs/final/ (the final model: extract clearsar_final_output.zip from notebooks/train_clearsar.py there)
and outputs/ (Phase 1: notebooks/kaggle_pipeline.py). Every page works with whatever is present.

  ClearSAR-Net         Home · Results · How it learned · Explore patches
  Expert system        Advisor (explainable decision support) · Knowledge base
  Earlier experiments  Phase 1 model comparison · Phase 1 experiments
  About                How it works
"""
import streamlit as st

st.set_page_config(page_title="ClearSAR", page_icon="🛰️", layout="wide", initial_sidebar_state="auto")

from frontend.common import inject_css, sidebar_source  # noqa: E402

pages = {
    "ClearSAR-Net": [
        st.Page("frontend/home.py", title="Home", icon=":material/satellite_alt:", default=True),
        st.Page("frontend/final_results.py", title="Results", icon=":material/leaderboard:"),
        st.Page("frontend/training.py", title="How it learned", icon=":material/monitoring:"),
        st.Page("frontend/explore.py", title="Explore patches", icon=":material/image_search:"),
    ],
    "Expert system": [
        st.Page("frontend/advisor.py", title="Advisor", icon=":material/psychology:"),
        st.Page("frontend/knowledge.py", title="Knowledge base", icon=":material/account_tree:"),
    ],
    "Earlier experiments": [
        st.Page("frontend/results.py", title="Phase 1 · model comparison", icon=":material/compare:"),
        st.Page("frontend/experiments.py", title="Phase 1 · experiments", icon=":material/science:"),
    ],
    "About": [
        st.Page("frontend/about.py", title="How it works", icon=":material/info:"),
    ],
}
nav = st.navigation(pages)
inject_css()
sidebar_source()
nav.run()
