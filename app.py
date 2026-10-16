import streamlit as st
import folium
from streamlit_folium import st_folium
import os, glob
from PIL import Image

st.set_page_config(layout="wide", page_title="ClearSAR GeoAI Portal")

st.title("🛰️ ClearSAR: Interactive SAR-to-Optical Cloud Removal")
st.markdown("Reconstructing cloud-covered Sentinel-2 optical imagery using Sentinel-1 C-band SAR radar.")

st.sidebar.header("Model Configuration")
selected_arch = st.sidebar.selectbox("Choose Architecture", ["Model A (CNN U-Net)", "Model B (Hybrid Swin-U-Net)", "Model C (Pure ViT)"])
show_uncertainty = st.sidebar.checkbox("Show Aleatoric Uncertainty Map", value=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Geographic Location Selector")
    m = folium.Map(location=[31.7756, 117.4198], zoom_start=8)
    folium.Marker([31.7756, 117.4198], popup="Anhui Hotspot Scene", tooltip="Select ROI").add_to(m)
    st_folium(m, height=400, width="100%")

with col2:
    st.subheader("2. Model Visual Outputs")
    results_dir = "clearsar_final_output/results"
    
    plot_path = os.path.join(results_dir, "master_architecture_comparison.png")
    if os.path.exists(plot_path):
        st.image(plot_path, caption=f"Full Architecture Comparison Matrix ({selected_arch} Active)", use_container_width=True)
    else:
        st.info("Run Kaggle execution script to generate and download model checkpoints & visual results.")