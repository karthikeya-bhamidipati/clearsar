import numpy as np
import pandas as pd
import streamlit as st

from frontend.common import (CORE, SPLIT_COLORS, SPLIT_NAMES, best_main_model, card, compare_slider, hero, kpi,
                             labels, load, no_results, rgb_u8, sample_names, section)

root, data = load()
hero("SAR-to-optical cloud removal · Sentinel-1 + Sentinel-2 · Asia",
     "Seeing the ground <span class='cs-grad'>through the clouds</span>",
     "Clouds hide most optical satellite images. Radar sees straight through them but has no colour. "
     "ClearSAR trains three deep-learning architectures to fuse the two and reconstruct the cloud-free "
     "image - then an expert system decides how far each reconstruction can be trusted.")
if data is None:
    no_results()

lab = labels(data)
main = data["main"]
best = best_main_model(main)
ident = main[main.group == "identity"].set_index("split")

# ---------------------------------------------------------------- before / after
s, meta = data["samples"], data["samples_meta"]
if s is not None and best is not None:
    names = sample_names(meta)
    left, right = st.columns([1.35, 1], gap="large")
    with right:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        i = st.selectbox("Test patch", range(len(names)), format_func=lambda k: names[k], key="home_patch")
        runs = [r for r in meta["runs"] if f"pred_{r}" in s and r != "identity"]
        run = st.selectbox("Model", runs, index=runs.index(best.group) if best.group in runs else 0,
                           format_func=lambda r: lab.get(r, r), key="home_model")
        sc = meta["samples"][i]["scores"]
        ps, pc = sc.get(run, {}).get("psnr"), sc.get("identity", {}).get("psnr")
        st.markdown(
            f"<div class='cs-card'><h4>{lab.get(run, run)}</h4><p>Drag the handle to compare the cloudy "
            f"Sentinel-2 input with the reconstruction. This patch was never seen during training.</p>"
            f"<p style='margin-top:.7rem'><b style='color:#E6EAF2'>{ps:.2f} dB</b> PSNR for the reconstruction vs "
            f"<b style='color:#E6EAF2'>{pc:.2f} dB</b> for the cloudy image itself.</p></div>"
            if ps is not None and pc is not None else "", unsafe_allow_html=True)
        with st.expander("What the model sees"):
            x = s["x"][i].astype(np.float32)
            c1, c2 = st.columns(2)
            c1.image(np.clip(x[0], 0, 1), caption="Radar VV")
            c2.image(rgb_u8(s["y"][i]), caption="True clear image")
    with left:
        x = s["x"][i].astype(np.float32)
        compare_slider(rgb_u8(x[2:]), s[f"pred_{run}"][i], "Cloudy input", lab.get(run, run), size=560)

# ---------------------------------------------------------------- headline numbers
section("At a glance", "Mean over every held-out test patch, all 13 spectral bands.")
c = st.columns(4)
if best is not None:
    kpi(c[0], "Best architecture", lab.get(best.group, best.group).replace("Model ", ""),
        f"{best.psnr:.2f} dB PSNR on Asian test scenes")
    if "test_id" in ident.index:
        kpi(c[1], "Gain over the cloudy image", f"+{best.psnr - ident.loc['test_id', 'psnr']:.1f} dB",
            f"cloudy input alone: {ident.loc['test_id', 'psnr']:.1f} dB")
    ood = main[(main.group == best.group) & (main.split == "ood")]
    if len(ood):
        kpi(c[2], "On other continents", f"{ood.iloc[0].psnr:.2f} dB",
            f"{ood.iloc[0].psnr - best.psnr:+.2f} dB vs Asia (domain shift)")
sc = data["scene_splits"]
if sc is not None:
    kpi(c[3], "Data used", f"{int(sc.n_used.sum()):,}", f"patches from {len(sc)} scenes, split by scene")

# ---------------------------------------------------------------- the three architectures
section("Three architectures, one protocol",
        "Same data, same budget, same loss - only the network changes, so differences come from the design.")
desc = {
    "A_unet": ("Model A · U-Net", "A convolutional encoder-decoder. Sees small neighbourhoods and builds up - "
               "a strong built-in assumption that nearby pixels belong together."),
    "B_swin": ("Model B · Swin U-Net", "The same U-Net with shifted-window self-attention in the bottleneck, "
               "so distant regions can exchange information - useful under large clouds."),
    "C_vit": ("Model C · Pure ViT", "No convolutions at all: 8×8 patches that all attend to each other. "
              "Maximum flexibility, but it must learn from data that neighbours are related."),
}
cols = st.columns(3, gap="medium")
for col, g in zip(cols, CORE):
    row = main[(main.group == g) & (main.split == "test_id")]
    extra = f"<br><br><b style='color:#2DD4BF'>{row.iloc[0].psnr:.2f} dB</b> · SSIM {row.iloc[0].ssim:.3f}" if len(row) else ""
    card(col, desc[g][0], desc[g][1] + extra)

# ---------------------------------------------------------------- map
if sc is not None and sc.lat.notna().any():
    section("Where the data comes from",
            "Each dot is one ~50 km scene. Training, validation and Asian test scenes never overlap; four scenes "
            "from other continents are used only to test generalisation.")
    import folium
    from streamlit_folium import st_folium
    m = folium.Map(location=[22, 40], zoom_start=2, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                   attr="Tiles &copy; Esri", min_zoom=2, world_copy_jump=True)
    for r in sc.itertuples():
        if pd.isna(r.lat):
            continue
        folium.CircleMarker([r.lat, r.lon], radius=6, color=SPLIT_COLORS.get(r.split, "#888"), fill=True,
                            fill_opacity=.85, weight=1,
                            tooltip=f"{SPLIT_NAMES.get(r.split, r.split)} · {r.country} · {int(r.n_used)} patches").add_to(m)
    st_folium(m, height=380, use_container_width=True, returned_objects=[])
    st.markdown(" ".join(f"<span style='color:{SPLIT_COLORS[k]}'>●</span>&nbsp;{v}&nbsp;&nbsp;"
                         for k, v in SPLIT_NAMES.items()), unsafe_allow_html=True)
