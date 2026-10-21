"""
ClearSAR results portal.

    streamlit run app.py

Reads the folder produced by notebooks/kaggle_pipeline.py (download clearsar_output.zip
from Kaggle and extract it into ./outputs/). Pages:
    Overview        headline numbers + map of which scenes went to which split
    Results         model comparison: in-domain vs out-of-continent, cloud-cover bins, countries
    Training        validation curves and run statistics
    Gallery         test patches with every model's prediction side by side
    Live inference  run any trained checkpoint on a gallery patch or on your own GeoTIFFs
"""
import io
import json
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.dataset import N_S1, OUT_CHANNELS, RGB_IDX, preprocess_s1, preprocess_s2, to_rgb  # noqa: E402

st.set_page_config(page_title="ClearSAR", page_icon="🛰️", layout="wide")

SPLIT_NAMES = {"train": "Train", "val": "Validation", "test_id": "In-domain test (Asia)",
               "ood": "Out-of-continent test"}
SPLIT_COLORS = {"train": "#2563eb", "val": "#16a34a", "test_id": "#ea580c", "ood": "#9333ea"}
METRICS = {"psnr": ("PSNR (dB)", True), "ssim": ("SSIM", True), "sam": ("SAM (deg)", False),
           "mae": ("MAE", False), "rmse": ("RMSE", False), "fid": ("FID", False)}


# ------------------------------------------------------------------ loading
def find_run_root(base):
    """Accept the extracted folder itself or any parent up to two levels above it."""
    base = os.path.abspath(base)
    for cand in [base] + [os.path.join(base, d) for d in sorted(os.listdir(base))
                          if os.path.isdir(os.path.join(base, d))] if os.path.isdir(base) else []:
        if os.path.exists(os.path.join(cand, "results", "metrics_summary.csv")):
            return cand
    return None


@st.cache_data(show_spinner=False)
def read_csv(path, mtime):
    return pd.read_csv(path)


def load_csv(root, *parts):
    p = os.path.join(root, *parts)
    return read_csv(p, os.path.getmtime(p)) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def load_samples(path, mtime):
    z = np.load(path, allow_pickle=False)
    data = {k: z[k] for k in z.files if k != "meta"}
    return data, json.loads(str(z["meta"]))


@st.cache_resource(show_spinner=False)
def load_model(path, mtime):
    from src.models.architectures import load_model_from_checkpoint
    return load_model_from_checkpoint(path, "cpu")


def to_png(arr, cmap=None, vmax=None):
    """[H,W] or [H,W,3] float -> uint8 PIL image."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        hi = vmax if vmax is not None else (np.percentile(a, 99) or 1.0)
        a = np.clip(a / max(hi, 1e-6), 0, 1)
        if cmap:
            import matplotlib
            a = matplotlib.colormaps[cmap](a)[..., :3]
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def image_row(items, cols_per_row=5):
    """items: list of (image, caption)."""
    for i in range(0, len(items), cols_per_row):
        cols = st.columns(cols_per_row)
        for c, (img, cap) in zip(cols, items[i:i + cols_per_row]):
            c.image(img, caption=cap)


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛰️ ClearSAR")
st.sidebar.caption("SAR-to-optical cloud removal on SEN12MS-CR (Asia subset)")
base = st.sidebar.text_input("Results folder", value=os.environ.get("CLEARSAR_RESULTS", "outputs"),
                             help="Folder where you extracted clearsar_output.zip from Kaggle")
root = find_run_root(os.path.join(ROOT, base) if not os.path.isabs(base) else base)
if root:
    st.sidebar.success(f"Loaded run: `{os.path.relpath(root, ROOT) if root.startswith(ROOT) else root}`")
else:
    st.sidebar.warning("No results found yet.")

cfg = {}
if root and os.path.exists(os.path.join(root, "config.json")):
    with open(os.path.join(root, "config.json")) as f:
        cfg = json.load(f)
labels = cfg.get("labels", {})

st.title("ClearSAR: seeing through clouds with radar")
st.markdown("Reconstructing cloud-covered **Sentinel-2** optical imagery from **Sentinel-1** SAR plus the "
            "cloudy image, comparing a CNN U-Net, a Swin-Transformer U-Net and a pure Vision Transformer "
            "under one identical training protocol.")

if not root:
    st.info("**No results yet.** Run the Kaggle pipeline first:\n\n"
            "1. On Kaggle, attach the four SEN12MS-CR datasets, choose GPU **T4**, Internet **on**.\n"
            "2. Run `notebooks/kaggle_pipeline.py` (paste it into a cell, or `!python` it after cloning).\n"
            "3. Download `/kaggle/working/clearsar_output.zip` and extract it into `outputs/` next to `app.py`.\n"
            "4. Refresh this page.")
    census = pd.read_csv(os.path.join(ROOT, "data", "scans", "scenes.csv"))
    st.subheader("Scene census (data/scans/scenes.csv)")
    st.map(census.rename(columns={"lat": "latitude", "lon": "longitude"}), size=20000)
    st.dataframe(census.groupby("continent").agg(scenes=("scene", "count"), patches=("n_patches", "sum")))
    st.stop()

summary = load_csv(root, "results", "metrics_summary.csv")
scenes = load_csv(root, "results", "scene_splits.csv")
by_bin = load_csv(root, "results", "metrics_by_cloud_bin.csv")
by_country = load_csv(root, "results", "metrics_by_country.csv")
run_info = load_csv(root, "results", "run_info.csv")
unc = load_csv(root, "results", "uncertainty_summary.csv")
sparsif = load_csv(root, "results", "sparsification.csv")
summary["split_name"] = summary.split.map(SPLIT_NAMES)
model_rows = summary[summary.run != "identity"]

tab_over, tab_res, tab_train, tab_gal, tab_live = st.tabs(
    ["Overview", "Results", "Training", "Gallery", "Live inference"])

# ------------------------------------------------------------------ overview
with tab_over:
    ident = summary[summary.run == "identity"].set_index("split")
    best_id = model_rows[model_rows.split == "test_id"].sort_values("psnr", ascending=False)
    c = st.columns(4)
    if len(best_id):
        b = best_id.iloc[0]
        c[0].metric("Best in-domain PSNR", f"{b.psnr:.2f} dB",
                    f"{b.psnr - ident.loc['test_id', 'psnr']:+.2f} dB vs cloudy input" if "test_id" in ident.index else None)
        c[1].caption("Best model")
        c[1].markdown(f"**{b.label}**")
        ood = model_rows[(model_rows.run == b.run) & (model_rows.split == "ood")]
        if len(ood):
            c[2].metric("Same model, out-of-continent", f"{ood.iloc[0].psnr:.2f} dB",
                        f"{ood.iloc[0].psnr - b.psnr:+.2f} dB domain shift")
    if scenes is not None:
        c[3].metric("Scenes used", f"{len(scenes)}", f"{int(scenes.n_used.sum())} patches", delta_color="off")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Where the data comes from")
        if scenes is not None:
            import folium
            from streamlit_folium import st_folium
            m = folium.Map(location=[25, 60], zoom_start=2, tiles="OpenStreetMap")
            for r in scenes.itertuples():
                if pd.isna(r.lat):
                    continue
                folium.CircleMarker(
                    [r.lat, r.lon], radius=5 + 6 * min(1.0, r.n_used / max(1, scenes.n_used.max())),
                    color=SPLIT_COLORS.get(r.split, "#666"), fill=True, fill_opacity=0.8,
                    tooltip=f"{SPLIT_NAMES.get(r.split, r.split)} - {r.country} - {r.season} scene {r.scene}",
                    popup=f"{r.season} scene {r.scene}<br>{r.country} ({r.continent})<br>"
                          f"{int(r.n_used)} of {int(r.n_patches)} patches used").add_to(m)
            st_folium(m, height=430, use_container_width=True, returned_objects=[])
            st.markdown(" ".join(f"<span style='color:{SPLIT_COLORS[k]}'>●</span> {v} &nbsp;"
                                 for k, v in SPLIT_NAMES.items()), unsafe_allow_html=True)
    with right:
        st.subheader("How the data was split")
        st.markdown("Splits are made **by scene**, never by patch: neighbouring patches are near-duplicates, "
                    "so a random patch split would leak test information into training. One Indian scene is "
                    "always held out for an India-only slice, and one scene from each other continent forms "
                    "the **out-of-continent** domain-shift test.")
        if scenes is not None:
            t = scenes.groupby("split").agg(scenes=("scene", "count"), patches=("n_used", "sum"),
                                            countries=("country", lambda s: ", ".join(sorted(set(s)))))
            t = t.reindex([k for k in SPLIT_NAMES if k in t.index])
            t.index = t.index.map(SPLIT_NAMES)
            st.dataframe(t)
        p = os.path.join(root, "results", "master_architecture_comparison.png")
    if os.path.exists(p):
        st.subheader("All models on the same test patches")
        st.image(p)

# ------------------------------------------------------------------ results
with tab_res:
    st.subheader("Main results")
    st.caption("Mean over every test patch, all 13 bands in [0, 1]. Identity = output the cloudy input unchanged; "
               "a useful model must beat it. Differences under ~0.3 dB PSNR are within seed noise.")
    metric = st.radio("Metric", [m for m in METRICS if m in summary and summary[m].notna().any()],
                      format_func=lambda m: METRICS[m][0], horizontal=True)
    chart = alt.Chart(summary).mark_bar().encode(
        x=alt.X("label:N", title=None, sort=list(summary.label.unique()), axis=alt.Axis(labelAngle=-20, labelLimit=220)),
        xOffset="split_name:N",
        y=alt.Y(f"{metric}:Q", title=METRICS[metric][0], scale=alt.Scale(zero=metric != "psnr")),
        color=alt.Color("split_name:N", title="Test set",
                        scale=alt.Scale(domain=[SPLIT_NAMES["test_id"], SPLIT_NAMES["ood"]],
                                        range=[SPLIT_COLORS["test_id"], SPLIT_COLORS["ood"]])),
        tooltip=["label", "split_name", alt.Tooltip(f"{metric}:Q", format=".4f"), "n_patches"],
    ).properties(height=340)
    st.altair_chart(chart, width="stretch")
    st.caption(("Higher" if METRICS[metric][1] else "Lower") + " is better.")

    table = summary.pivot_table(index="label", columns="split_name",
                                values=[m for m in METRICS if m in summary], sort=False)
    table.columns = [f"{METRICS[m][0]} · {s}" for m, s in table.columns]
    st.dataframe(table.style.format("{:.3f}"))

    if by_bin is not None:
        st.subheader("Performance by cloud cover")
        st.caption("Cloud cover is estimated per patch (cloudy minus clear visible reflectance > 0.08); "
                   "used only for grouping results.")
        split = st.selectbox("Test set", ["test_id", "ood"], format_func=SPLIT_NAMES.get, key="bin_split")
        d = by_bin[by_bin.split == split]
        st.altair_chart(alt.Chart(d).mark_line(point=True).encode(
            x=alt.X("cloud_bin:N", title="Cloud cover", sort=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]),
            y=alt.Y(f"{metric if metric != 'fid' else 'psnr'}:Q", scale=alt.Scale(zero=False),
                    title=METRICS[metric if metric != "fid" else "psnr"][0]),
            color=alt.Color("label:N", title="Model"),
            tooltip=["label", "cloud_bin", "n_patches", alt.Tooltip("psnr:Q", format=".2f")],
        ).properties(height=320), width="stretch")

    if by_country is not None:
        st.subheader("By country")
        split = st.selectbox("Test set ", ["test_id", "ood"], format_func=SPLIT_NAMES.get, key="cty_split")
        d = by_country[by_country.split == split].pivot_table(index="country", columns="label", values="psnr",
                                                              sort=False)
        st.caption("PSNR (dB). IN = the held-out Indian scene.")
        st.dataframe(d.style.format("{:.2f}").background_gradient(axis=None, cmap="Greens"))

    if unc is not None:
        st.subheader("Uncertainty (Model A + variance head)")
        st.caption("coverage95: fraction of true values inside the predicted 95% interval (ideal 0.95). "
                   "AUSE: area between the sparsification curve and the oracle (lower is better).")
        u = unc.copy()
        u["split"] = u.split.map(SPLIT_NAMES)
        st.dataframe(u.drop(columns=["run"]).set_index("split").style.format("{:.4f}"))
        if sparsif is not None:
            s = sparsif.melt(id_vars=["split", "fraction"], value_vars=["model_mae", "oracle_mae"],
                             var_name="curve", value_name="mae")
            s["split"] = s.split.map(SPLIT_NAMES)
            st.altair_chart(alt.Chart(s).mark_line().encode(
                x=alt.X("fraction:Q", title="fraction of most-uncertain pixels removed"),
                y=alt.Y("mae:Q", title="MAE of remaining pixels"),
                color="split:N", strokeDash="curve:N").properties(height=280), width="stretch")

    with st.expander("Static figures from the Kaggle run"):
        for f in ["metrics_comparison.png", "cloud_bin_psnr.png", "sparsification.png"]:
            p = os.path.join(root, "results", f)
            if os.path.exists(p):
                st.image(p, caption=f)
    with st.expander("Download tables"):
        for f in sorted(os.listdir(os.path.join(root, "results"))):
            if f.endswith(".csv"):
                with open(os.path.join(root, "results", f), "rb") as fh:
                    st.download_button(f, fh.read(), file_name=f, mime="text/csv", key=f"dl_{f}")

# ------------------------------------------------------------------ training
with tab_train:
    hist = []
    for f in sorted(os.listdir(os.path.join(root, "logs"))):
        if f.startswith("history_") and f.endswith(".csv"):
            h = load_csv(root, "logs", f)
            run = f[len("history_"):-4]
            h["label"] = labels.get(run, run)
            hist.append(h)
    if hist:
        h = pd.concat(hist)
        ycol = st.radio("Curve", ["val_l1", "val_psnr", "val_ssim", "train_rec", "grad_norm"], horizontal=True,
                        format_func=lambda c: c.replace("_", " ").replace("val", "validation").replace(
                            "rec", "reconstruction loss"))
        st.altair_chart(alt.Chart(h).mark_line(point=True).encode(
            x=alt.X("step:Q", title="training step"), y=alt.Y(f"{ycol}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("label:N", title="Run"), tooltip=["label", "step", alt.Tooltip(f"{ycol}:Q", format=".4f")],
        ).properties(height=360), width="stretch")
    if run_info is not None:
        st.subheader("Runs")
        ri = run_info.copy()
        ri["params (M)"] = ri.params / 1e6
        st.dataframe(ri[["label", "params (M)", "steps", "best_step", "best_val_l1", "best_val_psnr",
                         "train_minutes", "peak_gpu_gb"]].set_index("label").style.format("{:.3f}", subset=[
                             "params (M)", "best_val_l1", "best_val_psnr", "train_minutes", "peak_gpu_gb"]))
    if cfg:
        with st.expander("Full run configuration (config.json)"):
            st.json(cfg)

# ------------------------------------------------------------------ gallery
samples_path = os.path.join(root, "results", "samples.npz")
samples = load_samples(samples_path, os.path.getmtime(samples_path)) if os.path.exists(samples_path) else None

with tab_gal:
    if samples is None:
        st.info("samples.npz not found in this run.")
    else:
        data, meta = samples
        sm = meta["samples"]
        names = [f"{i + 1}. {SPLIT_NAMES[s['split']]} · {s['country']} · {s['season']} scene {s['scene']} "
                 f"p{s['patch']} · {int(s['cloud_frac'] * 100)}% cloud" for i, s in enumerate(sm)]
        i = st.selectbox("Test patch", range(len(sm)), format_func=lambda k: names[k])
        s = sm[i]
        x, y = data["x"][i].astype(np.float32), data["y"][i].astype(np.float32)
        runs = [r for r in meta["runs"] if f"pred_{r}" in data]
        chosen = st.multiselect("Models", runs, default=runs, format_func=lambda r: meta["labels"].get(r, r))
        items = [(to_png(x[0], vmax=1.0), "SAR VV"), (to_png(x[1], vmax=1.0), "SAR VH"),
                 (to_png(to_rgb(x[N_S1:])), "Cloudy input"), (to_png(to_rgb(y)), "Target (clear)")]
        for r in chosen:
            sc = s["scores"].get(r, {})
            cap = meta["labels"].get(r, r) + (f" · {sc['psnr']:.2f} dB" if sc else "")
            items.append((Image.fromarray(data[f"pred_{r}"][i]), cap))
        if st.checkbox("Show aleatoric uncertainty map", value=True) and np.any(data["sigma"][i]):
            items.append((to_png(data["sigma"][i].astype(np.float32), cmap="magma"),
                          "Uncertainty b (bright = unsure)"))
        image_row(items, 4)
        rows = [dict(model=meta["labels"].get(r, r), **v) for r, v in s["scores"].items()]
        st.dataframe(pd.DataFrame(rows).set_index("model").style.format("{:.4f}")
                     .highlight_max(subset=["psnr", "ssim"], color="#bbf7d0")
                     .highlight_min(subset=["sam", "mae", "rmse"], color="#bbf7d0"))

# ------------------------------------------------------------------ live inference
with tab_live:
    st.markdown("Run a trained checkpoint on the CPU. Each 256×256 patch is cut into four 128×128 tiles, exactly "
                "as during evaluation.")
    ck_dir = os.path.join(root, "checkpoints")
    cks = sorted(f for f in os.listdir(ck_dir) if f.startswith("best_") and f.endswith(".pt")) \
        if os.path.isdir(ck_dir) else []
    if not cks:
        st.info("No checkpoints in this run folder.")
    else:
        ck = st.selectbox("Checkpoint", cks, format_func=lambda f: labels.get(f[5:-3], f[5:-3]))
        source = st.radio("Input", ["Gallery patch", "Upload GeoTIFFs"], horizontal=True)
        x = y = profile = None
        if source == "Gallery patch":
            if samples is None:
                st.warning("No gallery samples in this run - upload GeoTIFFs instead.")
            else:
                data, meta = samples
                k = st.selectbox("Patch", range(len(meta["samples"])), key="live_patch",
                                 format_func=lambda j: f"{j + 1}. {SPLIT_NAMES[meta['samples'][j]['split']]} · "
                                                       f"{meta['samples'][j]['country']}")
                x, y = data["x"][k].astype(np.float32), data["y"][k].astype(np.float32)
        else:
            st.caption("SEN12MS-CR style patches: S1 (2 bands, dB), S2 cloudy (13 bands), optionally S2 clear "
                       "(13 bands) to score the prediction.")
            c1, c2, c3 = st.columns(3)
            f1 = c1.file_uploader("S1 (.tif)", type=["tif", "tiff"])
            f2 = c2.file_uploader("S2 cloudy (.tif)", type=["tif", "tiff"])
            f3 = c3.file_uploader("S2 clear, optional (.tif)", type=["tif", "tiff"])
            if f1 and f2:
                from rasterio.io import MemoryFile
                try:
                    with MemoryFile(f1.getvalue()) as mf, mf.open() as ds:
                        s1 = ds.read()[:2]
                    with MemoryFile(f2.getvalue()) as mf, mf.open() as ds:
                        s2c, profile = ds.read()[:13], ds.profile
                    if s1.shape[0] < 2 or s2c.shape[0] < 13 or s1.shape[1:] != s2c.shape[1:]:
                        raise ValueError(f"need 2 S1 + 13 S2 bands of equal size, got {s1.shape} / {s2c.shape}")
                    x = np.concatenate([preprocess_s1(s1), preprocess_s2(s2c)], axis=0)
                    if f3:
                        with MemoryFile(f3.getvalue()) as mf, mf.open() as ds:
                            y = preprocess_s2(ds.read()[:13])
                except Exception as e:
                    st.error(f"Could not read the GeoTIFFs: {e}")
                    x = None
        if x is not None and st.button("Run model", type="primary"):
            import torch
            from src.evaluation.evaluate import predict_tiled
            from src.evaluation.metrics import compute_metrics
            p = os.path.join(ck_dir, ck)
            model, meta_ck = load_model(p, os.path.getmtime(p))
            with st.spinner("Predicting..."), torch.no_grad():
                out = predict_tiled(model, torch.from_numpy(x)[None])[0]
            pred = out[:OUT_CHANNELS].numpy()
            items = [(to_png(x[0], vmax=1.0), "SAR VV"), (to_png(to_rgb(x[N_S1:])), "Cloudy input"),
                     (to_png(to_rgb(pred)), f"Prediction · {labels.get(ck[5:-3], ck)}")]
            if y is not None:
                items.append((to_png(to_rgb(y)), "Target (clear)"))
            if meta_ck.get("uncertainty"):
                items.append((to_png(np.exp(out[OUT_CHANNELS:].clamp(-10, 10).numpy()).mean(0), cmap="magma"),
                              "Uncertainty b"))
            image_row(items, len(items))
            if y is not None:
                m = compute_metrics(torch.from_numpy(pred)[None], torch.from_numpy(y)[None])
                mi = compute_metrics(torch.from_numpy(x[N_S1:N_S1 + OUT_CHANNELS])[None], torch.from_numpy(y)[None])
                cols = st.columns(5)
                for c, k in zip(cols, ["psnr", "ssim", "sam", "mae", "rmse"]):
                    c.metric(METRICS[k][0], f"{m[k][0]:.3f}", f"{m[k][0] - mi[k][0]:+.3f} vs cloudy",
                             delta_color="normal" if METRICS[k][1] else "inverse")
            buf = io.BytesIO()
            to_png(to_rgb(pred)).save(buf, format="PNG")
            d1, d2 = st.columns(2)
            d1.download_button("Download RGB preview (.png)", buf.getvalue(), "clearsar_prediction.png", "image/png")
            if profile is not None:
                from rasterio.io import MemoryFile
                prof = dict(profile, count=OUT_CHANNELS, dtype="uint16")
                with MemoryFile() as mf:
                    with mf.open(**prof) as dst:
                        dst.write((np.clip(pred, 0, 1) * 10000).astype(np.uint16))
                    d2.download_button("Download 13-band prediction (.tif, georeferenced)", mf.read(),
                                       "clearsar_prediction.tif", "image/tiff")
