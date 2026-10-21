"""
ClearSAR results portal.

    streamlit run app.py

Reads the folder produced by notebooks/kaggle_pipeline.py (download clearsar_output.zip
from Kaggle and extract it into ./outputs/). Pages:
    Overview        headline numbers + map of which scenes went to which split
    Results         model comparison: in-domain vs out-of-continent, cloud-cover bins, countries
    Experiments     E1-E7 ablations vs their reference, leakage check, receptive fields, uncertainty
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
main_tbl = load_csv(root, "results", "metrics_main.csv")
experiments = load_csv(root, "results", "experiments.csv")
lcurves = load_csv(root, "results", "learning_curves.csv")
erf_tbl = load_csv(root, "results", "erf_stats.csv")
scenes = load_csv(root, "results", "scene_splits.csv")
by_bin = load_csv(root, "results", "metrics_by_cloud_bin.csv")
by_country = load_csv(root, "results", "metrics_by_country.csv")
run_info = load_csv(root, "results", "run_info.csv")
unc = load_csv(root, "results", "uncertainty_summary.csv")
sparsif = load_csv(root, "results", "sparsification.csv")
summary["split_name"] = summary.split.map(SPLIT_NAMES)
# The headline comparison: mean over seeds when available (older runs: one row per run).
if main_tbl is None:
    main_tbl = summary[summary.split.isin(["test_id", "ood"])].rename(columns={"run": "group"})
main_tbl = main_tbl.copy()
main_tbl["split_name"] = main_tbl.split.map(SPLIT_NAMES)
MAIN_RUNS = set(main_tbl.group)
model_rows = main_tbl[~main_tbl.group.isin(["identity", "tiny_cnn"])]


def fig(name, caption=None):
    p = os.path.join(root, "results", name)
    if os.path.exists(p):
        st.image(p, caption=caption)


tab_over, tab_res, tab_exp, tab_train, tab_gal, tab_live = st.tabs(
    ["Overview", "Results", "Experiments", "Training", "Gallery", "Live inference"])

# ------------------------------------------------------------------ overview
with tab_over:
    ident = main_tbl[main_tbl.group == "identity"].set_index("split")
    best_id = model_rows[model_rows.split == "test_id"].sort_values("psnr", ascending=False)
    c = st.columns(4)
    if len(best_id):
        b = best_id.iloc[0]
        c[0].metric("Best in-domain PSNR", f"{b.psnr:.2f} dB",
                    f"{b.psnr - ident.loc['test_id', 'psnr']:+.2f} dB vs cloudy input" if "test_id" in ident.index else None)
        c[1].caption("Best model")
        c[1].markdown(f"**{b.label}**")
        ood = model_rows[(model_rows.group == b.group) & (model_rows.split == "ood")]
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
                    "so a random patch split would leak test information into training (the *Experiments* tab "
                    "measures exactly how much). One Indian scene is always held out for an India-only slice, "
                    "and one scene from each other continent forms the **out-of-continent** domain-shift test.")
        if scenes is not None:
            t = scenes.groupby("split").agg(scenes=("scene", "count"), patches=("n_used", "sum"),
                                            countries=("country", lambda s: ", ".join(sorted(set(s)))))
            t = t.reindex([k for k in SPLIT_NAMES if k in t.index])
            t.index = t.index.map(SPLIT_NAMES)
            st.dataframe(t)
    st.subheader("All models on the same test patches")
    fig("master_architecture_comparison.png")

# ------------------------------------------------------------------ results
with tab_res:
    st.subheader("Main results")
    st.caption("Mean over every test patch, all 13 bands in [0, 1]; error bars = spread over training seeds "
               "where several seeds were run. Identity = output the cloudy input unchanged; a useful model "
               "must beat it. Treat differences under ~0.3 dB PSNR as noise.")
    metric = st.radio("Metric", [m for m in METRICS if m in main_tbl and main_tbl[m].notna().any()],
                      format_func=lambda m: METRICS[m][0], horizontal=True)
    order = list(main_tbl.label.unique())
    bars = alt.Chart(main_tbl).mark_bar().encode(
        x=alt.X("label:N", title=None, sort=order, axis=alt.Axis(labelAngle=-30, labelLimit=220, labelOverlap=False)),
        xOffset=alt.XOffset("split_name:N"),
        y=alt.Y(f"{metric}:Q", title=METRICS[metric][0], scale=alt.Scale(zero=metric != "psnr")),
        color=alt.Color("split_name:N", title="Test set",
                        scale=alt.Scale(domain=[SPLIT_NAMES["test_id"], SPLIT_NAMES["ood"]],
                                        range=[SPLIT_COLORS["test_id"], SPLIT_COLORS["ood"]])),
        tooltip=["label", "split_name", alt.Tooltip(f"{metric}:Q", format=".4f")] +
                (["n_seeds"] if "n_seeds" in main_tbl else []))
    layers = [bars]
    if f"{metric}_std" in main_tbl and main_tbl[f"{metric}_std"].notna().any():
        eb = main_tbl.assign(lo=main_tbl[metric] - main_tbl[f"{metric}_std"].fillna(0),
                             hi=main_tbl[metric] + main_tbl[f"{metric}_std"].fillna(0))
        layers.append(alt.Chart(eb).mark_rule(color="black").encode(
            x=alt.X("label:N", sort=order), xOffset="split_name:N", y="lo:Q", y2="hi:Q"))
    st.altair_chart(alt.layer(*layers).properties(height=340), width="stretch")
    st.caption(("Higher" if METRICS[metric][1] else "Lower") + " is better.")

    vals = [m for m in METRICS if m in main_tbl]
    table = main_tbl.pivot_table(index="label", columns="split_name", values=vals, sort=False)
    table.columns = [f"{METRICS[m][0]} · {s}" for m, s in table.columns]
    st.dataframe(table.style.format("{:.3f}"))

    if by_bin is not None:
        st.subheader("Performance by cloud cover")
        st.caption("Cloud cover is estimated per patch (cloudy minus clear visible reflectance > 0.08); "
                   "used only for grouping results.")
        split = st.selectbox("Test set", ["test_id", "ood"], format_func=SPLIT_NAMES.get, key="bin_split")
        d = by_bin[(by_bin.split == split) & by_bin.run.isin(MAIN_RUNS)]
        mm = metric if metric != "fid" else "psnr"
        st.altair_chart(alt.Chart(d).mark_line(point=True).encode(
            x=alt.X("cloud_bin:N", title="Cloud cover", sort=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]),
            y=alt.Y(f"{mm}:Q", scale=alt.Scale(zero=False), title=METRICS[mm][0]),
            color=alt.Color("label:N", title="Model"),
            tooltip=["label", "cloud_bin", "n_patches", alt.Tooltip(f"{mm}:Q", format=".3f")],
        ).properties(height=320), width="stretch")

    if by_country is not None:
        st.subheader("By country")
        split = st.selectbox("Test set ", ["test_id", "ood"], format_func=SPLIT_NAMES.get, key="cty_split")
        d = by_country[(by_country.split == split) & by_country.run.isin(MAIN_RUNS)].pivot_table(
            index="country", columns="label", values="psnr", sort=False)
        st.caption("PSNR (dB). IN = the held-out Indian scene.")
        st.dataframe(d.style.format("{:.2f}").background_gradient(axis=None, cmap="Greens"))

    with st.expander("Every run (all seeds and ablations)"):
        st.dataframe(summary.drop(columns=["split_name"]).style.format(precision=4))
    with st.expander("Download tables"):
        for f in sorted(os.listdir(os.path.join(root, "results"))):
            if f.endswith(".csv"):
                with open(os.path.join(root, "results", f), "rb") as fh:
                    st.download_button(f, fh.read(), file_name=f, mime="text/csv", key=f"dl_{f}")

# ------------------------------------------------------------------ experiments
with tab_exp:
    st.markdown("Every experiment changes **one thing** relative to a reference run and is trained under the "
                "identical protocol. Δ = experiment − reference (PSNR/SSIM: positive is better; SAM/MAE/FID: "
                "negative is better). The *prediction* column is the hypothesis from `docs/project_plan.md` §5 - "
                "write down whether it held, and why.")
    if experiments is None or experiments.empty:
        st.info("No experiment results in this run folder.")
    else:
        ex = experiments.copy()
        ex["question"] = ex["question"].fillna("")
        ex["split_name"] = ex.split.map(lambda s: SPLIT_NAMES.get(s, "Random-split test"))
        show = ex[["experiment", "label", "reference_label", "split_name", "psnr", "delta_psnr", "delta_ssim",
                   "delta_sam", "delta_fid", "question"]].rename(columns={
                       "label": "run", "reference_label": "vs", "split_name": "test set", "psnr": "PSNR",
                       "delta_psnr": "Δ PSNR", "delta_ssim": "Δ SSIM", "delta_sam": "Δ SAM", "delta_fid": "Δ FID",
                       "question": "prediction"})
        st.dataframe(show.style.format({"PSNR": "{:.2f}", "Δ PSNR": "{:+.2f}", "Δ SSIM": "{:+.4f}",
                                        "Δ SAM": "{:+.2f}", "Δ FID": "{:+.1f}"})
                     .background_gradient(subset=["Δ PSNR"], cmap="RdYlGn", vmin=-2, vmax=2), height=420)

    if lcurves is not None and (lcurves.experiment == "E1").any():
        st.subheader("E1 - learning curves (the most important experiment)")
        st.caption("Same models trained on nested 25% / 50% / 100% subsets of the training scenes' patches. "
                   "Prediction: A is best with little data; C improves fastest as data grows.")
        split = st.selectbox("Test set", ["test_id", "ood"], format_func=SPLIT_NAMES.get, key="lc_split")
        d = lcurves[lcurves.split == split].assign(percent=lambda t: t.fraction * 100)
        st.altair_chart(alt.Chart(d).mark_line(point=True).encode(
            x=alt.X("percent:Q", title="% of training patches", scale=alt.Scale(domain=[0, 105])),
            y=alt.Y("psnr:Q", title="PSNR (dB)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title="Model"),
            tooltip=["model", "percent", alt.Tooltip("psnr:Q", format=".2f"), alt.Tooltip("ssim:Q", format=".4f")],
        ).properties(height=320), width="stretch")

    if experiments is not None and len(experiments) and (experiments.experiment == "leakage").any():
        st.subheader("Leakage check - random patch split vs scene split")
        lk = experiments[experiments.experiment == "leakage"].iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Scene split (honest)", f"{lk.ref_psnr:.2f} dB")
        c2.metric("Random patch split", f"{lk.psnr:.2f} dB", f"{lk.delta_psnr:+.2f} dB inflation", delta_color="inverse")
        c3.caption("Same model and budget. With a random split, near-identical neighbouring patches land in both "
                   "train and test, so the score overstates how well the model generalises to new places.")

    if erf_tbl is not None:
        st.subheader("E3c - effective receptive field")
        st.caption("Gradient of the centre output pixel w.r.t. the input (log scale). r50/r90 = radius containing "
                   "50% / 90% of the influence. Usually far smaller than the theoretical receptive field.")
        fig("erf.png")
        st.dataframe(erf_tbl.set_index("label").drop(columns=["run"]).style.format("{:.1f}"))

    if unc is not None:
        st.subheader("E7 - uncertainty: variance head vs MC dropout vs ensemble")
        st.caption("coverage95: fraction of true values inside the predicted 95% interval (ideal 0.95). "
                   "mean_sigma rising from in-domain to out-of-continent = the method notices domain shift. "
                   "AUSE: area between the sparsification curve and the oracle (lower = better error ranking).")
        u = unc.copy()
        if "method" not in u:
            u["method"] = "Variance head (aleatoric)"
        u["split"] = u.split.map(SPLIT_NAMES)
        cols = [c for c in ["method", "split", "coverage95", "median_sigma", "mean_sigma", "mean_abs_error", "spearman_err_sigma",
                            "ause"] if c in u]
        st.dataframe(u[cols].set_index(cols[:2]).style.format("{:.4f}"))
        if sparsif is not None:
            s = sparsif.copy()
            if "method" not in s:
                s["method"] = "Variance head (aleatoric)"
            s["split"] = s.split.map(SPLIT_NAMES)
            s = s.melt(id_vars=["method", "split", "fraction"], value_vars=["model_mae", "oracle_mae"],
                       var_name="curve", value_name="mae")
            split = st.selectbox("Test set", list(s.split.unique()), key="sp_split")
            st.altair_chart(alt.Chart(s[s.split == split]).mark_line().encode(
                x=alt.X("fraction:Q", title="fraction of most-uncertain pixels removed"),
                y=alt.Y("mae:Q", title="MAE of remaining pixels", scale=alt.Scale(zero=False)),
                color="method:N", strokeDash="curve:N").properties(height=300), width="stretch")

    if "n_seeds" in main_tbl and (main_tbl.n_seeds > 1).any():
        st.subheader("Seed variance (the noise floor)")
        sv = main_tbl[main_tbl.n_seeds > 1][["label", "split_name", "n_seeds", "psnr", "psnr_std", "ssim", "ssim_std"]]
        st.dataframe(sv.set_index(["label", "split_name"]).style.format(precision=4))
        st.caption("Differences between models smaller than about 2x this spread are not meaningful.")

    with st.expander("Static figures"):
        for f, cap in [("learning_curves.png", "E1"), ("uncertainty_methods.png", "E7"),
                       ("metrics_comparison.png", "Main comparison"), ("cloud_bin_psnr.png", "Cloud bins")]:
            fig(f, cap)

# ------------------------------------------------------------------ training
with tab_train:
    hist = []
    for f in sorted(os.listdir(os.path.join(root, "logs"))):
        if f.startswith("history_") and f.endswith(".csv"):
            h = load_csv(root, "logs", f).copy()
            run = f[len("history_"):-4]
            h["run"], h["label"] = run, labels.get(run, run)
            hist.append(h)
    if hist:
        h = pd.concat(hist)
        all_runs = list(dict.fromkeys(h.run))
        default = [r for r in all_runs if r in MAIN_RUNS] or all_runs
        pick = st.multiselect("Runs", all_runs, default=default, format_func=lambda r: labels.get(r, r))
        ycol = st.radio("Curve", ["val_l1", "val_psnr", "val_ssim", "train_rec", "grad_norm"], horizontal=True,
                        format_func=lambda c: c.replace("_", " ").replace("val", "validation").replace(
                            "rec", "reconstruction loss"))
        st.altair_chart(alt.Chart(h[h.run.isin(pick)]).mark_line(point=True).encode(
            x=alt.X("step:Q", title="training step"), y=alt.Y(f"{ycol}:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("label:N", title="Run"), tooltip=["label", "step", alt.Tooltip(f"{ycol}:Q", format=".4f")],
        ).properties(height=360), width="stretch")
    if run_info is not None and "params" in run_info:
        st.subheader("Runs")
        ri = run_info.dropna(subset=["params"]).copy()
        ri["params (M)"] = ri.params / 1e6
        cols = [c for c in ["label", "params (M)", "steps", "steps_done", "best_step", "best_val_l1", "best_val_psnr",
                            "train_minutes", "peak_gpu_gb", "stopped_early"] if c in ri]
        st.dataframe(ri[cols].set_index("label").style.format(precision=3))
    if cfg:
        if cfg.get("status"):
            with st.expander("Job status (what ran, what was skipped)"):
                st.dataframe(pd.Series(cfg["status"], name="status"))
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
        if "sigma" in data and np.any(data["sigma"][i]):          # older single-uncertainty runs
            items.append((to_png(data["sigma"][i].astype(np.float32), cmap="magma"), "Uncertainty"))
        sig_runs = [r for r in runs if f"sigma_{r}" in data and np.any(data[f"sigma_{r}"][i])]
        for r in st.multiselect("Uncertainty maps", sig_runs, default=sig_runs,
                                format_func=lambda r: meta["labels"].get(r, r)):
            items.append((to_png(data[f"sigma_{r}"][i].astype(np.float32), cmap="magma"),
                          f"Uncertainty · {meta['labels'].get(r, r)}"))
        image_row(items, 4)
        rows = [dict(model=meta["labels"].get(r, r), **v) for r, v in s["scores"].items()]
        if rows:
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
