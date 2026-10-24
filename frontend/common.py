"""Shared loading, styling and widgets for the ClearSAR frontend."""
import base64
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data.dataset import N_S1, to_rgb  # noqa: E402

SPLIT_NAMES = {"train": "Train", "val": "Validation", "test_id": "Asia (in-domain)", "ood": "Other continents"}
SPLIT_COLORS = {"train": "#60A5FA", "val": "#34D399", "test_id": "#2DD4BF", "ood": "#A78BFA"}
METRICS = {"psnr": ("PSNR", "dB", True), "ssim": ("SSIM", "", True), "sam": ("SAM", "°", False),
           "mae": ("MAE", "", False), "fid": ("FID", "", False)}
CORE = ["A_unet", "B_swin", "C_vit"]
ACCENT, VIOLET, AMBER, ROSE, MUTED = "#2DD4BF", "#A78BFA", "#FBBF24", "#FB7185", "#8B95A7"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stSidebar"] { background: #0E1528; border-right: 1px solid #1E293B; }
.cs-eyebrow { color: #2DD4BF; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; margin-bottom: .35rem; }
.cs-hero h1 { font-size: 2.7rem; font-weight: 800; line-height: 1.08; margin: 0 0 .6rem 0; }
.cs-hero p { color: #A9B3C6; font-size: 1.06rem; max-width: 760px; line-height: 1.55; }
.cs-grad { background: linear-gradient(90deg, #2DD4BF, #A78BFA); -webkit-background-clip: text; background-clip: text; color: transparent; }
.cs-card { background: #121A2E; border: 1px solid #1F2A44; border-radius: 16px; padding: 18px 20px; height: 100%; }
.cs-card h4 { margin: 0 0 .35rem 0; font-size: 1.02rem; }
.cs-card p { color: #A9B3C6; font-size: .92rem; margin: 0; line-height: 1.5; }
.cs-kpi { background: #121A2E; border: 1px solid #1F2A44; border-radius: 16px; padding: 16px 18px; }
.cs-kpi .lab { color: #8B95A7; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }
.cs-kpi .val { font-size: 1.85rem; font-weight: 800; margin-top: .2rem; font-family: 'Inter'; }
.cs-kpi .sub { color: #A9B3C6; font-size: .85rem; margin-top: .1rem; }
.cs-pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin: 2px 4px 2px 0; border: 1px solid; }
.cs-good { color: #34D399; border-color: #14532D; background: #052E1C; }
.cs-bad  { color: #FB7185; border-color: #4C0519; background: #2A0A14; }
.cs-warn { color: #FBBF24; border-color: #713F12; background: #2A1B05; }
.cs-neutral { color: #A9B3C6; border-color: #334155; background: #111827; }
.cs-accent { color: #2DD4BF; border-color: #115E59; background: #042F2E; }
.cs-mono { font-family: 'JetBrains Mono', monospace; }
.cs-section { margin-top: 2.2rem; margin-bottom: .6rem; }
.cs-section h2 { font-size: 1.45rem; margin: 0; }
.cs-section p { color: #8B95A7; margin: .2rem 0 0 0; font-size: .95rem; }
.cs-rule { border-left: 3px solid #2DD4BF; background: #0F1729; border-radius: 10px; padding: 10px 14px; margin: 6px 0; }
.cs-rule.neg { border-left-color: #FB7185; }
.cs-rule .id { font-family: 'JetBrains Mono', monospace; color: #2DD4BF; font-size: .8rem; }
.cs-rule .txt { color: #D5DBE7; font-size: .92rem; }
.cs-cfbar { height: 6px; border-radius: 999px; background: #1E293B; overflow: hidden; margin-top: 8px; }
.cs-cfbar > div { height: 100%; background: linear-gradient(90deg, #2DD4BF, #A78BFA); }
.cs-vgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
.cs-verdict { border-radius: 16px; padding: 16px 18px; border: 1px solid #1F2A44; background: #121A2E; height: 100%; }
.cs-verdict .k { color: #8B95A7; font-size: .76rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
.cs-verdict .v { font-size: 1.35rem; font-weight: 800; margin: .25rem 0 .2rem 0; }
.cs-verdict .d { color: #A9B3C6; font-size: .86rem; line-height: 1.45; }
div[data-testid="stImage"] img { border-radius: 10px; }
div[data-testid="stCaptionContainer"] { color: #8B95A7; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------ data
def _find_root(base):
    base = base if os.path.isabs(base) else os.path.join(ROOT, base)
    if not os.path.isdir(base):
        return None
    for cand in [base] + [os.path.join(base, d) for d in sorted(os.listdir(base))]:
        if os.path.exists(os.path.join(cand, "results", "metrics_summary.csv")):
            return cand
    return None


def results_root():
    if "results_base" not in st.session_state:
        st.session_state.results_base = os.environ.get("CLEARSAR_RESULTS", "outputs")
    return _find_root(st.session_state.results_base)


def _stamp(root):
    p = os.path.join(root, "results", "metrics_summary.csv")
    return os.path.getmtime(p) if os.path.exists(p) else 0


@st.cache_data(show_spinner=False)
def _load(root, stamp):
    def csv(*parts):
        p = os.path.join(root, *parts)
        return pd.read_csv(p) if os.path.exists(p) else None

    d = {k: csv("results", f"{k}.csv") for k in
         ["metrics_summary", "metrics_main", "experiments", "learning_curves", "metrics_by_cloud_bin",
          "metrics_by_country", "uncertainty_summary", "sparsification", "erf_stats", "scene_splits", "run_info"]}
    cfg_p = os.path.join(root, "config.json")
    d["config"] = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    hist = []
    logs = os.path.join(root, "logs")
    if os.path.isdir(logs):
        for f in sorted(os.listdir(logs)):
            if f.startswith("history_") and f.endswith(".csv"):
                h = pd.read_csv(os.path.join(logs, f))
                h["run"] = f[len("history_"):-4]
                hist.append(h)
    d["history"] = pd.concat(hist) if hist else None
    sp = os.path.join(root, "results", "samples.npz")
    if os.path.exists(sp):
        z = np.load(sp, allow_pickle=False)
        d["samples"] = {k: z[k] for k in z.files if k != "meta"}
        d["samples_meta"] = json.loads(str(z["meta"]))
    else:
        d["samples"], d["samples_meta"] = None, None
    m = d["metrics_main"]
    if m is None and d["metrics_summary"] is not None:      # older single-seed runs
        m = d["metrics_summary"][d["metrics_summary"].split.isin(["test_id", "ood"])].rename(columns={"run": "group"})
    d["main"] = m
    return d


def load():
    root = results_root()
    if root is None:
        return None, None
    return root, _load(root, _stamp(root))


def labels(data):
    lab = dict(data["config"].get("labels", {}))
    if data["metrics_summary"] is not None:
        lab.update(dict(zip(data["metrics_summary"].run, data["metrics_summary"].label)))
    return lab


def sidebar_source():
    with st.sidebar:
        st.markdown("<div class='cs-eyebrow'>ClearSAR</div>", unsafe_allow_html=True)
        with st.expander("Data source", expanded=False):
            st.text_input("Results folder", key="results_base",
                          help="Folder where you extracted clearsar_output.zip from Kaggle")
            root = results_root()
            st.caption(f"Loaded: `{os.path.relpath(root, ROOT)}`" if root else "No results found")


def no_results():
    st.markdown("""
<div class='cs-hero'><div class='cs-eyebrow'>No results yet</div>
<h1>Run the Kaggle pipeline first</h1>
<p>Train the models on Kaggle (<code>notebooks/kaggle_pipeline.py</code>), download
<code>clearsar_output.zip</code> from the notebook's Output tab and extract it into <code>outputs/</code>
next to <code>app.py</code>. Then refresh this page.</p></div>""", unsafe_allow_html=True)
    st.stop()


# ------------------------------------------------------------------ widgets
def hero(eyebrow, title_html, text):
    st.markdown(f"<div class='cs-hero'><div class='cs-eyebrow'>{eyebrow}</div><h1>{title_html}</h1>"
                f"<p>{text}</p></div>", unsafe_allow_html=True)


def section(title, text=""):
    st.markdown(f"<div class='cs-section'><h2>{title}</h2>{f'<p>{text}</p>' if text else ''}</div>",
                unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='cs-kpi'><div class='lab'>{label}</div><div class='val'>{value}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def card(col, title, text):
    col.markdown(f"<div class='cs-card'><h4>{title}</h4><p>{text}</p></div>", unsafe_allow_html=True)


def pill(text, tone="neutral"):
    return f"<span class='cs-pill cs-{tone}'>{text}</span>"


def to_uint8(arr, cmap=None, vmax=None):
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 2:
        hi = vmax if vmax is not None else float(np.percentile(a, 99)) or 1.0
        a = np.clip(a / max(hi, 1e-6), 0, 1)
        if cmap:
            import matplotlib
            a = matplotlib.colormaps[cmap](a)[..., :3]
        else:
            a = np.repeat(a[..., None], 3, -1)
    return (np.clip(a, 0, 1) * 255).astype(np.uint8)


def png_b64(img_uint8):
    buf = io.BytesIO()
    Image.fromarray(img_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def compare_slider(left, right, left_label, right_label, size=520, key=None):
    """Drag-to-compare two uint8 RGB images (left image revealed on the left of the handle)."""
    a, b = png_b64(left), png_b64(right)
    html = f"""
<div id="w" style="position:relative;width:{size}px;max-width:100%;aspect-ratio:1/1;margin:0 auto;
     border-radius:14px;overflow:hidden;border:1px solid #1F2A44;font-family:Inter,sans-serif;user-select:none">
  <img src="data:image/png;base64,{b}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover">
  <img id="top" src="data:image/png;base64,{a}" style="position:absolute;inset:0;width:100%;height:100%;
       object-fit:cover;clip-path:inset(0 50% 0 0)">
  <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#E6EAF2;box-shadow:0 0 8px #000"></div>
  <div id="knob" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:34px;height:34px;
       border-radius:50%;background:#E6EAF2;color:#0B1020;display:flex;align-items:center;justify-content:center;
       font-weight:800;font-size:14px;box-shadow:0 2px 10px #0008">&#8596;</div>
  <span style="position:absolute;left:10px;top:10px;background:#0B1020cc;color:#E6EAF2;padding:4px 10px;border-radius:999px;font-size:12px">{left_label}</span>
  <span style="position:absolute;right:10px;top:10px;background:#0B1020cc;color:#2DD4BF;padding:4px 10px;border-radius:999px;font-size:12px">{right_label}</span>
  <input id="r" type="range" min="0" max="100" value="50"
         style="position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0">
</div>
<script>
const r=document.getElementById('r'),t=document.getElementById('top'),b=document.getElementById('bar'),k=document.getElementById('knob');
r.addEventListener('input',e=>{{const v=e.target.value;t.style.clipPath=`inset(0 ${{100-v}}% 0 0)`;b.style.left=v+'%';k.style.left=v+'%';}});
</script>"""
    if hasattr(st, "iframe"):            # Streamlit >= 1.5x
        st.iframe(html, height=size + 24)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=size + 24)


def verdict_grid(items):
    """items: [(key_label, value_title, color, description, cf)] -> responsive grid of decision cards."""
    cells = "".join(
        f"<div class='cs-verdict'><div class='k'>{k}</div><div class='v' style='color:{c}'>{v}</div>"
        f"<div class='d'>{d}</div><div class='cs-cfbar'><div style='width:{100 * (cf or 0):.0f}%'></div></div>"
        f"<div class='d' style='margin-top:4px'>certainty {(cf or 0):.2f}</div></div>" for k, v, c, d, cf in items)
    st.markdown(f"<div class='cs-vgrid'>{cells}</div>", unsafe_allow_html=True)


def rgb_u8(bands13):
    return to_uint8(to_rgb(np.asarray(bands13, np.float32)))


def sample_rgb(data, run, i):
    s = data["samples"]
    return s[f"pred_{run}"][i] if f"pred_{run}" in s else None


def sample_names(meta):
    out = []
    for s in meta["samples"]:
        where = "Asia" if s["split"] == "test_id" else s["continent"]
        out.append(f"{s['country']} · {where} · {int(round(s['cloud_frac'] * 100))}% cloud")
    return out


def best_main_model(main, split="test_id"):
    m = main[(main.split == split) & main.group.isin(CORE)]
    if not len(m):
        m = main[(main.split == split) & ~main.group.isin(["identity", "tiny_cnn"])]
    return m.sort_values("psnr", ascending=False).iloc[0] if len(m) else None


# ------------------------------------------------------------------ final model outputs (outputs/final/)
FINAL_RUNS = ["identity", "p1_unet", "before_gan", "final"]
FINAL_LABELS = {"identity": "Cloudy input", "p1_unet": "Phase 1 U-Net", "before_gan": "ClearSAR-Net · before GAN",
                "final": "ClearSAR-Net · final (GAN)"}
FINAL_COLORS = {"identity": MUTED, "p1_unet": VIOLET, "before_gan": AMBER, "final": ACCENT}
CLOUD_ORDER = ["0-10%", "10-30%", "30-60%", "60-90%", "90-100%"]


def final_root():
    """outputs/final (or a sub-folder of it) holding results/test_summary.csv from notebooks/train_clearsar.py."""
    base = os.path.join(ROOT, "outputs", "final")
    if not os.path.isdir(base):
        return None
    for cand in [base] + [os.path.join(base, d) for d in sorted(os.listdir(base))]:
        if os.path.exists(os.path.join(cand, "results", "test_summary.csv")):
            return cand
    return None


@st.cache_data(show_spinner=False)
def _load_final(root, stamp):
    def csv(*parts):
        p = os.path.join(root, *parts)
        return pd.read_csv(p) if os.path.exists(p) else None

    def js(*parts):
        p = os.path.join(root, *parts)
        return json.load(open(p)) if os.path.exists(p) else {}

    d = {k: csv("results", f"{k}.csv") for k in ["test_summary", "test_by_cloud", "test_by_continent",
                                                  "test_by_country", "test_per_patch", "scene_splits"]}
    d["train"], d["val"] = csv("logs", "train_history.csv"), csv("logs", "val_history.csv")
    d["summary"], d["uncertainty"], d["config"] = js("logs", "train_summary.json"), js("results", "uncertainty.json"), \
        js("config.json")
    sp = os.path.join(root, "results", "samples.npz")
    if os.path.exists(sp):
        z = np.load(sp, allow_pickle=False)
        d["samples"] = {k: z[k] for k in z.files if k != "meta"}
        d["samples_meta"] = json.loads(str(z["meta"]))
    else:
        d["samples"], d["samples_meta"] = None, None
    d["root"] = root
    return d


def load_final():
    root = final_root()
    if root is None:
        return None
    return _load_final(root, os.path.getmtime(os.path.join(root, "results", "test_summary.csv")))


def final_sample_names(meta):
    return [f"{m['country']} · {m['continent']} · {int(round(m['cloud_frac'] * 100))}% cloud" for m in meta["samples"]]


def no_final():
    st.markdown("""
<div class='cs-hero'><div class='cs-eyebrow'>No final-model results yet</div>
<h1>Run the final training first</h1>
<p>Run <code>notebooks/train_clearsar.py</code> on Kaggle (2 × T4, ~12 h), download
<code>clearsar_final_output.zip</code> from the notebook's Output tab and extract it into <code>outputs/final/</code>
next to <code>app.py</code>. Then refresh this page. The Phase 1 pages work without it.</p></div>""",
                unsafe_allow_html=True)
    st.stop()
