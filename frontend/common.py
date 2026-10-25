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
:root {
  --bg: #070B18; --panel: rgba(17, 26, 51, .72); --panel-solid: #111A33; --line: rgba(125, 211, 252, .16);
  --text: #E8EDF7; --muted: #A3AEC2; --dim: #8592A8;
  --teal: #2DD4BF; --sky: #38BDF8; --violet: #A78BFA; --amber: #FBBF24; --rose: #FB7185; --lime: #A3E635;
  --grad: linear-gradient(100deg, #2DD4BF 0%, #38BDF8 45%, #A78BFA 100%);
}
/* ---------- canvas: deep space with soft aurora glows */
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(45, 212, 191, .14), transparent 62%),
    radial-gradient(900px 520px at 96% -4%, rgba(167, 139, 250, .16), transparent 60%),
    radial-gradient(900px 700px at 55% 115%, rgba(56, 189, 248, .07), transparent 60%),
    var(--bg);
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 3.4rem; padding-bottom: 4.5rem; max-width: 1200px; }
h1, h2, h3, h4 { letter-spacing: -0.015em; }
p, li { line-height: 1.6; }
a { text-decoration-thickness: 1px; text-underline-offset: 3px; }

/* ---------- accessibility: visible keyboard focus everywhere */
*:focus-visible { outline: 2px solid #7DD3FC !important; outline-offset: 2px; border-radius: 6px; }

/* ---------- sidebar */
[data-testid="stSidebar"] { border-right: 1px solid var(--line);
  background: linear-gradient(180deg, rgba(45,212,191,.06), transparent 30%), #0A1024; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(45, 212, 191, .12); }
[data-testid="stSidebarNavSeparator"] { border-color: var(--line); }
.cs-side-foot { color: var(--dim); font-size: .78rem; line-height: 1.5; margin-top: .6rem; }
.cs-side-foot b { color: var(--muted); font-weight: 600; }

/* ---------- hero */
.cs-hero { position: relative; overflow: hidden; border-radius: 24px; padding: 30px 34px 28px 34px; margin-bottom: 1.2rem;
  background: linear-gradient(135deg, rgba(17, 26, 51, .92), rgba(9, 14, 32, .78));
  border: 1px solid var(--line); box-shadow: 0 20px 60px -30px rgba(45, 212, 191, .35); }
.cs-hero::before { content: ""; position: absolute; inset: 0; pointer-events: none;
  background: radial-gradient(500px 220px at 0% 0%, rgba(45, 212, 191, .18), transparent 70%),
              radial-gradient(420px 220px at 100% 100%, rgba(167, 139, 250, .16), transparent 70%); }
.cs-hero .cs-hero-img { position: absolute; top: 0; right: 0; bottom: 0; width: 46%; background-size: cover; background-position: center;
  -webkit-mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.55) 35%, #000 100%);
  mask-image: linear-gradient(90deg, transparent 0%, rgba(0,0,0,.55) 35%, #000 100%); opacity: .55; }
.cs-hero .cs-hero-body { position: relative; max-width: 700px; }
.cs-hero h1 { font-size: 2.55rem; font-weight: 700; line-height: 1.1; margin: .55rem 0 .65rem 0; padding: 0; }
.cs-hero p { color: var(--muted); font-size: 1.04rem; margin: 0; }
.cs-eyebrow { display: inline-flex; align-items: center; gap: .5rem; color: #99F6E4; font-size: .74rem; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase; padding: 4px 11px; border-radius: 999px;
  background: rgba(45, 212, 191, .1); border: 1px solid rgba(45, 212, 191, .28); }
.cs-eyebrow::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--teal); box-shadow: 0 0 10px var(--teal); }
.cs-grad { background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }

/* ---------- sections */
.cs-section { margin: 2.4rem 0 .8rem 0; }
.cs-section h2 { font-size: 1.5rem; margin: 0; padding: 0; display: flex; align-items: center; gap: .6rem; }
.cs-section h2::before { content: ""; width: 4px; height: 1.2em; border-radius: 4px; background: var(--grad); }
.cs-section p { color: var(--dim); margin: .35rem 0 0 0; font-size: .95rem; max-width: 820px; }

/* ---------- glass panels: cards, KPIs, verdicts */
.cs-card, .cs-kpi, .cs-verdict { position: relative; border-radius: 18px; height: 100%;
  background: linear-gradient(var(--panel-solid), var(--panel-solid)) padding-box,
              linear-gradient(140deg, rgba(45,212,191,.45), rgba(56,189,248,.12) 40%, rgba(167,139,250,.4)) border-box;
  border: 1px solid transparent; box-shadow: 0 12px 30px -18px rgba(0, 0, 0, .8); }
.cs-card { padding: 20px 22px; transition: transform .25s ease, box-shadow .25s ease; }
.cs-card:hover { transform: translateY(-2px); box-shadow: 0 18px 40px -20px rgba(45, 212, 191, .35); }
.cs-card h4 { margin: 0 0 .45rem 0; font-size: 1.05rem; padding: 0; }
.cs-card p { color: var(--muted); font-size: .92rem; margin: 0; line-height: 1.55; }
.cs-kpi { padding: 16px 18px 15px 18px; min-height: 132px; display: flex; flex-direction: column; }
.cs-kpi .lab { color: var(--dim); font-size: .74rem; font-weight: 700; text-transform: uppercase; letter-spacing: .1em; }
.cs-kpi .val { font-size: 2rem; font-weight: 700; margin-top: .25rem; line-height: 1.15;
  background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent; }
.cs-kpi .sub { color: var(--muted); font-size: .84rem; margin-top: auto; padding-top: .35rem; }
.cs-verdict { padding: 16px 18px; }
.cs-verdict .k { color: var(--dim); font-size: .74rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
.cs-verdict .v { font-size: 1.35rem; font-weight: 700; margin: .25rem 0 .2rem 0; }
.cs-verdict .d { color: var(--muted); font-size: .86rem; line-height: 1.45; }
.cs-cgrid { display: grid; gap: 14px; margin: .2rem 0 .6rem 0; }
.cs-vgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; }
/* equal heights for cards placed in st.columns */
[data-testid="stHorizontalBlock"] { align-items: stretch; }
[data-testid="stColumn"]:has(.cs-card, .cs-kpi) > div,
[data-testid="stColumn"]:has(.cs-card, .cs-kpi) [data-testid="stVerticalBlock"] { height: 100%; }
[data-testid="stElementContainer"]:has(> div > [data-testid="stMarkdown"] .cs-card),
[data-testid="stElementContainer"]:has(> div > [data-testid="stMarkdown"] .cs-kpi) { flex: 1 1 auto; }
[data-testid="stMarkdown"]:has(.cs-card, .cs-kpi), [data-testid="stMarkdown"]:has(.cs-card, .cs-kpi) > div,
[data-testid="stMarkdownContainer"]:has(.cs-card, .cs-kpi) { height: 100%; }

/* ---------- pills, rules, bars */
.cs-pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; margin: 2px 4px 2px 0; border: 1px solid; }
.cs-good { color: #6EE7B7; border-color: rgba(52, 211, 153, .35); background: rgba(52, 211, 153, .1); }
.cs-bad  { color: #FDA4AF; border-color: rgba(251, 113, 133, .35); background: rgba(251, 113, 133, .1); }
.cs-warn { color: #FCD34D; border-color: rgba(251, 191, 36, .35); background: rgba(251, 191, 36, .1); }
.cs-neutral { color: var(--muted); border-color: rgba(148, 163, 184, .3); background: rgba(148, 163, 184, .08); }
.cs-accent { color: #5EEAD4; border-color: rgba(45, 212, 191, .4); background: rgba(45, 212, 191, .1); }
.cs-mono { font-family: 'JetBrains Mono', monospace; }
.cs-rule { border-left: 3px solid var(--teal); background: rgba(17, 26, 51, .8); border-radius: 12px; padding: 10px 14px; margin: 6px 0; }
.cs-rule.neg { border-left-color: var(--rose); }
.cs-rule .id { font-family: 'JetBrains Mono', monospace; color: var(--teal); font-size: .8rem; }
.cs-rule .txt { color: #DCE3EF; font-size: .92rem; }
.cs-cfbar { height: 6px; border-radius: 999px; background: rgba(148, 163, 184, .18); overflow: hidden; margin-top: 8px; }
.cs-cfbar > div { height: 100%; background: var(--grad); }

/* ---------- media, tabs, expanders */
div[data-testid="stImage"] img { border-radius: 12px; border: 1px solid var(--line); }
div[data-testid="stCaptionContainer"], [data-testid="stImageCaption"] { color: var(--dim); }
[data-baseweb="tab-list"] { gap: .25rem; }
[data-baseweb="tab"][aria-selected="true"] p { color: #99F6E4; }
[data-testid="stExpander"] details { border-color: var(--line) !important; background: rgba(17, 26, 51, .45); border-radius: 14px; }
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] { background: rgba(17, 26, 51, .35); border-radius: 16px; padding: 10px 8px 4px 8px; border: 1px solid var(--line); }

/* ---------- motion only for people who want it */
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { transition: none !important; animation: none !important; } }
@media (max-width: 900px) { .cs-hero .cs-hero-img { display: none; } .cs-hero h1 { font-size: 2rem; } }
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
    st.logo(os.path.join(ROOT, "frontend", "assets", "logo.svg"), size="large")
    with st.sidebar:
        fin = os.path.exists(os.path.join(ROOT, "outputs", "final", "results", "test_summary.csv"))
        st.markdown("<div class='cs-side-foot'><b>SAR-to-optical cloud removal</b><br>Sentinel-1 radar + cloudy "
                    "Sentinel-2 &rarr; cloud-free Sentinel-2" + ("<br>Final model results loaded" if fin else "")
                    + "</div>", unsafe_allow_html=True)
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
@st.cache_data(show_spinner=False)
def _hero_strip(stamp):
    """Decorative banner for page headers: cloudy input and ClearSAR-Net output of two test patches."""
    sp = os.path.join(ROOT, "outputs", "final", "results", "samples.npz")
    if not os.path.exists(sp):
        return None
    z = np.load(sp, allow_pickle=False)
    meta = json.loads(str(z["meta"]))["samples"]
    pick = next((i for i, m in enumerate(meta) if m["country"] == "IN"),
                next((i for i, m in enumerate(meta) if 0.2 <= m["cloud_frac"] <= 0.6), 0))
    tiles = [rgb_u8(z["x"][pick][N_S1:].astype(np.float32)), z["pred_final"][pick]]
    img = Image.fromarray(np.concatenate(tiles, axis=1)).resize((256 * len(tiles), 256))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return base64.b64encode(buf.getvalue()).decode()


def hero(eyebrow, title_html, text):
    sp = os.path.join(ROOT, "outputs", "final", "results", "samples.npz")
    strip = _hero_strip(os.path.getmtime(sp) if os.path.exists(sp) else 0)
    img = (f"<div class='cs-hero-img' role='presentation' style=\"background-image:url(data:image/jpeg;base64,{strip})\"></div>"
           if strip else "")
    st.markdown(f"<div class='cs-hero'>{img}<div class='cs-hero-body'><span class='cs-eyebrow'>{eyebrow}</span>"
                f"<h1>{title_html}</h1><p>{text}</p></div></div>", unsafe_allow_html=True)


def section(title, text=""):
    st.markdown(f"<div class='cs-section'><h2>{title}</h2>{f'<p>{text}</p>' if text else ''}</div>",
                unsafe_allow_html=True)


def kpi(col, label, value, sub=""):
    col.markdown(f"<div class='cs-kpi'><div class='lab'>{label}</div><div class='val'>{value}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def card(col, title, text):
    col.markdown(f"<div class='cs-card'><h4>{title}</h4><p>{text}</p></div>", unsafe_allow_html=True)


def card_row(items, min_width=260):
    """A row of equally tall cards: items = [(title, text), ...]."""
    cells = "".join(f"<div class='cs-card'><h4>{t}</h4><p>{x}</p></div>" for t, x in items)
    st.markdown(f"<div class='cs-cgrid' style='grid-template-columns:repeat(auto-fit,minmax({min_width}px,1fr))'>"
                f"{cells}</div>", unsafe_allow_html=True)


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
     border-radius:16px;overflow:hidden;border:1px solid rgba(125,211,252,.25);box-shadow:0 20px 50px -25px rgba(45,212,191,.45);font-family:Inter,sans-serif;user-select:none">
  <img src="data:image/png;base64,{b}" alt="{right_label}" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover">
  <img id="top" src="data:image/png;base64,{a}" alt="{left_label}" style="position:absolute;inset:0;width:100%;height:100%;
       object-fit:cover;clip-path:inset(0 50% 0 0)">
  <div id="bar" style="position:absolute;top:0;bottom:0;left:50%;width:2px;background:#E6EAF2;box-shadow:0 0 8px #000"></div>
  <div id="knob" style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:34px;height:34px;
       border-radius:50%;background:#E6EAF2;color:#0B1020;display:flex;align-items:center;justify-content:center;
       font-weight:800;font-size:14px;box-shadow:0 2px 10px #0008">&#8596;</div>
  <span style="position:absolute;left:10px;top:10px;background:#0B1020cc;color:#E6EAF2;padding:4px 10px;border-radius:999px;font-size:12px">{left_label}</span>
  <span style="position:absolute;right:10px;top:10px;background:#0B1020cc;color:#5EEAD4;padding:4px 10px;border-radius:999px;font-size:12px">{right_label}</span>
  <input id="r" type="range" min="0" max="100" value="50" step="1"
         aria-label="Comparison slider: {left_label} on the left, {right_label} on the right. Use the arrow keys to move it."
         style="position:absolute;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0">
</div>
<script>
const r=document.getElementById('r'),t=document.getElementById('top'),b=document.getElementById('bar'),k=document.getElementById('knob');
r.addEventListener('input',e=>{{const v=e.target.value;t.style.clipPath=`inset(0 ${{100-v}}% 0 0)`;b.style.left=v+'%';k.style.left=v+'%';}});
r.addEventListener('focus',()=>{{k.style.boxShadow='0 0 0 3px #7DD3FC';}});
r.addEventListener('blur',()=>{{k.style.boxShadow='0 2px 10px #0008';}});
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
