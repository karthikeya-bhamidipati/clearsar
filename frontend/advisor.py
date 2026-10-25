import json
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from frontend.common import (FINAL_LABELS, compare_slider, final_sample_names, hero, labels, load, load_final,
                             no_results, pill, rgb_u8, sample_names, section, to_uint8, verdict_grid)

root, data = load()
fin = load_final()
hero("Expert system · explainable decision support",
     "Should you <span class='cs-grad'>trust this reconstruction?</span>",
     "The deep-learning models fill in what the clouds hide. The advisor measures the evidence - cloud cover, how "
     "much the model leaned on radar, whether the architectures agree, predicted uncertainty, what the experiments "
     "showed - and reasons over a knowledge base of rules to recommend what to do, explaining every step.")
if data is None:
    no_results()

from src.expert.advisor import Advisor, region_from_lonlat  # noqa: E402
from src.expert.engine import KnowledgeBase  # noqa: E402

lab = {**labels(data), **FINAL_LABELS}
kb = KnowledgeBase.load()
USE_CASES = kb.labels["use_case"]
REGION_NAMES = {"in_region": "Inside the training region (Asia)", "out_of_region": "Outside Asia",
                "unknown": "Unknown"}


@st.cache_resource(show_spinner=False)
def get_advisor(root_dir, final_dir, stamp):
    return Advisor(os.path.join(root_dir, "checkpoints"), os.path.join(root_dir, "results"), lab, final_dir=final_dir)


final_dir = fin["root"] if fin is not None else None
advisor = get_advisor(root, final_dir, (os.path.getmtime(os.path.join(root, "results", "metrics_summary.csv")),
                                        os.path.getmtime(os.path.join(final_dir, "results", "test_summary.csv"))
                                        if final_dir else 0))
if advisor.final_mode:
    st.caption("Reconstruction model: **ClearSAR-Net** (final, and before its GAN part). The three Phase 1 "
               "architectures run alongside as independent second opinions - their disagreement is one of the facts.")
if not advisor.available:
    st.warning("No model checkpoints in this run folder - the advisor needs `checkpoints/best_*.pt`.")
    st.stop()

# ---------------------------------------------------------------- input
c1, c2, c3 = st.columns([1.2, 1.1, 1])
source = c1.segmented_control("Patch", ["Test gallery", "Upload GeoTIFFs"], default="Test gallery", key="adv_src") \
    or "Test gallery"
use_case = c2.selectbox("Intended use", list(USE_CASES), format_func=USE_CASES.get, key="adv_use")
x = key = None
auto_region, region_note = "unknown", ""
if source == "Test gallery":
    use_final = advisor.final_mode and fin["samples"] is not None
    s, meta = (fin["samples"], fin["samples_meta"]) if use_final else (data["samples"], data["samples_meta"])
    if s is None:
        st.info("No gallery samples in this run - upload GeoTIFFs instead.")
        st.stop()
    names = final_sample_names(meta) if use_final else sample_names(meta)
    i = c3.selectbox("Test patch", range(len(names)), format_func=lambda k: names[k], key="adv_patch")
    x = s["x"][i].astype(np.float32)
    info = meta["samples"][i]
    auto_region = "in_region" if info["continent"] == "Asia" else "out_of_region"
    region_note = f"from the scene metadata ({info['country']})"
    key = f"g{i}"
else:
    u1, u2 = st.columns(2)
    f1 = u1.file_uploader("Sentinel-1 patch (2 bands, dB)", type=["tif", "tiff"], key="adv_s1")
    f2 = u2.file_uploader("Cloudy Sentinel-2 patch (13 bands)", type=["tif", "tiff"], key="adv_s2")
    if f1 and f2:
        from rasterio.io import MemoryFile
        from rasterio.warp import transform

        from src.data.dataset import preprocess_s1, preprocess_s2
        try:
            with MemoryFile(f1.getvalue()) as mf, mf.open() as ds:
                s1 = ds.read()[:2]
            with MemoryFile(f2.getvalue()) as mf, mf.open() as ds:
                s2c = ds.read()[:13]
                cx, cy = (ds.bounds.left + ds.bounds.right) / 2, (ds.bounds.bottom + ds.bounds.top) / 2
                lon, lat = transform(ds.crs, "EPSG:4326", [cx], [cy]) if ds.crs else ([None], [None])
            if s1.shape[0] < 2 or s2c.shape[0] < 13 or s1.shape[1:] != s2c.shape[1:]:
                raise ValueError(f"need 2 S1 + 13 S2 bands of equal size, got {s1.shape} and {s2c.shape}")
            x = np.concatenate([preprocess_s1(s1), preprocess_s2(s2c)], axis=0)
            h, w = (x.shape[1] // 128) * 128, (x.shape[2] // 128) * 128
            if h == 0 or w == 0:
                raise ValueError("patches must be at least 128 × 128 pixels")
            x = x[:, :h, :w]
            if lon[0] is not None:
                auto_region, cc = region_from_lonlat(lon[0], lat[0])
                region_note = f"from the GeoTIFF location ({cc or '?'}, {lat[0]:.2f}°, {lon[0]:.2f}°)"
            key = f"u{f1.name}{f2.name}{f1.size}{f2.size}"
        except Exception as e:
            st.error(f"Could not read the GeoTIFFs: {e}")
    else:
        st.caption("SEN12MS-CR style patches. The clear image is not needed - the advisor never sees the answer.")

if advisor.final_mode:                 # trained on every continent: no region is outside the training data
    region = "in_region"
    REGION_NAMES["in_region"] = "Inside the training data (the final model was trained on every continent)"
else:
    region = st.radio("Region", list(REGION_NAMES), index=list(REGION_NAMES).index(auto_region), horizontal=True,
                      format_func=REGION_NAMES.get, key=f"adv_region_{key}",
                      help=f"Detected automatically {region_note}" if region_note else None)
if x is None:
    st.stop()

@st.cache_resource(show_spinner=False)
def advice_store(_advisor_id):
    """Results already computed in this server session, shared by all visitors (gallery patches only)."""
    return {}


@st.cache_resource(show_spinner=False)
def warm_up(_advisor, _x_all, n, default_use, region_default):
    """Analyse every gallery patch for the default use in a background thread, once per server, so that
    switching patches is instant during a demo (the first analysis takes ~30-60 s on a laptop CPU)."""
    import threading

    store = advice_store(id(_advisor))

    def run():
        for i in range(n):
            k = f"g{i}|{default_use}|{region_default}"
            if k not in store:
                try:
                    store[k] = _advisor.advise(_x_all[i].astype(np.float32), use_case=default_use, region=region_default)
                except Exception:          # never let the warm-up break the page
                    pass
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


run_key = f"{key}|{use_case}|{region}"
store = advice_store(id(advisor))
if source == "Test gallery":
    warm_up(advisor, s["x"], len(names), list(USE_CASES)[0], region)
if run_key not in store or not run_key.startswith("g"):
    with st.spinner("Running the deep-learning models, explaining them and reasoning over the knowledge base "
                    "(first time for this patch and use only)..."):
        store[run_key] = advisor.advise(x, use_case=use_case, region=region)
adv = store[run_key]
res = adv.result

# ---------------------------------------------------------------- decision
section("Decision")
TONE = {"high": "#34D399", "medium": "#FBBF24", "low": "#FB7185", "suitable": "#34D399", "caution": "#FBBF24",
        "unsuitable": "#FB7185", "accept": "#34D399", "accept_with_mask": "#FBBF24", "sar_only": "#A78BFA",
        "reacquire": "#FB7185"}
items = []
for fact, title in zip(["trust", "recommended_model", "suitability", "action"],
                       ["Trust", "Recommended model", "Fit for purpose", "Recommended action"]):
    v, cf = res.value(fact)
    if fact == "recommended_model":
        items.append((title, lab.get(v, v), "#2DD4BF", "Best match for this cloud level and use in the test results.", cf))
    else:
        meta_l = kb.labels.get(fact, {}).get(v, {})
        items.append((title, meta_l.get("title", v), TONE.get(v, "#E6EAF2"), meta_l.get("text", ""), cf))
verdict_grid(items)

warn_text = {"domain_shift": "Outside the training region", "sar_gaps": "Radar has gaps", "cirrus": "Possible thin cirrus",
             "water": "Open water present", "avoid_gan": "Avoid the GAN model for this use"}
w = res.warnings()
if w:
    st.markdown("<div style='margin-top:.8rem'>" + "".join(pill("⚠ " + warn_text.get(k, k), "warn") for k in w) + "</div>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------- explanation tabs
t_vis, t_why, t_facts, t_rep = st.tabs(["Visual explanation", "Reasoning (how & why not)", "Facts", "Report"])

with t_vis:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        compare_slider(rgb_u8(x[2:]), rgb_u8(adv.reconstruction), "Cloudy input",
                       f"{lab.get(adv.reconstruction_run, adv.reconstruction_run)}", size=470)
    with right:
        preds = [adv.preds[r] for r in ("A_unet", "B_swin", "C_vit") if r in adv.preds]
        dis = np.mean([np.abs(a - b).mean(0) for k, a in enumerate(preds) for b in preds[k + 1:]], axis=0) \
            if len(preds) > 1 else np.zeros(x.shape[1:])
        g = st.columns(2)
        g[0].image(to_uint8(adv.cloud_mask.astype(np.float32), vmax=1.0), caption=f"Detected cloud · {100 * adv.numeric['cloud_cover']:.0f}%")
        g[1].image(to_uint8(adv.occlusion["reliance"], cmap="cividis", vmax=1.0),
                   caption=f"Radar reliance · {100 * adv.numeric['sar_reliance']:.0f}% under cloud")
        g[0].image(to_uint8(dis, cmap="inferno", vmax=0.1), caption="Disagreement between the Phase 1 architectures")
        if adv.sigma is not None:
            g[1].image(to_uint8(adv.sigma, cmap="magma"),
                       caption="Uncertainty (flip test, scaled to expected error)" if advisor.final_mode else "Predicted uncertainty")
    st.caption("Radar reliance: how much the output changes when the radar is removed, relative to removing the "
               "optical input (bright = the model is using the radar there). Under thick cloud it should be bright.")
    from src.expert.xai import CHANNEL_NAMES
    sal = pd.DataFrame(dict(channel=CHANNEL_NAMES, importance=adv.saliency,
                            source=["Radar"] * 2 + ["Cloudy optical"] * 13))
    st.altair_chart(alt.Chart(sal).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("channel:N", sort=CHANNEL_NAMES, title=None, axis=alt.Axis(labelAngle=-40)),
        y=alt.Y("importance:Q", title="share of influence", axis=alt.Axis(format="%")),
        color=alt.Color("source:N", title=None, scale=alt.Scale(range=["#A78BFA", "#2DD4BF"]),
                        legend=alt.Legend(orient="top")),
        tooltip=["channel", alt.Tooltip("importance:Q", format=".1%")]).properties(
        height=230, title="Which input bands drove the reconstruction under the clouds (|input × gradient|)"),
        width="stretch")

with t_why:
    for fact, title in [("trust", "Trust"), ("recommended_model", "Recommended model"),
                        ("suitability", "Fit for purpose"), ("action", "Recommended action")]:
        v, cf = res.value(fact)
        shown = lab.get(v, v) if fact == "recommended_model" else kb.labels.get(fact, {}).get(v, {}).get("title", v)
        st.markdown(f"#### {title}: {shown} <span style='color:#8B95A7;font-size:.9rem'>· certainty {cf:.2f}</span>",
                    unsafe_allow_html=True)
        alts = res.alternatives(fact)
        if len(alts) > 1:
            st.markdown("Considered: " + " ".join(
                pill(f"{(lab.get(a, a) if fact == 'recommended_model' else a)} {c:.2f}", "accent" if a == v else "neutral")
                for a, c in alts), unsafe_allow_html=True)
        for r in res.support(fact, v):
            prem = " AND ".join(f"{p[0]} = {p[2]}" for p in r.premises)
            st.markdown(f"<div class='cs-rule'><span class='id'>{r.id} · +{r.cf:.2f}</span> "
                        f"<span class='cs-mono' style='color:#8B95A7;font-size:.78rem'>IF {prem}</span>"
                        f"<div class='txt'>{r.explanation}</div></div>", unsafe_allow_html=True)
        for a, _ in alts:
            if a == v:
                continue
            for r in res.support(fact, a):
                st.markdown(f"<div class='cs-rule neg'><span class='id' style='color:#FB7185'>{r.id} · "
                            f"supports {a} (+{r.cf:.2f})</span><div class='txt'>{r.explanation}</div></div>",
                            unsafe_allow_html=True)
        group = next((rr["group"] for rr in kb.rules if rr["then"]["fact"] == fact), None)
        nf = [n for n in res.not_fired if n[1] == group and n[2][0] == fact]
        if nf:
            with st.expander(f"Why not? - {len(nf)} rule(s) for {title.lower()} did not fire"):
                for rid, _, (f_, val), reason in nf:
                    st.markdown(f"- **{rid}** would conclude *{val}*, but {reason}.")
    if w:
        st.markdown("#### Warnings")
        for r in [r for r in res.fired if r.conclusion[0].startswith("warning_")]:
            st.markdown(f"<div class='cs-rule' style='border-left-color:#FBBF24'><span class='id' style='color:#FBBF24'>{r.id}</span>"
                        f"<div class='txt'>{r.explanation}</div></div>", unsafe_allow_html=True)

with t_facts:
    st.caption("Numeric facts are turned into symbolic levels by the thresholds in the knowledge base. "
               "Values close to a threshold are 'borderline' and enter the reasoning with certainty 0.7 instead of 1.0.")
    rows = [dict(fact=k, value=v[0], level=v[1], certainty=v[2], borderline="yes" if v[3] else "")
            for k, v in res.fuzzified.items()]
    rows += [dict(fact=k, value=None, level=str(val), certainty=1.0, borderline="") for k, val in adv.symbolic.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"value": st.column_config.NumberColumn(format="%.4f"),
                                "certainty": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

with t_rep:
    lines = [f"# ClearSAR Advisor report", "", f"- Intended use: {USE_CASES[use_case]}",
             f"- Region: {REGION_NAMES[region]}", ""]
    for fact in ["trust", "recommended_model", "suitability", "action"]:
        v, cf = res.value(fact)
        lines.append(f"## {fact.replace('_', ' ').title()}: {lab.get(v, v)} (certainty {cf:.2f})")
        lines += [f"- [{r.id}] {r.explanation}" for r in res.support(fact, v)] + [""]
    if w:
        lines.append("## Warnings")
        lines += [f"- [{r.id}] {r.explanation}" for r in res.fired if r.conclusion[0].startswith("warning_")] + [""]
    lines.append("## Facts")
    lines += [f"- {k}: {v[0]:.4f} -> {v[1]}{' (borderline)' if v[3] else ''}" for k, v in res.fuzzified.items()]
    report = "\n".join(lines)
    st.markdown(report)
    st.download_button("Download report (.md)", report, "clearsar_advice.md", "text/markdown")
    st.download_button("Download trace (.json)", json.dumps(dict(
        decision={f: res.value(f) for f in ["trust", "recommended_model", "suitability", "action"]},
        warnings=w, fired=[dict(id=r.id, conclusion=r.conclusion, cf=r.cf, explanation=r.explanation) for r in res.fired],
        not_fired=[dict(id=n[0], conclusion=n[2], reason=n[3]) for n in res.not_fired],
        facts={k: dict(value=v[0], level=v[1], cf=v[2]) for k, v in res.fuzzified.items()}), indent=2, default=str),
        "clearsar_trace.json", "application/json")
