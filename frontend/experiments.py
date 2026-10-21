import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from frontend.common import SPLIT_NAMES, hero, labels, load, no_results, pill, section

root, data = load()
hero("Experiments", "One change at a time, <span class='cs-grad'>one question each</span>",
     "Every experiment changes a single thing relative to a reference run and is trained under the identical "
     "protocol. The prediction was written down before training (docs/project_plan.md §5); the verdict is "
     "computed from the results. Differences inside the noise floor are reported as such, not as findings.")
if data is None:
    no_results()

lab = labels(data)
ex, main = data["experiments"], data["main"]
NOISE = {"psnr": 0.3, "ssim": 0.005, "sam": 0.3, "fid": 5.0}
UNITS = {"psnr": " dB", "ssim": "", "sam": "°", "fid": ""}
SPEC = {
    "baseline": ("Is a big network needed at all?", "A 3-layer CNN scores clearly below the U-Net.", "psnr", "<"),
    "E1": ("How much does each architecture depend on data?", "Less data lowers every model; the ViT loses most.", "psnr", "<"),
    "leakage": ("What does a careless split do?", "Splitting patches at random (instead of by scene) inflates the score.", "psnr", ">"),
    "E3a": ("Why does a U-Net need skip connections?", "Without skips, fine detail is lost: lower SSIM.", "ssim", "<"),
    "E3b": ("What does the loss function optimise?", "L2 averages over plausible answers, so images get blurrier than with L1: lower SSIM.", "ssim", "<"),
    "E4": ("Does an adversarial loss help?", "Sharper, more realistic images (lower FID) - but lower PSNR (perception-distortion trade-off).", "psnr", "<"),
    "E5a": ("Does shifting the attention windows matter?", "Without shifts, information cannot cross window borders: lower PSNR.", "psnr", "<"),
    "E5b": ("Does more attention context help?", "Open question - larger windows may help under large clouds.", "psnr", None),
    "E6a": ("What are positional embeddings for?", "Without them attention ignores token order and spatial layout is lost: lower PSNR.", "psnr", "<"),
    "E6b": ("How fine must the ViT's patches be?", "16×16 patches cause blocky artefacts: lower SSIM.", "ssim", "<"),
    "E7a": ("Does the variance head cost accuracy?", "Predicting uncertainty as well should not hurt the mean prediction much.", "psnr", None),
    "E7b": ("Does MC dropout cost accuracy?", "Dropout slightly regularises; little change expected.", "psnr", None),
    "E7c": ("Does an ensemble beat a single model?", "Averaging three seeds improves PSNR.", "psnr", ">"),
}
TOPICS = {"E1 · Data": ["E1"], "Leakage": ["leakage"], "E3 · U-Net": ["E3a", "E3b"], "E4 · GAN": ["E4"],
          "E5 · Swin": ["E5a", "E5b"], "E6 · ViT": ["E6a", "E6b"], "E7 · Uncertainty": ["E7a", "E7b", "E7c"],
          "Baselines & seeds": ["baseline"]}


def noise_floor(metric, ref):
    base = NOISE[metric]
    if main is not None and f"{metric}_std" in main:
        s = main[(main.group == ref) & (main.split == "test_id")][f"{metric}_std"]
        if len(s) and pd.notna(s.iloc[0]):
            base = max(base, 2 * float(s.iloc[0]))
    return base


def verdict(delta, expect, metric, ref):
    if delta is None or np.isnan(delta):
        return "No data", "neutral"
    if expect is None:
        return "Open question", "accent"
    if abs(delta) < noise_floor(metric, ref):
        return "Within noise", "warn"
    return ("Supported", "good") if (delta < 0) == (expect == "<") else ("Contradicted", "bad")


def fmt_delta(v, metric):
    return "—" if v is None or np.isnan(v) else f"{v:+.{4 if metric == 'ssim' else 2}f}{UNITS[metric]}"


def exp_card(rows, code):
    q, pred, metric, expect = SPEC[code]
    for run, g in rows.groupby("run", sort=False):
        r = g.set_index("split")
        primary = "test_id" if "test_id" in r.index else r.index[0]
        d_main = r.loc[primary, f"delta_{metric}"]
        v, tone = verdict(d_main, expect, metric, g.reference.iloc[0])
        chips = []
        for split in r.index:
            for m in ["psnr", "ssim", "fid"]:
                val = r.loc[split, f"delta_{m}"]
                if pd.notna(val):
                    good = (val > 0) if m in ("psnr", "ssim") else (val < 0)
                    chips.append(pill(f"Δ{m.upper()} {fmt_delta(val, m)} · {SPLIT_NAMES.get(split, 'random-split test')}",
                                      "good" if good else "bad"))
        st.markdown(f"""
<div class='cs-card' style='margin-bottom:12px'>
  <div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap'>
    <h4>{g.label.iloc[0]} <span style='color:#8B95A7;font-weight:500'>vs {g.reference_label.iloc[0]}</span></h4>
    <div>{pill(v, tone)}</div>
  </div>
  <p><b style='color:#E6EAF2'>{q}</b></p>
  <p style='margin-top:.3rem'>Prediction: {pred}</p>
  <div style='margin-top:.6rem'>{''.join(chips)}</div>
</div>""", unsafe_allow_html=True)


topic = st.pills("Experiment", list(TOPICS), default="E1 · Data", key="exp_topic") or "E1 · Data"
codes = TOPICS[topic]

if ex is None or ex.empty:
    st.info("No experiment results in this run folder.")
    st.stop()

if topic == "E1 · Data":
    lc = data["learning_curves"]
    if lc is not None and (lc.experiment == "E1").any():
        split = st.segmented_control("Test set", ["test_id", "ood"], default="test_id",
                                     format_func=SPLIT_NAMES.get, key="lc_split") or "test_id"
        d = lc[lc.split == split].assign(percent=lambda t: (t.fraction * 100).round())
        st.altair_chart(alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=80), strokeWidth=3).encode(
            x=alt.X("percent:Q", title="% of training patches", scale=alt.Scale(domain=[20, 105]),
                    axis=alt.Axis(values=[25, 50, 100])),
            y=alt.Y("psnr:Q", title="PSNR (dB)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=["model", "percent", alt.Tooltip("psnr:Q", format=".2f"), alt.Tooltip("ssim:Q", format=".4f")],
        ).properties(height=340), width="stretch")
        drops = []
        for mname, g in d.groupby("model"):
            g = g.set_index("percent")
            if 100 in g.index and 25 in g.index:
                drops.append((mname, float(g.loc[100, "psnr"] - g.loc[25, "psnr"])))
        if drops:
            worst = max(drops, key=lambda t: t[1])
            best25 = d[d.percent == 25].sort_values("psnr", ascending=False)
            cols = st.columns(len(drops))
            for col, (mname, dr) in zip(cols, drops):
                col.markdown(f"<div class='cs-kpi'><div class='lab'>{mname}</div><div class='val'>{dr:+.2f} dB</div>"
                             f"<div class='sub'>gain from 25% → 100% data</div></div>", unsafe_allow_html=True)
            st.markdown(
                f"{pill('Most data-hungry: ' + worst[0], 'accent')} "
                + (pill('Best with 25% of the data: ' + best25.iloc[0].model, 'neutral') if len(best25) else ""),
                unsafe_allow_html=True)
            st.caption("Prediction: the U-Net is best with little data (strong inductive bias); the ViT gains most "
                       "from more data. Compare the gains above against the seed noise (~0.3 dB).")
    section("Each reduced-data run vs its full-data reference")
    exp_card(ex[ex.experiment == "E1"], "E1")

elif topic == "Leakage":
    lk = ex[ex.experiment == "leakage"]
    if len(lk):
        r = lk.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='cs-kpi'><div class='lab'>Honest: split by scene</div><div class='val'>{r.ref_psnr:.2f} dB</div>"
                    f"<div class='sub'>test scenes never seen in training</div></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='cs-kpi'><div class='lab'>Careless: random patches</div><div class='val'>{r.psnr:.2f} dB</div>"
                    f"<div class='sub'>neighbouring patches in train and test</div></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='cs-kpi'><div class='lab'>Inflation</div><div class='val'>{r.delta_psnr:+.2f} dB</div>"
                    f"<div class='sub'>{verdict(r.delta_psnr, '>', 'psnr', 'A_unet')[0]}</div></div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        exp_card(lk, "leakage")

elif topic == "E7 · Uncertainty":
    u = data["uncertainty_summary"]
    if u is not None:
        section("Can the models tell when they are wrong?",
                "coverage95: share of true values inside the predicted 95% interval (ideal 0.95) · "
                "AUSE: how well uncertainty ranks errors (0 = perfect) · σ rising on other continents = the method notices the domain shift.")
        cols = st.columns(len(u.method.unique()))
        for col, (method, g) in zip(cols, u.groupby("method", sort=False)):
            g = g.set_index("split")
            sig = "median_sigma" if "median_sigma" in g else "mean_sigma"
            cov = g.loc["test_id", "coverage95"] if "test_id" in g.index else np.nan
            rise = (g.loc["ood", sig] / g.loc["test_id", sig] - 1) if {"ood", "test_id"} <= set(g.index) else np.nan
            calib = "good" if abs(cov - .95) < .05 else ("warn" if abs(cov - .95) < .2 else "bad")
            col.markdown(f"""<div class='cs-card'><h4>{method}</h4>
<p>coverage95 {pill(f'{cov:.2f}', calib)}</p>
<p>AUSE {pill(f"{g.loc['test_id', 'ause']:.4f}" if 'test_id' in g.index else '—', 'neutral')}</p>
<p>σ on other continents {pill(f'{rise:+.0%}', 'good' if rise > 0.05 else ('bad' if rise < -0.05 else 'neutral'))}</p></div>""", unsafe_allow_html=True)
        sp = data["sparsification"]
        if sp is not None:
            split = st.segmented_control("Test set", ["test_id", "ood"], default="test_id",
                                         format_func=SPLIT_NAMES.get, key="sp_split") or "test_id"
            s = sp[sp.split == split]
            if "method" not in s:
                s = s.assign(method="Variance head (aleatoric)")
            s = s.melt(id_vars=["method", "fraction"], value_vars=["model_mae", "oracle_mae"], var_name="curve",
                       value_name="mae")
            s["curve"] = s.curve.map({"model_mae": "by predicted uncertainty", "oracle_mae": "oracle (by true error)"})
            st.altair_chart(alt.Chart(s).mark_line(strokeWidth=2.5).encode(
                x=alt.X("fraction:Q", title="fraction of most-uncertain pixels removed"),
                y=alt.Y("mae:Q", title="error of the remaining pixels", scale=alt.Scale(zero=False)),
                color=alt.Color("method:N", title=None, legend=alt.Legend(orient="top")),
                strokeDash=alt.StrokeDash("curve:N", title=None)).properties(height=300), width="stretch")
            st.caption("If uncertainty is meaningful, removing the most uncertain pixels lowers the remaining error "
                       "(solid line falls), ideally as fast as the oracle (dashed).")
    section("Accuracy cost of each uncertainty method")
    for code in codes:
        exp_card(ex[ex.experiment == code], code)

elif topic == "Baselines & seeds":
    exp_card(ex[ex.experiment == "baseline"], "baseline")
    if main is not None and "n_seeds" in main and (main.n_seeds > 1).any():
        section("Seed variance - the noise floor", "Same model, same data, different random initialisation.")
        sv = main[main.n_seeds > 1].copy()
        sv["model"] = sv.group.map(lambda g: lab.get(g, g))
        sv["test set"] = sv.split.map(SPLIT_NAMES)
        st.dataframe(sv[["model", "test set", "n_seeds", "psnr", "psnr_std", "ssim", "ssim_std"]], hide_index=True,
                     width="stretch",
                     column_config={"psnr": st.column_config.NumberColumn("PSNR", format="%.2f dB"),
                                    "psnr_std": st.column_config.NumberColumn("± PSNR", format="%.2f"),
                                    "ssim": st.column_config.NumberColumn("SSIM", format="%.4f"),
                                    "ssim_std": st.column_config.NumberColumn("± SSIM", format="%.4f")})
        st.caption("Differences between models smaller than about twice this spread are not meaningful.")
else:
    for code in codes:
        rows = ex[ex.experiment == code]
        if len(rows):
            exp_card(rows, code)
    if topic == "E3 · U-Net" and data["erf_stats"] is not None:
        section("E3c · Effective receptive field",
                "How far from a pixel does the model actually look? Gradient of the centre output w.r.t. the input.")
        import os
        p = os.path.join(root, "results", "erf.png")
        if os.path.exists(p):
            st.image(p)
        e = data["erf_stats"]
        st.dataframe(e[["label", "radius50", "radius90"]].rename(columns={
            "label": "Model", "radius50": "50% of influence within (px)", "radius90": "90% within (px)"}),
            hide_index=True, width="stretch")

with st.expander("For your report - prediction → result → explanation"):
    rows = ex[ex.experiment.isin(codes)]
    lines = []
    for run, g in rows.groupby("run", sort=False):
        code = g.experiment.iloc[0]
        q, pred, metric, expect = SPEC.get(code, ("", "", "psnr", None))
        r = g.set_index("split")
        primary = "test_id" if "test_id" in r.index else r.index[0]
        v, _ = verdict(r.loc[primary, f"delta_{metric}"], expect, metric, g.reference.iloc[0])
        lines.append(f"**{g.label.iloc[0]}** (vs {g.reference_label.iloc[0]})  \n"
                     f"- Prediction: {pred}  \n"
                     f"- Result: Δ{metric.upper()} = {fmt_delta(r.loc[primary, f'delta_{metric}'], metric)} → {v}  \n"
                     f"- Explanation: *why did this happen?*")
    st.markdown("\n\n".join(lines) if lines else "No runs for this topic.")
