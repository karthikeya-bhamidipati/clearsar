import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from frontend.common import (CLOUD_ORDER, FINAL_COLORS, FINAL_LABELS, FINAL_RUNS, hero, kpi, load_final, no_final,
                             section)

fin = load_final()
hero("Results · one honest test", "How close does it get <span class='cs-grad'>to the real image?</span>",
     "Everything here is measured once, at the end, on 18 scenes the model never saw - from every continent. "
     "The same patches are scored for the cloudy input itself, the Phase 1 U-Net, and ClearSAR-Net before and "
     "after the GAN part.")
if fin is None:
    no_final()

METRIC = {"psnr": ("PSNR (dB)", True, "Pixel accuracy, all 13 bands. +3 dB = half the squared error."),
          "psnr_aligned": ("PSNR, date shift removed (dB)", True,
                           "PSNR after fitting brightness per band - removes the part of the error caused by the "
                           "clear image being photographed on a different day."),
          "ssim": ("SSIM", True, "Structure similarity (1 = identical local patterns)."),
          "sam": ("SAM (°)", False, "Spectral angle - is the colour / spectrum right, regardless of brightness?"),
          "mae": ("MAE", False, "Mean absolute reflectance error."),
          "sharpness": ("Sharpness ratio", None, "Edge energy of the output / of the truth. 1 = as sharp as reality, "
                                                 "< 1 blurry, > 1 invented detail."),
          "detail_err": ("Fine-detail error", False, "Error in the high-frequency part (edges, texture)."),
          "colour_err": ("Coarse colour error", False, "Error after 16×16 averaging (large-scale colour).")}
order = [r for r in FINAL_RUNS if r in set(fin["test_summary"].run)]
lab = {r: FINAL_LABELS.get(r, r) for r in order}
color = alt.Scale(domain=[lab[r] for r in order], range=[FINAL_COLORS[r] for r in order])

# ---------------------------------------------------------------- headline table
summ = fin["test_summary"].set_index("run").loc[order].reset_index()
summ["Model"] = summ.run.map(lab)
c = st.columns(4)
f = summ.set_index("run").loc["final"]
idn = summ.set_index("run").loc["identity"] if "identity" in set(summ.run) else None
kpi(c[0], "ClearSAR-Net PSNR", f"{f.psnr:.2f} dB", f"+{f.psnr - idn.psnr:.1f} dB over the cloudy input" if idn is not None else "")
kpi(c[1], "SSIM", f"{f.ssim:.3f}", f"cloudy input {idn.ssim:.3f}" if idn is not None else "")
kpi(c[2], "Spectral angle", f"{f.sam:.1f}°", f"cloudy input {idn.sam:.1f}°" if idn is not None else "")
kpi(c[3], "Test patches", f"{int(f.n):,}", "18 scenes, never seen in training")

section("All metrics", "Identity = the cloudy image itself; any useful model must beat it.")
cols = ["Model"] + [m for m in METRIC if m in summ]
st.dataframe(summ[cols].rename(columns={m: METRIC[m][0] for m in METRIC}), hide_index=True, width="stretch",
             column_config={METRIC["psnr"][0]: st.column_config.NumberColumn(format="%.2f"),
                            METRIC["psnr_aligned"][0]: st.column_config.NumberColumn(format="%.2f"),
                            METRIC["ssim"][0]: st.column_config.NumberColumn(format="%.4f"),
                            METRIC["sam"][0]: st.column_config.NumberColumn(format="%.2f"),
                            METRIC["mae"][0]: st.column_config.NumberColumn(format="%.4f"),
                            METRIC["sharpness"][0]: st.column_config.NumberColumn(format="%.2f"),
                            METRIC["detail_err"][0]: st.column_config.NumberColumn(format="%.4f"),
                            METRIC["colour_err"][0]: st.column_config.NumberColumn(format="%.4f")})
with st.expander("What do these metrics mean?"):
    for m, (name, _, text) in METRIC.items():
        st.markdown(f"**{name}** - {text}")

# ---------------------------------------------------------------- breakdowns
section("Where is it hard?", "Pick a metric; every chart below uses it.")
avail = [m for m in METRIC if m in fin["test_by_cloud"]]
metric = st.segmented_control("Metric", avail, default="psnr", format_func=lambda m: METRIC[m][0].split(" (")[0],
                              key="fr_metric") or "psnr"
mname = METRIC[metric][0]
t1, t2, t3 = st.tabs(["By cloud cover", "By continent", "By country"])
with t1:
    b = fin["test_by_cloud"].copy()
    b["Model"] = b.run.map(lab)
    st.altair_chart(alt.Chart(b[b.run.isin(order)]).mark_line(point=alt.OverlayMarkDef(size=70), strokeWidth=2.5).encode(
        x=alt.X("cloud_bin:N", title="Cloud cover of the patch", sort=CLOUD_ORDER, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{metric}:Q", title=mname, scale=alt.Scale(zero=False)),
        color=alt.Color("Model:N", scale=color, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", "cloud_bin", "n", alt.Tooltip(f"{metric}:Q", format=".3f")]).properties(height=340),
        width="stretch")
    st.caption("Under thin cloud most of the answer is visible; under thick cloud only the radar is left - the gap "
               "between the cloudy input and the models shows what the network adds.")
with t2:
    b = fin["test_by_continent"].copy()
    b["Model"] = b.run.map(lab)
    st.altair_chart(alt.Chart(b[b.run.isin(order)]).mark_bar(cornerRadiusEnd=3).encode(
        y=alt.Y("continent:N", title=None), yOffset=alt.YOffset("Model:N", sort=[lab[r] for r in order]),
        x=alt.X(f"{metric}:Q", title=mname, scale=alt.Scale(zero=metric not in ("psnr", "psnr_aligned", "ssim"))),
        color=alt.Color("Model:N", scale=color, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Model", "continent", "n", alt.Tooltip(f"{metric}:Q", format=".3f")]).properties(height=360),
        width="stretch")
with t3:
    b = fin["test_by_country"]
    w = b[b.run.isin(order)].pivot_table(index=["country", "continent"], columns="run", values=metric).reset_index()
    w = w.merge(b[b.run == "final"][["country", "n"]], on="country")
    w = w.rename(columns={**lab, "country": "Country", "continent": "Continent", "n": "Patches"})
    st.dataframe(w, hide_index=True, width="stretch")
    if "IN" in set(b.country):
        st.caption("IN = India, reported separately as planned.")

# ---------------------------------------------------------------- what the GAN changed
pp = fin["test_per_patch"]
if pp is not None and {"before_gan", "final"} <= set(pp.run):
    section("What did the GAN change?",
            "The same generator before and after part B. A GAN trades a little pixel accuracy for realistic texture: "
            "sharpness should move towards 1 while PSNR stays roughly where it was.")
    s2 = summ.set_index("run")
    d = pd.DataFrame({"Metric": [METRIC[m][0] for m in ("psnr", "ssim", "sharpness", "detail_err", "sam")],
                      "Before GAN": [s2.loc["before_gan", m] for m in ("psnr", "ssim", "sharpness", "detail_err", "sam")],
                      "Final (GAN)": [s2.loc["final", m] for m in ("psnr", "ssim", "sharpness", "detail_err", "sam")]})
    d["Change"] = d["Final (GAN)"] - d["Before GAN"]
    c1, c2 = st.columns([1, 1.3], gap="large")
    c1.dataframe(d, hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ("Before GAN", "Final (GAN)", "Change")})
    w = pp[pp.run.isin(["before_gan", "final"])].pivot_table(index="row", columns="run", values=["psnr", "sharpness"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.sample(n=min(len(w), 1500), random_state=0).reset_index()
    lo, hi = float(w[["sharpness_before_gan", "sharpness_final"]].min().min()),         float(w[["sharpness_before_gan", "sharpness_final"]].max().max())
    diag = alt.Chart(pd.DataFrame({"a": [lo, hi], "b": [lo, hi]})).mark_line(color="#8B95A7", strokeDash=[4, 4]).encode(
        x="a:Q", y="b:Q")
    pts = alt.Chart(w).mark_circle(size=14, opacity=.45, color=FINAL_COLORS["final"]).encode(
        x=alt.X("sharpness_before_gan:Q", title="Sharpness before GAN", scale=alt.Scale(zero=False)),
        y=alt.Y("sharpness_final:Q", title="Sharpness after GAN", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip("psnr_before_gan:Q", format=".2f"), alt.Tooltip("psnr_final:Q", format=".2f")])
    c2.altair_chart((diag + pts).properties(height=300, title="Sharpness per test patch"), width="stretch")
    c2.caption("Each dot is one test patch; above the dashed diagonal = sharper after the GAN.")

# ---------------------------------------------------------------- uncertainty and the cloud gate
if pp is not None and "tta_std" in pp:
    fp = pp[pp.run == "final"].dropna(subset=["tta_std"])
    section("Does the model know when it is wrong?",
            "Uncertainty from test-time augmentation: the model sees the patch four times, flipped; where its four "
            "answers disagree, it is unsure. A useful uncertainty grows with the real error.")
    unc = fin["uncertainty"]
    c1, c2 = st.columns(2, gap="large")
    smp = fp.sample(n=min(len(fp), 2000), random_state=0)
    c1.altair_chart(alt.Chart(smp).mark_circle(size=14, opacity=.45, color=FINAL_COLORS["final"]).encode(
        x=alt.X("tta_std:Q", title="Uncertainty (spread of the 4 flipped predictions)"),
        y=alt.Y("mae:Q", title="Real error (MAE)")).properties(
        height=300, title=f"Spearman correlation {unc.get('spearman_tta_std_vs_mae', float('nan')):.2f}"), width="stretch")
    if "gate_mean" in fp:
        c2.altair_chart(alt.Chart(smp).mark_circle(size=14, opacity=.45, color=FINAL_COLORS["before_gan"]).encode(
            x=alt.X("cloud_frac:Q", title="Cloud cover of the patch", axis=alt.Axis(format="%")),
            y=alt.Y("gate_mean:Q", title="Share of pixels the gate keeps", axis=alt.Axis(format="%"))).properties(
            height=300, title="The learned gate: more cloud -> less of the input is kept"), width="stretch")
    st.caption("The gate was never told where the clouds are - it learned it from the loss alone.")
