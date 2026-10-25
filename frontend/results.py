import altair as alt
import numpy as np
import streamlit as st

from frontend.common import (METRICS, SPLIT_COLORS, SPLIT_NAMES, hero, labels, load, no_results, section)

root, data = load()
hero("Model comparison", "Which architecture <span class='cs-grad'>reconstructs best?</span>",
     "Every model was trained under the identical protocol and scored on held-out scenes it never saw. "
     "Error bars show the spread over training seeds; differences smaller than that are noise.")
if data is None:
    no_results()

lab = labels(data)
main = data["main"].copy()
main["test_set"] = main.split.map(SPLIT_NAMES)
main["model"] = main.group.map(lambda g: lab.get(g, g))
avail = [m for m in METRICS if m in main and main[m].notna().any()]

c1, c2 = st.columns([2, 1])
metric = c1.segmented_control("Metric", avail, default="psnr",
                              format_func=lambda m: METRICS[m][0], key="res_metric") or "psnr"
name, unit, higher = METRICS[metric]
c2.markdown(f"<div style='text-align:right;color:#8B95A7;padding-top:2rem'>{'Higher' if higher else 'Lower'} is better</div>",
            unsafe_allow_html=True)

order = main.sort_values(["split", metric], ascending=[True, not higher]).drop_duplicates("group").model.tolist()
base = alt.Chart(main).encode(
    y=alt.Y("model:N", sort=order, title=None, axis=alt.Axis(labelLimit=260, labelFontSize=12)),
    yOffset=alt.YOffset("test_set:N", sort=[SPLIT_NAMES["test_id"], SPLIT_NAMES["ood"]]),
    color=alt.Color("test_set:N", title=None, legend=alt.Legend(orient="top"),
                    scale=alt.Scale(domain=[SPLIT_NAMES["test_id"], SPLIT_NAMES["ood"]],
                                    range=[SPLIT_COLORS["test_id"], SPLIT_COLORS["ood"]])))
# dot plot, not bars: the models differ by tenths of a dB, which a bar starting at zero would hide
xscale = alt.Scale(zero=False, nice=True)
dots = base.mark_circle(size=130, opacity=1).encode(
    x=alt.X(f"{metric}:Q", title=f"{name} {unit}".strip(), scale=xscale),
    tooltip=["model", "test_set", alt.Tooltip(f"{metric}:Q", format=".3f")])
layers = [dots]
if f"{metric}_std" in main and main[f"{metric}_std"].notna().any():
    eb = main.assign(lo=main[metric] - main[f"{metric}_std"].fillna(0), hi=main[metric] + main[f"{metric}_std"].fillna(0))
    layers.insert(0, alt.Chart(eb).mark_rule(color="#A3AEC2", strokeWidth=2).encode(
        y=alt.Y("model:N", sort=order), yOffset=alt.YOffset("test_set:N", sort=[SPLIT_NAMES["test_id"], SPLIT_NAMES["ood"]]),
        x=alt.X("lo:Q", scale=xscale), x2="hi:Q"))
st.altair_chart(alt.layer(*layers).properties(height=40 * main.group.nunique() + 40), width="stretch")

# ---------------------------------------------------------------- leaderboard
section("Leaderboard")
split = st.segmented_control("Test set", ["test_id", "ood"], default="test_id", format_func=SPLIT_NAMES.get,
                             key="res_split") or "test_id"
t = main[main.split == split].copy()
t = t.sort_values(metric, ascending=not higher)
cols = ["model"] + [m for m in ["psnr", "ssim", "sam", "mae", "fid"] if m in t]
show = t[cols].rename(columns={"model": "Model", **{m: METRICS[m][0] for m in METRICS}})
if "n_seeds" in t:
    show["Seeds"] = t.n_seeds.values
show.insert(0, "#", range(1, len(show) + 1))
st.dataframe(show.set_index("#"), width="stretch",
             column_config={METRICS["psnr"][0]: st.column_config.NumberColumn(format="%.2f dB"),
                            METRICS["ssim"][0]: st.column_config.NumberColumn(format="%.4f"),
                            METRICS["sam"][0]: st.column_config.NumberColumn(format="%.2f°"),
                            METRICS["mae"][0]: st.column_config.NumberColumn(format="%.4f"),
                            METRICS["fid"][0]: st.column_config.NumberColumn(format="%.1f")})
st.caption("Identity = the cloudy image itself. Every useful model must beat it. "
           "PSNR/SSIM: pixel and structure fidelity · SAM: colour (spectral) correctness · FID: realism.")

# ---------------------------------------------------------------- breakdowns
section("Where do the models struggle?")
tab1, tab2 = st.tabs(["By cloud cover", "By country"])
main_runs = set(main.group)
with tab1:
    b = data["metrics_by_cloud_bin"]
    if b is not None:
        b = b[(b.split == split) & b.run.isin(main_runs)].copy()
        b["model"] = b.run.map(lambda r: lab.get(r, r))
        m2 = metric if metric in b else "psnr"
        st.altair_chart(alt.Chart(b).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2.5).encode(
            x=alt.X("cloud_bin:N", title="Cloud cover of the patch",
                    sort=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"], axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{m2}:Q", title=METRICS[m2][0], scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title=None, legend=alt.Legend(orient="bottom", columns=3, labelLimit=260)),
            tooltip=["model", "cloud_bin", "n_patches", alt.Tooltip(f"{m2}:Q", format=".2f")],
        ).properties(height=330), width="stretch")
        st.caption("Thick cloud leaves only the radar - this is where architectures differ most.")
with tab2:
    c = data["metrics_by_country"]
    if c is not None:
        c = c[(c.split == split) & c.run.isin(main_runs)].copy()
        c["model"] = c.run.map(lambda r: lab.get(r, r))
        m2 = metric if metric in c else "psnr"
        st.altair_chart(alt.Chart(c).mark_rect(cornerRadius=3).encode(
            x=alt.X("country:N", title=None), y=alt.Y("model:N", title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color(f"{m2}:Q", title=METRICS[m2][0], scale=alt.Scale(scheme="tealblues", reverse=not higher)),
            tooltip=["model", "country", "n_patches", alt.Tooltip(f"{m2}:Q", format=".2f")],
        ).properties(height=40 * c.model.nunique() + 30), width="stretch")
        st.caption("IN = the Indian scene held out for an India-only slice.")
