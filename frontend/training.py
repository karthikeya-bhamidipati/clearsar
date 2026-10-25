import altair as alt
import pandas as pd
import streamlit as st

from frontend.common import ACCENT, AMBER, MUTED, ROSE, VIOLET, hero, kpi, load_final, no_final, section

fin = load_final()
hero("How it learned", "Eleven hours on two GPUs, <span class='cs-grad'>step by step</span>",
     "The training log of the final model: how the loss fell, what validation said every 20 minutes, and how the "
     "generator and the discriminator played against each other once the GAN joined.")
if fin is None:
    no_final()

tr, va, sm, cfg = fin["train"], fin["val"], fin["summary"], fin["config"].get("cfg", {})
if tr is None or va is None:
    st.warning("The training history (logs/train_history.csv, logs/val_history.csv) is missing from this run.")
    st.stop()

c = st.columns(4)
kpi(c[0], "Training steps", f"{sm.get('steps', int(tr.step.max())):,}",
    f"batch {sm.get('batch_per_gpu', '?')} per GPU × {sm.get('world', '?')} GPUs")
kpi(c[1], "Training time", f"{sm.get('train_hours', tr.time_h.max()):.1f} h", "time-based schedule: always fits the session")
kpi(c[2], "Complete passes over the data", f"{sm.get('epochs_started', int(tr.epoch.max()))}",
    f"{sm.get('fresh_fraction', tr.fresh.mean()):.0%} of the batches were fresh from disk")
kpi(c[3], "Parameters", f"{sm.get('generator_params', 0) / 1e6:.1f} M",
    f"discriminator {sm.get('discriminator_params', 0) / 1e6:.1f} M")
gan_h = tr[tr.phase == "B"].time_h.min() if (tr.phase == "B").any() else None
rule = alt.Chart(pd.DataFrame({"time_h": [gan_h]})).mark_rule(color=ROSE, strokeDash=[5, 4]).encode(x="time_h:Q") \
    if gan_h is not None else None


def with_rule(ch):
    return ch + rule if rule is not None else ch


# ---------------------------------------------------------------- loss + validation
section("Loss and validation", "Dashed red line = the GAN joins (part B). Validation uses the averaged (EMA) weights "
        "on 600 patches from 10 scenes that are never trained on.")
t1, t2, t3 = st.tabs(["Loss", "Validation accuracy", "Sharpness"])
with t1:
    d = tr.melt(id_vars="time_h", value_vars=[c for c in ("l1", "ssim_loss") if c in tr], var_name="term")
    d["term"] = d.term.map({"l1": "L1 (train)", "ssim_loss": "1 - SSIM (train)"})
    v = va[["time_h", "mae"]].assign(term="L1 (validation)").rename(columns={"mae": "value"})
    d = pd.concat([d, v])
    st.altair_chart(with_rule(alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=alt.X("time_h:Q", title="hours"), y=alt.Y("value:Q", title="loss", scale=alt.Scale(type="log")),
        color=alt.Color("term:N", title=None, scale=alt.Scale(range=[ACCENT, AMBER, VIOLET]),
                        legend=alt.Legend(orient="bottom")))).properties(height=330), width="stretch")
    st.caption("Training loss is noisy because every batch is different; validation is the honest curve.")
with t2:
    d = va.melt(id_vars=["time_h", "step"], value_vars=["psnr", "psnr_aligned"], var_name="m")
    d["m"] = d.m.map({"psnr": "PSNR", "psnr_aligned": "PSNR, date shift removed"})
    st.altair_chart(with_rule(alt.Chart(d).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("time_h:Q", title="hours"), y=alt.Y("value:Q", title="dB", scale=alt.Scale(zero=False)),
        color=alt.Color("m:N", title=None, scale=alt.Scale(range=[ACCENT, VIOLET]), legend=alt.Legend(orient="bottom")),
        tooltip=["step", alt.Tooltip("value:Q", format=".2f")])).properties(height=330), width="stretch")
with t3:
    st.altair_chart(with_rule(alt.Chart(va).mark_line(point=True, strokeWidth=2, color=AMBER).encode(
        x=alt.X("time_h:Q", title="hours"), y=alt.Y("sharpness:Q", title="sharpness (1 = as sharp as reality)"),
        tooltip=["step", alt.Tooltip("sharpness:Q", format=".2f")])).properties(height=330), width="stretch")
    st.caption("Pixel losses produce blurry averages (sharpness < 1). The GAN's job in part B is to push this towards 1.")

# ---------------------------------------------------------------- GAN dynamics
b = tr[tr.phase == "B"]
if len(b) and "d_real" in b:
    section("The GAN game (part B)",
            "The discriminator D scores real clear images towards 1 and generated ones towards 0. A healthy game: "
            "D(real) and D(fake) stay apart but neither reaches 1 or 0 - then D keeps giving the generator useful advice.")
    d = b.melt(id_vars="time_h", value_vars=["d_real", "d_fake"], var_name="k")
    d["k"] = d.k.map({"d_real": "D(real image)", "d_fake": "D(generated image)"})
    c1, c2 = st.columns(2, gap="large")
    c1.altair_chart(alt.Chart(d).mark_line(strokeWidth=2).encode(
        x=alt.X("time_h:Q", title="hours"), y=alt.Y("value:Q", title="discriminator score", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("k:N", title=None, scale=alt.Scale(domain=["D(real image)", "D(generated image)"],
                                                           range=[ACCENT, ROSE]), legend=alt.Legend(orient="bottom"))
    ).properties(height=300), width="stretch")
    if "lam" in b:
        c2.altair_chart(alt.Chart(b).mark_line(strokeWidth=2, color=VIOLET).encode(
            x=alt.X("time_h:Q", title="hours"), y=alt.Y("lam:Q", title="adversarial weight λ")).properties(
            height=300, title="Adaptive GAN weight λ"), width="stretch")
        c2.caption("λ is re-set so the GAN's gradient stays at 20 % of the pixel loss's: it falls as D gets stronger.")

# ---------------------------------------------------------------- engineering
section("Under the hood", "How the run used the hardware.")
c1, c2 = st.columns(2, gap="large")
c1.altair_chart(alt.Chart(tr).mark_line(strokeWidth=1.5, color=ACCENT).encode(
    x=alt.X("time_h:Q", title="hours"), y=alt.Y("patches_read_s:Q", title="patches / s")
).properties(height=240, title="Patches read from disk per second"), width="stretch")
c2.altair_chart(alt.Chart(tr).mark_line(strokeWidth=1.5, color=MUTED).encode(
    x=alt.X("time_h:Q", title="hours"), y=alt.Y("it_s:Q", title="steps / s")).properties(
    height=240, title="Training steps per second (16 patches each)"),
    width="stretch")
events = sm.get("nan_guard_events") or []
st.markdown(f"**NaN guard:** {'no events - every step was finite.' if not events else f'{len(events)} event(s):'}")
for e in events:
    st.markdown(f"- step {e['step']}: {e['action']} ({e['why']})")
with st.expander("All hyper-parameters of this run"):
    st.json(cfg)
