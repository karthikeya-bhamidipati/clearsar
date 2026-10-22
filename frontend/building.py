import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from frontend.common import (compare_slider, hero, kpi, lab_note, load_v4, pill, rgb_u8, section, to_uint8, v4_labels)

hero("Plan v4 · ClearSAR-Net", "One model, <span class='cs-grad'>built step by step</span>",
     "Each stage changes one thing, is judged on validation scenes only, and hands its winner to the next stage. "
     "The 25 test scenes - including all of Phase 1's - are used once, at the very end.")
v4 = load_v4()
if v4 is None:
    st.info("No plan-v4 results yet. Run `notebooks/kaggle_pipeline_v4.py` on Kaggle, download "
            "`clearsar_v4_output.zip` and extract it into `outputs/v4/`.")
    st.stop()

lab = v4_labels(v4)
runs = v4["v4_runs"].set_index("run")
dec = v4["config"].get("decisions", {})
METR = [("psnr", "PSNR", "dB", True), ("psnr_aligned", "PSNR (date shift removed)", "dB", True), ("ssim", "SSIM", "", True),
        ("sharpness", "Sharpness (1 = real)", "", None), ("colour_err", "Colour error", "", False),
        ("detail_err", "Detail error", "", False)]


def val_table(names):
    rows = []
    for n in names:
        h = v4["history"].get(n)
        if h is None or h.empty:
            continue
        sc = h.val_l1 * (1 + (1 - h.get("val_sharpness", 1)).abs()) if "val_sharpness" in h else h.val_l1
        b = h.loc[h.val_l1.idxmin()]
        rows.append(dict(run=n, model=lab.get(n, n), **{"val L1": b.val_l1, "val PSNR": b.val_psnr,
                                                       "val SSIM": b.val_ssim, "val sharpness": b.get("val_sharpness"),
                                                       "balanced score": float(sc.min())}))
    return pd.DataFrame(rows)


def notes(stage):
    n = lab_note(stage)
    if n:
        st.markdown("<div class='cs-card'><h4>Your lab notes</h4>" + "".join(
            f"<p><b style='color:#E6EAF2'>{k.title()}:</b> {v}</p>" for k, v in n.items()) + "</div>",
            unsafe_allow_html=True)
    else:
        st.caption(f"Write your prediction and explanation for this stage in `docs/lab_notes.md` (Stage {stage}) - "
                   "they will appear here.")


def lesson(title, text):
    st.markdown(f"<div class='cs-rule'><span class='id'>LESSON</span><div class='txt'><b>{title}.</b> {text}</div></div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- headline
final = "v4_final" if "v4_final" in runs.index else None
p1 = [r for r in ("p1_A_unet", "p1_B_swin", "p1_C_vit", "p1_tiny_cnn") if r in runs.index]
best_p1 = max(p1, key=lambda r: runs.loc[r, "psnr"]) if p1 else None
if final:
    c = st.columns(4)
    kpi(c[0], "ClearSAR-Net, test PSNR", f"{runs.loc[final, 'psnr']:.2f} dB",
        f"Phase-1 best: {runs.loc[best_p1, 'psnr']:.2f} dB" if best_p1 else "")
    kpi(c[1], "Date shift removed", f"{runs.loc[final, 'psnr_aligned']:.2f} dB",
        "quality of the image itself (stage 0)")
    kpi(c[2], "Sharpness", f"{runs.loc[final, 'sharpness']:.2f}",
        f"Phase-1 best: {runs.loc[best_p1, 'sharpness']:.2f}" if best_p1 else "1.0 = as sharp as reality")
    if "identity" in runs.index:
        kpi(c[3], "Gain over cloudy input", f"+{runs.loc[final, 'psnr'] - runs.loc['identity', 'psnr']:.1f} dB",
            f"cloudy input: {runs.loc['identity', 'psnr']:.2f} dB")
    st.markdown("<div style='margin-top:.6rem'>" + " ".join(
        pill(f"{k}: {v}", "accent") for k, v in dec.items()) + "</div>", unsafe_allow_html=True)
    st.caption("Decisions taken automatically on the validation scenes: output mode (stage 2), loss (stage 4), "
               "context (stage 5).")

STAGES = ["0 · Data", "1 · Mechanics", "2 · Keep what's seen", "3 · Diversity", "4 · Loss", "5 · Context",
          "6 · Final model", "7 · Test"]
stage = st.pills("Stage", STAGES, default=STAGES[0], key="v4_stage") or STAGES[0]
k = int(stage[0])

if k == 0:
    section("Stage 0 · What is achievable at all?")
    lesson("Irreducible error", "The cloudy and the clear image are taken on different dates. Even on visible ground they "
           "differ, and under thick cloud the true colours are not in the input at all. No model can remove that part of "
           "the error - so measure it first.")
    t = v4["stage0_ceiling"]
    if t is not None:
        m = t.melt(id_vars=["cloud_bin", "n"], value_vars=[c for c in ["identity_psnr", "clear_px_psnr", "oracle_shift_psnr"]
                                                          if c in t], var_name="measure", value_name="PSNR")
        names = {"identity_psnr": "cloudy image, whole patch", "clear_px_psnr": "cloudy image, visible pixels only",
                 "oracle_shift_psnr": "visible pixels after removing the date shift (oracle)"}
        m["measure"] = m.measure.map(names)
        st.altair_chart(alt.Chart(m).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("cloud_bin:N", title="cloud cover", sort=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]),
            y=alt.Y("PSNR:Q", title="PSNR vs the clear image (dB)", scale=alt.Scale(zero=False)),
            color=alt.Color("measure:N", title=None, legend=alt.Legend(orient="top", columns=1)),
            tooltip=["cloud_bin", "measure", alt.Tooltip("PSNR:Q", format=".2f"), "n"]).properties(height=320),
            width="stretch")
        st.caption("Visible pixels score only ~25 dB against the 'truth', but ~31-40 dB once a per-band brightness/colour "
                   "shift is removed: most of the visible-ground error is the date difference, not the model. That is why "
                   "every model's PSNR stalls in the high 20s, and why ClearSAR also reports PSNR with the date shift removed.")
    c1, c2 = st.columns(2)
    for col, f, cap in [(c1, "stage0_triplets.png", "Real triplets across cloud levels"),
                        (c2, "stage0_cloud_hist.png", "Cloud cover of the validation scenes")]:
        p = os.path.join(v4["root"], "results", f)
        if os.path.exists(p):
            col.image(p, caption=cap)
    notes(0)

elif k == 1:
    section("Stage 1 · How training works")
    lesson("Two checks before any long run", "(1) Can the network memorise one batch? If not, something is broken. "
           "(2) Which learning rate? Raise it exponentially and watch where the loss explodes.")
    c1, c2 = st.columns(2)
    of, lr = v4["stage1_overfit"], v4["stage1_lr_range"]
    if of is not None:
        c1.altair_chart(alt.Chart(of).mark_line(strokeWidth=2.5, color="#2DD4BF").encode(
            x=alt.X("step:Q"), y=alt.Y("l1:Q", title="L1 on the batch", scale=alt.Scale(type="log"))).properties(
            height=260, title="Overfit one batch"), width="stretch")
    if lr is not None:
        c2.altair_chart(alt.Chart(lr).mark_line(strokeWidth=2.5, color="#A78BFA").encode(
            x=alt.X("lr:Q", scale=alt.Scale(type="log"), title="learning rate"),
            y=alt.Y("smoothed:Q", title="smoothed loss", scale=alt.Scale(zero=False))).properties(
            height=260, title="Learning-rate range test"), width="stretch")
    h = v4["history"].get("s1_direct")
    if h is not None:
        m = h.melt(id_vars="step", value_vars=["train_rec", "val_l1"], var_name="curve", value_name="L1")
        st.altair_chart(alt.Chart(m).mark_line(point=True).encode(
            x="step:Q", y=alt.Y("L1:Q", scale=alt.Scale(zero=False)), color="curve:N").properties(
            height=260, title="Plain U-Net on the new worldwide data: training vs validation"), width="stretch")
        st.caption("Training loss keeps falling; if validation stops falling (or rises) the model starts to memorise "
                   "the training scenes - overfitting. The best validation checkpoint is kept.")
    notes(1)

elif k == 2:
    section("Stage 2 · Keep what the satellite saw")
    lesson("Residual learning", "Predicting a correction to the input is easier than redrawing the whole image. The gated "
           "model also learns a mask: where to keep the cloudy image, where to fill in.")
    t = val_table(["s1_direct", "s2_residual", "s2_gated"])
    if len(t):
        t["chosen"] = t.run.map(lambda r: "✓" if {"s1_direct": "direct", "s2_residual": "residual",
                                                  "s2_gated": "gated"}[r] == dec.get("mode") else "")
        st.dataframe(t.drop(columns=["run"]).set_index("model").style.format(precision=4), width="stretch")
        st.caption("Decision rule: lowest validation L1.")
    s, meta = v4["samples"], v4["samples_meta"]
    if s is not None:
        names = [f"{m['country']} · {int(m['cloud_frac'] * 100)}% cloud" for m in meta["samples"]]
        i = st.selectbox("Patch", range(len(names)), format_func=lambda j: names[j], key="v4_s2_patch")
        cols = st.columns(4)
        cols[0].image(rgb_u8(s["x"][i, 2:].astype(np.float32)), caption="cloudy input")
        for col, r in zip(cols[1:], ["s1_direct", "s2_residual", "s2_gated"]):
            if f"pred_{r}" in s:
                col.image(s[f"pred_{r}"][i], caption=lab.get(r, r).replace("Stage 2 - ", "").replace("Stage 1 - ", ""))
        if "gate_s2_gated" in s:
            c1, c2 = st.columns([1, 3])
            c1.image(to_uint8(s["gate_s2_gated"][i].astype(np.float32), cmap="viridis", vmax=1.0),
                     caption="learned gate m (bright = keep the input)")
            c2.markdown("The gate is the model's own cloud mask: it learned **without any cloud labels** where the input "
                        "can be trusted. Compare it with the clouds in the cloudy image.")
    notes(2)

elif k == 3:
    section("Stage 3 · Does variety beat quantity?")
    lesson("Generalisation", "Phase 1 showed that more patches from the same scenes barely help. Here the number of patches "
           "is fixed at 3 000 while the number of different scenes grows.")
    pts = []
    for n, sc in [("s3_sc33", 33), ("s3_sc66", 66), ("s3_sc132", 132)]:
        if n in runs.index:
            pts.append(dict(scenes=sc, model="3 000 patches", **runs.loc[n, ["psnr", "psnr_aligned", "colour_err"]].to_dict()))
    winner = {"direct": "s1_direct", "residual": "s2_residual", "gated": "s2_gated"}.get(dec.get("mode"))
    if winner in runs.index:
        pts.append(dict(scenes=132, model="all 12 000 patches", **runs.loc[winner, ["psnr", "psnr_aligned", "colour_err"]].to_dict()))
    if pts:
        d = pd.DataFrame(pts)
        st.altair_chart(alt.Chart(d).mark_line(point=alt.OverlayMarkDef(size=90), strokeWidth=3).encode(
            x=alt.X("scenes:Q", title="number of training scenes", scale=alt.Scale(type="log")),
            y=alt.Y("psnr:Q", title="test PSNR (dB)", scale=alt.Scale(zero=False)),
            color=alt.Color("model:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=["scenes", "model", alt.Tooltip("psnr:Q", format=".2f"), alt.Tooltip("colour_err:Q", format=".4f")],
        ).properties(height=320), width="stretch")
        st.dataframe(d.set_index(["model", "scenes"]).style.format(precision=4), width="stretch")
    notes(3)

elif k == 4:
    section("Stage 4 · What does the loss optimise?")
    lesson("Perception-distortion trade-off", "L1 is minimised by the median of all plausible answers - smooth. SSIM rewards "
           "structure, a perceptual (VGG) loss rewards texture, an adversarial loss rewards realism. Checkpoints are now "
           "chosen by val L1 × (1 + |1 − sharpness|) so a blurry model cannot win by default.")
    mode_run = {"direct": "s1_direct", "residual": "s2_residual", "gated": "s2_gated"}.get(dec.get("mode"))
    t = val_table([r for r in [mode_run, "s4_ssim", "s4_perc", "s4_adv"] if r])
    if len(t):
        opt = {mode_run: "l1", "s4_ssim": "ssim", "s4_perc": "perc", "s4_adv": "adv"}
        t["chosen"] = t.run.map(lambda r: "✓" if opt.get(r) == dec.get("loss") else "")
        st.altair_chart(alt.Chart(t).mark_circle(size=220).encode(
            x=alt.X("val sharpness:Q", scale=alt.Scale(zero=False), title="validation sharpness (1 = real)"),
            y=alt.Y("val PSNR:Q", scale=alt.Scale(zero=False)), color=alt.Color("model:N", title=None),
            tooltip=list(t.columns)).properties(height=320, title="Sharp vs accurate"), width="stretch")
        st.dataframe(t.drop(columns=["run"]).set_index("model").style.format(precision=4), width="stretch")
    notes(4)

elif k == 5:
    section("Stage 5 · Context from far away")
    lesson("Receptive field and attention", "Under thick cloud a pixel has to be inferred from context far away. A global "
           "self-attention bottleneck lets every position see every other; a wider U-Net with the same number of "
           "parameters checks that any gain is due to attention, not size.")
    loss_run = {"l1": {"direct": "s1_direct", "residual": "s2_residual", "gated": "s2_gated"}.get(dec.get("mode")),
                "ssim": "s4_ssim", "perc": "s4_perc", "adv": "s4_adv"}.get(dec.get("loss"))
    names = [r for r in [loss_run, "s5_attn", "s5_wide"] if r]
    t = val_table(names)
    if len(t):
        st.dataframe(t.drop(columns=["run"]).set_index("model").style.format(precision=4), width="stretch")
    b = v4["v4_by_cloud_bin"]
    if b is not None:
        b = b[b.run.isin(names)].assign(model=lambda d: d.run.map(lambda r: lab.get(r, r)))
        st.altair_chart(alt.Chart(b).mark_line(point=True, strokeWidth=2.5).encode(
            x=alt.X("cloud_bin:N", sort=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"], title="cloud cover"),
            y=alt.Y("psnr:Q", scale=alt.Scale(zero=False), title="test PSNR (dB)"), color=alt.Color("model:N", title=None),
        ).properties(height=300, title="Where does context help?"), width="stretch")
    notes(5)

elif k == 6:
    section("Stage 6 · Train it properly")
    lesson("Scale and stability", "All winners combined, full 256 × 256 patches (more visible context per example), twice the "
           "steps, and an exponential moving average (EMA) of the weights, which smooths out the noise of individual "
           "updates. Two seeds show the spread; a third run adds the uncertainty head for the expert system.")
    hs = {n: v4["history"][n] for n in ("v4_final", "v4_final_s43", "v4_final_unc") if n in v4["history"]}
    if hs:
        m = pd.concat([h.assign(model=lab.get(n, n)) for n, h in hs.items()])
        st.altair_chart(alt.Chart(m).mark_line(point=True).encode(
            x="step:Q", y=alt.Y("val_psnr:Q", scale=alt.Scale(zero=False), title="validation PSNR (dB)"),
            color=alt.Color("model:N", title=None)).properties(height=300), width="stretch")
    fin = [r for r in ("v4_final", "v4_final_s43", "v4_final_unc") if r in runs.index]
    if fin:
        st.dataframe(runs.loc[fin, [m[0] for m in METR] + ["params"]].rename(index=lab).style.format(precision=4),
                     width="stretch")
    notes(6)

else:
    section("Stage 7 · One honest test")
    lesson("Evaluation discipline", "All decisions above were taken on validation scenes. The test scenes are used only here, "
           "for every model - including the Phase-1 models, evaluated on exactly the same 25 scenes.")
    order = [r for r in ["identity", "p1_tiny_cnn", "p1_A_unet", "p1_B_swin", "p1_C_vit", "s1_direct", "v4_final",
                         "v4_final_s43", "v4_final_unc"] if r in runs.index]
    t = runs.loc[order, [m[0] for m in METR] + ["fid"]].rename(index=lab)
    st.dataframe(t.style.format(precision=3).highlight_max(subset=["psnr", "psnr_aligned", "ssim"], color="#134E4A")
                 .highlight_min(subset=["colour_err", "detail_err"], color="#134E4A"), width="stretch")
    reg = v4["v4_by_region"]
    if reg is not None:
        r = reg[reg.run.isin(order)].assign(model=lambda d: d.run.map(lambda x: lab.get(x, x)))
        st.altair_chart(alt.Chart(r).mark_bar(cornerRadiusEnd=3).encode(
            y=alt.Y("model:N", sort=[lab.get(o, o) for o in order], title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("psnr:Q", scale=alt.Scale(zero=False), title="test PSNR (dB)"), color=alt.Color("region:N", title=None),
            yOffset="region:N").properties(height=34 * len(order) + 40), width="stretch")
    s, meta = v4["samples"], v4["samples_meta"]
    if s is not None and "pred_v4_final" in s:
        names = [f"{m['country']} · {m['continent']} · {int(m['cloud_frac'] * 100)}% cloud" for m in meta["samples"]]
        i = st.selectbox("Test patch", range(len(names)), format_func=lambda j: names[j], key="v4_s7_patch")
        other = st.segmented_control("Compare against", ["True clear image", "Cloudy input"], default="True clear image",
                                     key="v4_s7_cmp") or "True clear image"
        left = rgb_u8(s["y"][i].astype(np.float32)) if other == "True clear image" else rgb_u8(s["x"][i, 2:].astype(np.float32))
        compare_slider(left, s["pred_v4_final"][i], other, "ClearSAR-Net", size=520)
        sc = meta["samples"][i]["scores"]
        rows = [dict(model=lab.get(r, r), **{m: v for m, v in sc[r].items() if m in ("psnr", "psnr_aligned", "ssim", "sharpness")})
                for r in order if r in sc]
        st.dataframe(pd.DataFrame(rows).set_index("model").style.format(precision=3), width="stretch")
    notes(7)
