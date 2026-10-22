import numpy as np
import pandas as pd
import streamlit as st

from frontend.common import (compare_slider, hero, labels, load, no_results, rgb_u8, sample_names, section,
                             to_uint8)

from frontend.common import load_v4, v4_labels  # noqa: E402

root, data = load()
v4 = load_v4()
hero("Explore patches", "Look at the <span class='cs-grad'>reconstructions</span> yourself",
     "Held-out test patches from Asia and from other continents, chosen across the whole range of cloud cover. "
     "Your eyes catch artefacts that metrics miss - blur, invented texture, blocky patches.")
sources = (["Plan v4 test scenes"] if v4 is not None and v4["samples"] is not None else []) +           (["Phase 1 test scenes"] if data is not None and data["samples"] is not None else [])
if not sources:
    no_results()
src = st.segmented_control("Gallery", sources, default=sources[0], key="ex_src") or sources[0]
if src.startswith("Plan v4"):
    s, meta, lab = v4["samples"], v4["samples_meta"], v4_labels(v4)
    names = [f"{m['country']} · {m['continent']} · {int(round(m['cloud_frac'] * 100))}% cloud" for m in meta["samples"]]
else:
    s, meta, lab = data["samples"], data["samples_meta"], labels(data)
    names = sample_names(meta)
runs = [r for r in meta["runs"] if f"pred_{r}" in s and r != "identity"]
if src.startswith("Plan v4") and "v4_final" in runs:
    runs = ["v4_final"] + [r for r in runs if r != "v4_final"]

c1, c2 = st.columns([1, 1])
i = c1.selectbox("Patch", range(len(names)), format_func=lambda k: names[k], key="ex_patch")
run = c2.selectbox("Model", runs, format_func=lambda r: lab.get(r, r), key="ex_model")
mode = st.segmented_control("Compare against", ["Cloudy input", "True clear image"], default="Cloudy input",
                            key="ex_mode") or "Cloudy input"

x, y = s["x"][i].astype(np.float32), s["y"][i].astype(np.float32)
pred = s[f"pred_{run}"][i]
truth = rgb_u8(y)
left = rgb_u8(x[2:]) if mode == "Cloudy input" else truth
compare_slider(left, pred, mode, lab.get(run, run), size=600)

info = meta["samples"][i]
sc = info["scores"].get(run, {})
st.markdown(
    f"<div style='text-align:center;color:#8B95A7;margin-top:.4rem'>{info['season'].split('_')[1].title()} · scene "
    f"{info['scene']} · patch {info['patch']} · {info['country']} ({info['continent']})"
    + (f" · <b style='color:#E6EAF2'>{sc['psnr']:.2f} dB</b> PSNR · SSIM {sc['ssim']:.3f}" if sc else "")
    + "</div>", unsafe_allow_html=True)

section("What went in, what came out")
err = np.abs(pred.astype(np.float32) - truth.astype(np.float32)).mean(-1) / 255.0
tiles = [(to_uint8(np.clip(x[0], 0, 1), vmax=1.0), "Radar VV"), (to_uint8(np.clip(x[1], 0, 1), vmax=1.0), "Radar VH"),
         (rgb_u8(x[2:]), "Cloudy input"), (truth, "True clear image"), (pred, lab.get(run, run)),
         (to_uint8(err, cmap="inferno", vmax=0.25), "Error vs truth")]
if f"gate_{run}" in s:
    tiles.append((to_uint8(s[f"gate_{run}"][i].astype(np.float32), cmap="viridis", vmax=1.0), "Gate: where it keeps the input"))
sig_runs = [r for r in runs if f"sigma_{r}" in s and np.any(s[f"sigma_{r}"][i])]
for r in sig_runs:
    tiles.append((to_uint8(s[f"sigma_{r}"][i].astype(np.float32), cmap="magma"), f"Uncertainty · {lab.get(r, r)}"))
cols = st.columns(4)
for k, (img, cap) in enumerate(tiles):
    cols[k % 4].image(img, caption=cap)

section("Every model on this patch")
rows = [dict(Model=lab.get(r, r), **{k.upper(): v for k, v in sc.items()}) for r, sc in info["scores"].items()]
if rows:
    df = pd.DataFrame(rows).sort_values("PSNR", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"PSNR": st.column_config.ProgressColumn(
                     "PSNR (dB)", format="%.2f", min_value=float(df.PSNR.min()) - 1, max_value=float(df.PSNR.max())),
                     "SSIM": st.column_config.NumberColumn(format="%.4f"), "SAM": st.column_config.NumberColumn(format="%.2f°"),
                     "MAE": st.column_config.NumberColumn(format="%.4f"), "RMSE": st.column_config.NumberColumn(format="%.4f")})
