import numpy as np
import pandas as pd
import streamlit as st

from frontend.common import (FINAL_LABELS, compare_slider, final_sample_names, hero, labels, load, load_final,
                             no_results, rgb_u8, sample_names, section, to_uint8)

root, data = load()
fin = load_final()
hero("Explore patches", "Look at the <span class='cs-grad'>reconstructions</span> yourself",
     "Held-out test patches from every continent, chosen across the whole range of cloud cover - per cloud level the "
     "patches closest to the median score, not the best ones. Your eyes catch what metrics miss: blur, invented "
     "texture, wrong colours.")
sources = (["Final model (18 test scenes)"] if fin is not None and fin["samples"] is not None else []) +           (["Phase 1 (Asian test scenes)"] if data is not None and data["samples"] is not None else [])
if not sources:
    no_results()
src = st.segmented_control("Gallery", sources, default=sources[0], key="ex_src") or sources[0]
if src.startswith("Final"):
    s, meta, lab = fin["samples"], fin["samples_meta"], dict(FINAL_LABELS)
    names = final_sample_names(meta)
else:
    s, meta, lab = data["samples"], data["samples_meta"], labels(data)
    names = sample_names(meta)
runs = [r for r in meta["runs"] if f"pred_{r}" in s and r != "identity"]
if "final" in runs:
    runs = ["final"] + [r for r in runs if r != "final"]

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
    tiles.append((to_uint8(s[f"gate_{run}"][i].astype(np.float32), cmap="viridis", vmax=1.0), "Cloud gate: bright = input kept"))
sig_runs = [r for r in runs if f"sigma_{r}" in s and np.any(s[f"sigma_{r}"][i])]
for r in sig_runs:
    tiles.append((to_uint8(s[f"sigma_{r}"][i].astype(np.float32), cmap="magma"),
                  f"Uncertainty · {lab.get(r, r)}" + (" (flip test)" if r == "final" else "")))
cols = st.columns(4)
for k, (img, cap) in enumerate(tiles):
    cols[k % 4].image(img, caption=cap)

section("Every model on this patch")
NICE = {"psnr": "PSNR", "ssim": "SSIM", "sam": "SAM", "mae": "MAE", "rmse": "RMSE", "fid": "FID",
        "sharpness": "Sharpness", "psnr_aligned": "PSNR, date shift removed"}
rows = [dict(Model=lab.get(r, r), **{NICE.get(k, k): v for k, v in sc.items()}) for r, sc in info["scores"].items()]
if rows:
    df = pd.DataFrame(rows).sort_values("PSNR", ascending=False)
    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"PSNR": st.column_config.ProgressColumn(
                     "PSNR (dB)", format="%.2f", min_value=float(df.PSNR.min()) - 1, max_value=float(df.PSNR.max())),
                     "SSIM": st.column_config.NumberColumn(format="%.3f"), "SAM": st.column_config.NumberColumn("SAM (°)", format="%.2f"),
                     "MAE": st.column_config.NumberColumn(format="%.4f"), "RMSE": st.column_config.NumberColumn(format="%.4f"),
                     "Sharpness": st.column_config.NumberColumn(format="%.2f"),
                     "PSNR, date shift removed": st.column_config.NumberColumn("PSNR, date shift removed (dB)", format="%.2f")})
