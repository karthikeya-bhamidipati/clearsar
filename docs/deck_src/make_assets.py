"""Images and chart data for the presentation, all from outputs/final (real test results)."""
import json
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

ROOT = r"C:\Users\karth\Desktop\clearsar"
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402

from src.data.dataset import to_rgb  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(OUT, exist_ok=True)
z = np.load(os.path.join(ROOT, "outputs", "final", "results", "samples.npz"))
meta = json.loads(str(z["meta"]))["samples"]
u8 = lambda a: (np.clip(a, 0, 1) * 255).astype(np.uint8)
idx = {m["country"] + str(int(round(m["cloud_frac"] * 100))): i for i, m in enumerate(meta)}
print(sorted(idx))


def save(arr, name, size=512):
    Image.fromarray(arr).resize((size, size), Image.LANCZOS).save(os.path.join(OUT, name))


def tiles(i, tag):
    x, y = z["x"][i].astype(np.float32), z["y"][i].astype(np.float32)
    save(u8(to_rgb(x[2:])), f"{tag}_cloudy.png")
    save(u8(to_rgb(y)), f"{tag}_truth.png")
    save(z["pred_final"][i], f"{tag}_final.png")
    save(z["pred_before_gan"][i], f"{tag}_before.png")
    vv = np.clip(x[0], 0, 1)
    save(u8(np.repeat(vv[..., None], 3, -1)), f"{tag}_radar.png")
    g = matplotlib.colormaps["viridis"](z["gate_final"][i].astype(np.float32))[..., :3]
    save(u8(g), f"{tag}_gate.png")


for tag in ("IN28", "CN2", "FR31", "AR49", "PT81", "TM99"):
    tiles(idx[tag], tag)

# zoomed crop to show what the GAN changed (same region, before vs after)
for tag, (r0, c0) in (("PT81", (96, 40)), ("CN2", (64, 64))):
    i = idx[tag]
    for k in ("pred_before_gan", "pred_final"):
        crop = z[k][i][r0:r0 + 112, c0:c0 + 112]
        Image.fromarray(crop).resize((448, 448), Image.NEAREST).save(os.path.join(OUT, f"{tag}_zoom_{k.split('_', 1)[1]}.png"))
    crop = u8(to_rgb(z["y"][i].astype(np.float32)))[r0:r0 + 112, c0:c0 + 112]
    Image.fromarray(crop).resize((448, 448), Image.NEAREST).save(os.path.join(OUT, f"{tag}_zoom_truth.png"))

# chart data
res = os.path.join(ROOT, "outputs", "final")
cl = pd.read_csv(os.path.join(res, "results", "test_by_cloud.csv"))
order = ["0-10%", "10-30%", "30-60%", "60-90%", "90-100%"]
cloud = {r: [round(float(cl[(cl.run == r) & (cl.cloud_bin == b)].psnr.iloc[0]), 2) for b in order]
         for r in ("identity", "p1_unet", "before_gan", "final")}
va = pd.read_csv(os.path.join(res, "logs", "val_history.csv"))
va = va[va.phase.isin(["A", "B"])]
tr = pd.read_csv(os.path.join(res, "logs", "train_history.csv"))
b = tr[tr.phase == "B"].iloc[::4]
summ = pd.read_csv(os.path.join(res, "results", "test_summary.csv")).set_index("run")
json.dump(dict(
    cloud_bins=order, cloud=cloud,
    val_h=[round(float(v), 2) for v in va.time_h], val_psnr=[round(float(v), 2) for v in va.psnr],
    val_sharp=[round(float(v), 3) for v in va.sharpness],
    gan_h=[round(float(v), 2) for v in b.time_h], d_real=[round(float(v), 3) for v in b.d_real],
    d_fake=[round(float(v), 3) for v in b.d_fake],
    summary={r: {k: round(float(summ.loc[r, k]), 4) for k in ("psnr", "ssim", "sam", "psnr_aligned", "sharpness")}
             for r in summ.index}), open(os.path.join(OUT, "data.json"), "w"), indent=1)
print("assets in", OUT)
