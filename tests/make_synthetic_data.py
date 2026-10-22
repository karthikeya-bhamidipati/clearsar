"""
Write a tiny fake SEN12MS-CR folder (real naming convention, real scene ids from
data/scans/scenes.csv) so the whole Kaggle pipeline can be smoke-tested on a laptop:

    python tests/make_synthetic_data.py --out <dir> [--patches 6]

    CLEARSAR_INPUT_ROOTS=<dir> CLEARSAR_OUT_DIR=<out> CLEARSAR_CACHE_DIR=<cache> \
    CLEARSAR_RESUME_DIR=<resume> CLEARSAR_STEPS=40 CLEARSAR_VAL_EVERY=20 \
    CLEARSAR_CAPS=train=48,val=8,test_id=12,ood=12 python notebooks/kaggle_pipeline.py

S1 is written in dB (like the real dataset), S2 as uint16 digital numbers. The clear image
is a smooth function of a latent field that the SAR also sees, and the cloudy image adds
bright blobs, so a model can genuinely learn something. One file is deliberately corrupt
to exercise the "unreadable patch" path.
"""
import argparse
import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))


def pick_scenes(csv, n_extra=0):
    df = pd.read_csv(csv)
    asia = df[df.continent == "Asia"]
    chosen = [asia[asia.country == "IN"].iloc[0]]
    chosen += [r for _, r in asia[asia.country != "IN"].head(7).iterrows()]
    for cont in ("Europe", "Africa", "North America", "South America"):
        chosen.append(df[df.continent == cont].iloc[0])
    keys = [(r.season, int(r.scene)) for r in chosen]
    rest = df[[(r.season, int(r.scene)) not in set(keys) for r in df.itertuples()]]
    for _, r in rest.groupby("continent", group_keys=False).head(max(1, n_extra // 4)).head(n_extra).iterrows():
        keys.append((r.season, int(r.scene)))           # extra scenes, spread over continents (plan v4 tests)
    return keys


def write(path, arr, dtype):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype=dtype, crs="EPSG:32633", transform=from_origin(500000, 4000000, 10, 10)) as dst:
        dst.write(arr.astype(dtype))


def make_patch(rng, size=256):
    latent = gaussian_filter(rng.standard_normal((size, size)), 6)
    latent = (latent - latent.min()) / (np.ptp(latent) + 1e-6)              # [0, 1]
    vv = -20 + 14 * latent + rng.normal(0, 1.5, (size, size))              # dB, mostly negative
    vh = vv - 7 + rng.normal(0, 1.0, (size, size))
    vv[rng.random((size, size)) < 0.001] = np.nan                         # a few NaNs like real S1
    s1 = np.stack([vv, vh]).astype(np.float32)
    base = np.linspace(0.05, 0.3, 13)[:, None, None]
    clear = (base + 0.25 * latent[None]) * 10000
    cloud = gaussian_filter(rng.standard_normal((size, size)), 12)
    cover = rng.uniform(0.05, 0.95)
    mask = cloud > np.quantile(cloud, 1 - cover)
    cloudy = clear.copy()
    cloudy[:, mask] = 4000 + 3000 * rng.random()
    return s1, clear.clip(0, 10000), cloudy.clip(0, 10000)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--patches", type=int, default=6)
    p.add_argument("--extra-scenes", type=int, default=0, help="add more scenes across continents")
    p.add_argument("--csv", default=os.path.join(HERE, "..", "data", "scans", "scenes.csv"))
    a = p.parse_args()
    rng = np.random.default_rng(0)
    scenes = pick_scenes(a.csv, a.extra_scenes)
    n = 0
    for season, scene in scenes:
        for patch in range(1, a.patches + 1):
            s1, clear, cloudy = make_patch(rng)
            for mod, arr, dt in (("s1", s1, "float32"), ("s2", clear, "uint16"), ("s2_cloudy", cloudy, "uint16")):
                d = os.path.join(a.out, f"mirror_{season.split('_')[1]}", f"{season}_{mod}", f"{mod}_{scene}")
                write(os.path.join(d, f"{season}_{mod}_{scene}_p{patch}.tif"), arr, dt)
            n += 1
    # one corrupt file (must be skipped, not crash the run)
    season, scene = scenes[1]
    bad = os.path.join(a.out, f"mirror_{season.split('_')[1]}", f"{season}_s2", f"s2_{scene}",
                       f"{season}_s2_{scene}_p1.tif")
    with open(bad, "wb") as f:
        f.write(b"not a tiff")
    print(f"wrote {n} triplets for {len(scenes)} scenes to {a.out} (1 corrupted on purpose)")


if __name__ == "__main__":
    main()
