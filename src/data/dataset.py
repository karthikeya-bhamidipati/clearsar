"""
SEN12MS-CR data pipeline (docs/project_plan.md §2):

    index_triplets()        find every (S1, S2-cloudy, S2-clear) patch triplet by filename
    attach_scene_metadata() join country / continent from data/scans/scenes.csv
    make_splits()           split BY SCENE (no spatial leakage) + an out-of-continent test set
    build_cache()           read + preprocess every selected patch ONCE into a .npy memmap
    CachedPatchDataset      fast random 128x128 crops + flips/rot90 for training,
                            full 256x256 patches for evaluation

Preprocessing (standard for SEN12MS-CR):
    S1  : values are already in dB. Clip VV to [-25, 0], VH to [-32.5, 0], rescale to [0, 1].
    S2  : clip to [0, 10000], rescale to [0, 1]. All 13 bands kept.
    Lee : optional speckle filter (ablation only), applied in the LINEAR power domain.

Model input  = [VV, VH, 13 cloudy S2 bands]  -> 15 channels
Model target = 13 clear S2 bands             -> 13 channels
"""
import glob
import hashlib
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import DataLoader, Dataset

PATCH_RE = re.compile(r"(ROIs\d+_[a-z]+)_(s1|s2_cloudy|s2)_(\d+)_p(\d+)\.tif$")
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
RGB_IDX = (3, 2, 1)                       # B4, B3, B2 inside the 13-band S2 stack
S1_DB_RANGE = ((-25.0, 0.0), (-32.5, 0.0))  # VV, VH
N_S1, N_S2 = 2, 13
IN_CHANNELS, OUT_CHANNELS = N_S1 + N_S2, N_S2
PATCH_SIZE = 256
RGB_GAIN = 3.0                            # display stretch for reflectance in [0, 1]


# --------------------------------------------------------------------------------------
# 1. Indexing
# --------------------------------------------------------------------------------------
def index_triplets(roots, log=print):
    """Walk every root (glob patterns allowed) and pair files purely by filename:
    ROIs1158_spring_s1_12_p345.tif <-> ..._s2_12_p345.tif <-> ..._s2_cloudy_12_p345.tif.
    Duplicates across mirror datasets collapse onto the same key.
    Returns one row per complete triplet: season, scene, patch, s1, s2, s2_cloudy."""
    if isinstance(roots, str):
        roots = [roots]
    expanded = sorted({p for r in roots for p in glob.glob(r)})
    if not expanded:
        raise FileNotFoundError(f"No input directories match {roots}")

    files = defaultdict(dict)
    n_tif = 0
    for root in expanded:
        for dirpath, _, filenames in os.walk(root, followlinks=True):
            for fname in filenames:
                if not fname.endswith(".tif"):
                    continue
                n_tif += 1
                m = PATCH_RE.search(fname)
                if m:
                    season, modality, scene, patch = m.groups()
                    files[(season, int(scene), int(patch))][modality] = os.path.join(dirpath, fname)

    rows = [dict(season=k[0], scene=k[1], patch=k[2], **v)
            for k, v in files.items() if {"s1", "s2", "s2_cloudy"} <= v.keys()]
    df = pd.DataFrame(rows, columns=["season", "scene", "patch", "s1", "s2", "s2_cloudy"])
    df = df.sort_values(["season", "scene", "patch"]).reset_index(drop=True)
    incomplete = len(files) - len(df)
    log(f"[index] {n_tif} .tif files under {len(expanded)} root(s) -> {len(files)} patch keys, "
        f"{len(df)} complete triplets ({incomplete} incomplete skipped), "
        f"{df.groupby(['season', 'scene']).ngroups if len(df) else 0} scenes")
    if df.empty:
        raise RuntimeError("Found 0 complete S1/S2/S2-cloudy triplets. Are the SEN12MS-CR "
                           "datasets attached as inputs? Expected names like "
                           "ROIs1158_spring_s1_1_p30.tif")
    return df


def attach_scene_metadata(df, scenes_csv, log=print):
    """Add lon/lat/country/continent per (season, scene) from the scene census."""
    meta = pd.read_csv(scenes_csv)[["season", "scene", "lon", "lat", "country", "continent"]]
    meta["scene"] = meta["scene"].astype(int)
    out = df.merge(meta, on=["season", "scene"], how="left")
    missing = out["continent"].isna()
    if missing.any():
        n = out.loc[missing, ["season", "scene"]].drop_duplicates().shape[0]
        log(f"[meta] {n} scene(s) not in {os.path.basename(scenes_csv)} -> continent 'Unknown' (excluded)")
    out["continent"] = out["continent"].fillna("Unknown")
    out["country"] = out["country"].fillna("??")
    return out


# --------------------------------------------------------------------------------------
# 2. Splitting
# --------------------------------------------------------------------------------------
def _sample_evenly(df, cap, rng):
    """Take up to `cap` patches spread evenly across the scenes in df."""
    if cap is None or len(df) <= cap:
        return df
    groups = [g for _, g in df.groupby(["season", "scene"])]
    quota = int(np.ceil(cap / len(groups)))
    parts = [g.sample(n=min(len(g), quota), random_state=int(rng.integers(1 << 31))) for g in groups]
    out = pd.concat(parts)
    if len(out) > cap:
        out = out.sample(n=cap, random_state=int(rng.integers(1 << 31)))
    return out


def make_splits(df, train_region="Asia", val_frac=0.2, test_frac=0.2,
                ood_continents=("Europe", "Africa", "North America", "South America"),
                force_test_countries=("IN",), caps=None, seed=42, log=print):
    """Scene-level split (docs/project_plan.md §2.3):
      - scenes in `train_region` -> train / val / test_id   (~60/20/20 of the scenes)
      - one scene from each country in `force_test_countries` goes to test_id
        (so there is an India-only slice to report)
      - one random scene from each of `ood_continents` -> ood (domain-shift test only)
    Then each split is capped at caps[split] patches, sampled evenly across its scenes.
    Returns (patch_df with a 'split' column, scene_df with one row per used scene)."""
    caps = caps or {}
    rng = np.random.default_rng(seed)
    scenes = df.groupby(["season", "scene"]).agg(
        n_patches=("patch", "size"), country=("country", "first"), continent=("continent", "first"),
        lon=("lon", "first"), lat=("lat", "first")).reset_index()

    region = scenes[scenes.continent.str.lower() == train_region.lower()]
    if len(region) < 3:
        raise RuntimeError(f"Need >= 3 scenes in {train_region}, found {len(region)}")
    order = list(rng.permutation(len(region)))
    region = region.iloc[order].reset_index(drop=True)

    assign = {}
    forced = []
    for cc in force_test_countries or ():
        cand = region[(region.country == cc)]
        if len(cand):
            r = cand.iloc[0]
            forced.append((r.season, r.scene))
    n = len(region)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    remaining = [(r.season, r.scene) for r in region.itertuples() if (r.season, r.scene) not in forced]
    test_keys = forced + remaining[: max(0, n_test - len(forced))]
    remaining = [k for k in remaining if k not in test_keys]
    val_keys, train_keys = remaining[:n_val], remaining[n_val:]
    for k in train_keys:
        assign[k] = "train"
    for k in val_keys:
        assign[k] = "val"
    for k in test_keys:
        assign[k] = "test_id"

    for cont in ood_continents or ():
        cand = scenes[scenes.continent == cont]
        if len(cand):
            r = cand.iloc[int(rng.integers(len(cand)))]
            assign[(r.season, r.scene)] = "ood"

    scenes["split"] = [assign.get((r.season, r.scene)) for r in scenes.itertuples()]
    scenes = scenes[scenes.split.notna()].reset_index(drop=True)

    df = df.merge(scenes[["season", "scene", "split"]], on=["season", "scene"], how="inner")
    parts = [_sample_evenly(df[df.split == s], caps.get(s), rng) for s in ("train", "val", "test_id", "ood")]
    out = pd.concat(parts).sort_values(["split", "season", "scene", "patch"]).reset_index(drop=True)
    used = out.groupby(["season", "scene"]).size().rename("n_used").reset_index()
    scenes = scenes.merge(used, on=["season", "scene"], how="left").fillna({"n_used": 0})

    for s in ("train", "val", "test_id", "ood"):
        sc = scenes[scenes.split == s]
        log(f"[split] {s:8s}: {len(sc):2d} scenes, {int((out.split == s).sum()):5d} patches  "
            f"countries={sorted(sc.country.unique().tolist())}")
    return out, scenes


# --------------------------------------------------------------------------------------
# 3. Preprocessing
# --------------------------------------------------------------------------------------
def lee_filter(img, size=3):
    """Simple Lee speckle filter on a 2D LINEAR-intensity array."""
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img ** 2, (size, size))
    img_var = np.maximum(img_sqr_mean - img_mean ** 2, 0)
    overall_var = np.var(img)
    weights = img_var / (img_var + overall_var + 1e-12)
    return img_mean + weights * (img - img_mean)


def preprocess_s1(s1, apply_lee=False):
    """[2, H, W] S1 backscatter -> [2, H, W] float32 in [0, 1].
    SEN12MS-CR ships S1 in dB (mostly negative values). If the array looks linear
    (no negative values), it is converted to dB first."""
    s1 = s1.astype(np.float32)
    finite = s1[np.isfinite(s1)]
    if finite.size and finite.min() >= 0:                    # linear power -> dB
        s1 = 10.0 * np.log10(np.clip(s1, 1e-6, None))
    out = np.empty_like(s1)
    for c, (lo, hi) in enumerate(S1_DB_RANGE):
        band = np.nan_to_num(s1[c], nan=lo, posinf=hi, neginf=lo)
        if apply_lee:
            lin = lee_filter(10.0 ** (band / 10.0))
            band = 10.0 * np.log10(np.clip(lin, 1e-6, None))
        out[c] = (np.clip(band, lo, hi) - lo) / (hi - lo)
    return out


def preprocess_s2(s2):
    """[13, H, W] S2 digital numbers -> [13, H, W] float32 reflectance in [0, 1]."""
    s2 = np.nan_to_num(s2.astype(np.float32), nan=0.0, posinf=10000.0, neginf=0.0)
    return np.clip(s2, 0.0, 10000.0) / 10000.0


def cloud_fraction(cloudy, clear, threshold=0.08):
    """Cheap cloud-cover proxy used ONLY for binning results (never as a model input):
    a pixel counts as cloudy if the cloudy image is brighter than the clear target by more
    than `threshold` reflectance, averaged over B2/B3/B4. Clouds add reflectance; this
    misses cloud shadows, which is acceptable for coarse 20%-wide bins."""
    vis = [1, 2, 3]
    diff = cloudy[vis].mean(axis=0) - clear[vis].mean(axis=0)
    return float((diff > threshold).mean())


def to_rgb(bands13, gain=RGB_GAIN):
    """[13, H, W] reflectance -> [H, W, 3] display RGB in [0, 1]."""
    rgb = np.stack([bands13[i] for i in RGB_IDX], axis=-1)
    return np.clip(rgb * gain, 0.0, 1.0)


def read_triplet(paths, apply_lee=False):
    """Read and preprocess one triplet. Returns x [15, H, W], y [13, H, W] float32."""
    import rasterio
    with rasterio.open(paths["s1"]) as src:
        s1 = src.read()[:N_S1]
    with rasterio.open(paths["s2_cloudy"]) as src:
        s2c = src.read()
    with rasterio.open(paths["s2"]) as src:
        s2 = src.read()
    if s1.shape[0] < N_S1 or s2c.shape[0] < N_S2 or s2.shape[0] < N_S2:
        raise ValueError(f"Unexpected band counts S1={s1.shape[0]} S2c={s2c.shape[0]} S2={s2.shape[0]}")
    x = np.concatenate([preprocess_s1(s1, apply_lee), preprocess_s2(s2c[:N_S2])], axis=0)
    return x, preprocess_s2(s2[:N_S2])


# --------------------------------------------------------------------------------------
# 4. One-off cache (.npy memmaps)
# --------------------------------------------------------------------------------------
class _TripletReader(Dataset):
    def __init__(self, records, apply_lee):
        self.records, self.apply_lee = records, apply_lee

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        try:
            x, y = read_triplet(self.records[i], self.apply_lee)
            if x.shape[1:] != (PATCH_SIZE, PATCH_SIZE) or y.shape[1:] != (PATCH_SIZE, PATCH_SIZE):
                raise ValueError(f"patch shape {x.shape[1:]}")
            return i, x.astype(np.float16), y.astype(np.float16), True, ""
        except Exception as e:  # corrupt / truncated file: keep going, mark invalid
            z = np.zeros((IN_CHANNELS, PATCH_SIZE, PATCH_SIZE), np.float16)
            return i, z, z[:OUT_CHANNELS], False, f"{self.records[i]['s1']}: {e}"


def _collate(batch):
    return batch


def build_cache(split_df, cache_dir, name, apply_lee=False, num_workers=4, log=print):
    """Preprocess every patch of one split once and store it as float16 .npy memmaps
    (x: [N, 15, 256, 256], y: [N, 13, 256, 256]). Re-uses an existing cache when the
    patch list and preprocessing settings are unchanged. Returns a cache-info dict and the
    split dataframe with 'cache_idx', 'valid' and 'cloud_frac' columns."""
    os.makedirs(cache_dir, exist_ok=True)
    df = split_df.reset_index(drop=True).copy()
    key = hashlib.md5(("|".join(df.s1.tolist()) + f"|lee={apply_lee}").encode()).hexdigest()[:12]
    xp, yp = os.path.join(cache_dir, f"{name}_x.npy"), os.path.join(cache_dir, f"{name}_y.npy")
    mp = os.path.join(cache_dir, f"{name}_meta.json")
    info = dict(name=name, x=xp, y=yp, n=len(df))

    if os.path.exists(mp):
        with open(mp) as f:
            meta = json.load(f)
        if meta.get("key") == key and os.path.exists(xp) and os.path.exists(yp):
            df["valid"], df["cloud_frac"] = meta["valid"], meta["cloud_frac"]
            df["cache_idx"] = np.arange(len(df))
            log(f"[cache] {name}: re-using {len(df)} cached patches")
            return info, df

    X = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float16,
                                  shape=(len(df), IN_CHANNELS, PATCH_SIZE, PATCH_SIZE))
    Y = np.lib.format.open_memmap(yp, mode="w+", dtype=np.float16,
                                  shape=(len(df), OUT_CHANNELS, PATCH_SIZE, PATCH_SIZE))
    records = df[["s1", "s2", "s2_cloudy"]].to_dict("records")
    loader = DataLoader(_TripletReader(records, apply_lee), batch_size=16, num_workers=num_workers,
                        collate_fn=_collate, shuffle=False)
    valid = np.zeros(len(df), bool)
    cloud = np.zeros(len(df), np.float32)
    errors = []
    done = 0
    for batch in loader:
        for i, x, y, ok, err in batch:
            X[i], Y[i], valid[i] = x, y, ok
            if ok:
                cloud[i] = cloud_fraction(x[N_S1:].astype(np.float32), y.astype(np.float32))
            else:
                errors.append(err)
        done += len(batch)
        if done % 1000 < len(batch) or done == len(df):
            log(f"[cache] {name}: {done}/{len(df)} patches")
    X.flush()
    Y.flush()
    del X, Y
    for e in errors[:5]:
        log(f"[cache] WARNING unreadable: {e}")
    if errors:
        log(f"[cache] {name}: {len(errors)} unreadable patch(es) excluded")
    with open(mp, "w") as f:
        json.dump(dict(key=key, valid=valid.tolist(), cloud_frac=cloud.tolist()), f)
    df["valid"], df["cloud_frac"], df["cache_idx"] = valid, cloud, np.arange(len(df))
    return info, df


class CachedPatchDataset(Dataset):
    """Reads from build_cache() memmaps. Opened lazily so it is safe with DataLoader workers.

    crop    : int -> random crop of that size (training); None -> full 256x256 patch
    augment : random horizontal/vertical flips and 90-degree rotations (training only)
    Returns (x [15, h, w], y [13, h, w], cache_idx)."""

    def __init__(self, cache_info, indices, crop=None, augment=False):
        self.xp, self.yp = cache_info["x"], cache_info["y"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.crop, self.augment = crop, augment
        self._X = self._Y = None

    def __len__(self):
        return len(self.indices)

    def _open(self):
        if self._X is None:
            self._X = np.load(self.xp, mmap_mode="r")
            self._Y = np.load(self.yp, mmap_mode="r")

    def __getitem__(self, k):
        self._open()
        i = int(self.indices[k])
        if self.crop:
            c = self.crop
            r0 = int(torch.randint(0, PATCH_SIZE - c + 1, (1,)))
            c0 = int(torch.randint(0, PATCH_SIZE - c + 1, (1,)))
            x = np.array(self._X[i, :, r0:r0 + c, c0:c0 + c], dtype=np.float32)
            y = np.array(self._Y[i, :, r0:r0 + c, c0:c0 + c], dtype=np.float32)
        else:
            x = np.array(self._X[i], dtype=np.float32)
            y = np.array(self._Y[i], dtype=np.float32)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        if self.augment:
            if torch.rand(1) < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if torch.rand(1) < 0.5:
                x, y = x.flip(-2), y.flip(-2)
            k90 = int(torch.randint(0, 4, (1,)))
            if k90:
                x, y = torch.rot90(x, k90, (-2, -1)), torch.rot90(y, k90, (-2, -1))
        return x.contiguous(), y.contiguous(), i
