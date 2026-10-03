#!/usr/bin/env python3
"""
find_region_scenes.py - find (and extract) SEN12MS-CR scenes inside one continent
                         or one country.

SEN12MS-CR is organised by season and numbered scene, not by place. This script
streams the archives straight from the TUM server (nothing large is written to disk
during the scan), reads the georeference of one patch per scene, and looks up its
country, continent and state.

Install:
    pip install rasterio reverse_geocoder pandas

Step 1 - scan (streams only the Sentinel-1 archives, the smallest ones):
    python find_region_scenes.py scan --out scenes.csv --continent Asia

Step 2 - extract only the scenes you want. Streams the S1, S2 and S2-cloudy archives
for the seasons that contain your region and keeps only the matching files:
    python find_region_scenes.py extract --csv scenes.csv --continent Asia --out data/

    Filter to one country instead:
        python find_region_scenes.py extract --csv scenes.csv --country IN --out data/

    Optional: add out-of-region test scenes (free in bandwidth, since you're already
    streaming those seasons), e.g. --extra ROIs1158_spring:42 ROIs1868_summer:7

Notes
- Country/continent come from the nearest populated place (reverse_geocoder), so a
  scene right on a border can be mislabelled. Spot-check borderline ones on a map
  using the lon/lat columns in scenes.csv.
- If the connection drops mid-scan or mid-extract, just re-run: finished steps are
  cached (scan) or marked done (extract) and are skipped.
- Run this inside a Kaggle CPU notebook with internet enabled, so the large full-
  archive streams use Kaggle's bandwidth instead of yours. Save the extracted
  subset as a Kaggle dataset afterwards.
"""
import argparse
import os
import re
import sys
import tarfile
import urllib.request
from collections import defaultdict

from continents import to_continent

BASE = "ftp://m1554803:m1554803@dataserv.ub.tum.de/"
SEASONS = ["ROIs1158_spring", "ROIs1868_summer", "ROIs1970_fall", "ROIs2017_winter"]
MODALITIES = ["s1", "s2", "s2_cloudy"]
# e.g. ROIs1158_spring_s2_cloudy_12_p345.tif
PATCH_RE = re.compile(r"(ROIs\d+_[a-z]+)_(s1|s2_cloudy|s2)_(\d+)_p(\d+)\.tif$")


def stream_tar(season, modality):
    url = f"{BASE}{season}_{modality}.tar.gz"
    print(f"  streaming {url}", flush=True)
    resp = urllib.request.urlopen(url, timeout=120)
    return resp, tarfile.open(fileobj=resp, mode="r|gz")  # "r|" = sequential stream


def parse(name):
    m = PATCH_RE.search(os.path.basename(name))
    if not m:
        return None
    season, modality, scene, patch = m.groups()
    return season, modality, int(scene), int(patch)


def centre_lonlat(tif_bytes):
    from rasterio.io import MemoryFile
    from rasterio.warp import transform

    with MemoryFile(tif_bytes) as mem, mem.open() as ds:
        cx = (ds.bounds.left + ds.bounds.right) / 2
        cy = (ds.bounds.bottom + ds.bounds.top) / 2
        lon, lat = transform(ds.crs, "EPSG:4326", [cx], [cy])
    return lon[0], lat[0]


def scan(args):
    import pandas as pd
    import reverse_geocoder as rg

    frames = []
    for season in SEASONS:
        cache = f"{args.out}.{season}.csv"
        if os.path.exists(cache):
            print(f"{season}: using cached {cache}")
            frames.append(pd.read_csv(cache))
            continue
        print(f"{season}:")
        counts, centres = defaultdict(int), {}
        resp, tar = stream_tar(season, "s1")
        with resp, tar:
            for member in tar:
                if not member.isfile():
                    continue
                info = parse(member.name)
                if info is None:
                    continue
                scene = info[2]
                counts[scene] += 1
                if scene not in centres:
                    centres[scene] = centre_lonlat(tar.extractfile(member).read())
        if not counts:
            sys.exit("No files matched the expected naming pattern. "
                     "Print a few tar member names and adjust PATCH_RE.")
        scenes = sorted(counts)
        places = rg.search([(centres[s][1], centres[s][0]) for s in scenes], mode=1)
        df = pd.DataFrame({
            "season": season,
            "scene": scenes,
            "n_patches": [counts[s] for s in scenes],
            "lon": [round(centres[s][0], 4) for s in scenes],
            "lat": [round(centres[s][1], 4) for s in scenes],
            "country": [p["cc"] for p in places],
            "continent": [to_continent(p["cc"]) for p in places],
            "state": [p["admin1"] for p in places],
            "nearest_place": [p["name"] for p in places],
        })
        df.to_csv(cache, index=False)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} scenes to {args.out}\n")

    print("By continent:")
    print(df.groupby("continent")
            .agg(scenes=("scene", "count"), patches=("n_patches", "sum"))
            .sort_values("patches", ascending=False)
            .to_string())

    print("\nBy country (top 20 by patch count):")
    print(df.groupby("country")
            .agg(scenes=("scene", "count"), patches=("n_patches", "sum"))
            .sort_values("patches", ascending=False)
            .head(20)
            .to_string())

    if args.continent:
        sub = df[df.continent.str.lower() == args.continent.lower()]
        print(f"\nScenes in continent={args.continent}: {len(sub)} scenes, "
              f"{int(sub.n_patches.sum())} patches")
        print(sub.to_string(index=False) if len(sub) else "  none")
    if args.country:
        sub = df[df.country == args.country.upper()]
        print(f"\nScenes in country={args.country.upper()}: {len(sub)} scenes, "
              f"{int(sub.n_patches.sum())} patches")
        print(sub.to_string(index=False) if len(sub) else "  none")


def extract(args):
    import pandas as pd

    df = pd.read_csv(args.csv)
    if args.continent:
        sel = df[df.continent.str.lower() == args.continent.lower()]
    elif args.country:
        sel = df[df.country == args.country.upper()]
    else:
        sys.exit("Pass --continent or --country.")
    if args.state:
        sel = sel[sel.state.str.lower() == args.state.lower()]
    if args.max_scenes:
        sel = sel.sort_values("n_patches", ascending=False).head(args.max_scenes)

    wanted = {(r.season, int(r.scene)) for r in sel.itertuples()}
    for item in args.extra or []:
        season, scene = item.split(":")
        wanted.add((season, int(scene)))
    if not wanted:
        sys.exit("No scenes match. Check the scan output / your filters.")
    print(f"Extracting {len(wanted)} scenes: {sorted(wanted)}")

    os.makedirs(args.out, exist_ok=True)
    for season in sorted({s for s, _ in wanted}):
        for modality in MODALITIES:
            done = os.path.join(args.out, f".done_{season}_{modality}")
            if os.path.exists(done):
                print(f"{season} {modality}: already done")
                continue
            kept = 0
            resp, tar = stream_tar(season, modality)
            with resp, tar:
                for member in tar:
                    if not member.isfile():
                        continue
                    info = parse(member.name)
                    if (info is None or info[1] != modality
                            or (info[0], info[2]) not in wanted):
                        continue
                    dest = os.path.join(args.out, season, f"{modality}_{info[2]}",
                                        os.path.basename(member.name))
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "wb") as f:
                        f.write(tar.extractfile(member).read())
                    kept += 1
            open(done, "w").close()
            print(f"    kept {kept} files")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan", help="locate every scene (streams S1 archives only)")
    s.add_argument("--out", default="scenes.csv")
    s.add_argument("--continent", default="Asia",
                   help="continent to summarise, e.g. Asia, Europe, Africa")
    s.add_argument("--country", default=None, help="ISO-2 code to also summarise, e.g. IN")
    s.set_defaults(func=scan)

    e = sub.add_parser("extract", help="extract only the scenes of one region")
    e.add_argument("--csv", default="scenes.csv")
    e.add_argument("--continent", default=None, help="e.g. Asia")
    e.add_argument("--country", default=None, help="ISO-2 code, e.g. IN")
    e.add_argument("--state", default=None, help="optional: one state/province only")
    e.add_argument("--max-scenes", type=int, default=None,
                   help="cap to the N scenes with the most patches (keeps download/disk small)")
    e.add_argument("--extra", nargs="*", help="extra scenes as SEASON:SCENE")
    e.add_argument("--out", default="data")
    e.set_defaults(func=extract)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
