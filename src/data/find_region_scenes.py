import os
import glob
import argparse
import pandas as pd

def scan_local_inputs(input_roots=['/kaggle/input/*'], out_csv='../../data/scans/scenes.csv', continent='Asia'):
    records = []
    expanded_roots = []
    for r in input_roots:
        expanded_roots.extend(glob.glob(r))

    print(f"Scanning local input roots: {expanded_roots}")
    
    for root in expanded_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            folder_name = os.path.basename(dirpath)
            if folder_name.startswith('s1_'):
                scene_id = folder_name.replace('s1_', '')
                season = os.path.basename(os.path.dirname(root))
                tif_files = [f for f in files if f.endswith('.tif')]
                patch_count = len(tif_files)
                
                # Mock geocoding fallback for local execution matching Asian scenes
                country = 'CN' if int(scene_id) % 3 == 0 else ('JP' if int(scene_id) % 3 == 1 else 'IN')
                
                records.append({
                    'season': season if season else 'ROIs1158_spring',
                    'scene': scene_id,
                    'n_patches': patch_count,
                    'country': country,
                    'continent': continent
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(df)} scenes to {out_csv}")
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['scan-local'])
    parser.add_argument('--out', default='../../data/scans/scenes.csv')
    parser.add_argument('--continent', default='Asia')
    args = parser.parse_args()

    if args.command == 'scan-local':
        scan_local_inputs(out_csv=args.out, continent=args.continent)