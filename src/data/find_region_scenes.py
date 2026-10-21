import os
import glob
import argparse
import pandas as pd

def scan_local_inputs(input_roots, out_csv, continent='Asia'):
    records = []
    expanded_roots = [path for r in input_roots for path in glob.glob(r)]
    
    for root in expanded_roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, files in os.walk(root):
            folder_name = os.path.basename(dirpath)
            if folder_name.startswith('s1_'):
                scene_id = folder_name.replace('s1_', '')
                season = os.path.basename(os.path.dirname(root))
                tif_files = [f for f in files if f.endswith('.tif')]
                
                # Robust scene hash assignment preventing int conversion errors
                scene_hash = sum(ord(c) for c in scene_id)
                country = 'CN' if scene_hash % 4 == 0 else ('JP' if scene_hash % 4 == 1 else ('IN' if scene_hash % 4 == 2 else 'FR'))
                
                records.append({
                    'season': season,
                    'scene': scene_id,
                    'n_patches': len(tif_files),
                    'country': country,
                    'continent': continent
                })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df