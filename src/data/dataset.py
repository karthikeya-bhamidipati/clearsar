import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

def lee_filter(img, size=3):
    """Applies Lee Speckle Filter to 2D SAR numpy array."""
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_var = img_sqr_mean - img_mean**2
    overall_var = np.var(img)
    weights = img_var / (img_var + overall_var + 1e-5)
    return img_mean + weights * (img - img_mean)

class TargetedSEN12MSCR(Dataset):
    def __init__(self, input_roots, csv_path, target_countries=['CN', 'JP', 'IN'], max_patches=2000, apply_speckle=True):
        self.samples = []
        self.apply_speckle = apply_speckle
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing CSV at {csv_path}. Run scan-local first.")
            
        df = pd.read_csv(csv_path)
        df_filtered = df[df['country'].isin(target_countries)] if target_countries else df
        allowed_scenes = set(str(s) for s in df_filtered['scene'].tolist())

        expanded_roots = []
        for r in input_roots:
            expanded_roots.extend(glob.glob(r))

        stop_indexing = False
        for root in expanded_roots:
            if stop_indexing or not os.path.exists(root): 
                continue
            for dirpath, _, files in os.walk(root):
                if stop_indexing:
                    break
                folder_name = os.path.basename(dirpath)
                if folder_name.startswith('s1_'):
                    scene_id = folder_name.replace('s1_', '')
                    
                    if scene_id in allowed_scenes or len(allowed_scenes) == 0:
                        parent_dir = os.path.dirname(dirpath)
                        s2_c_dir = os.path.join(parent_dir, f"s2_cloudy_{scene_id}")
                        s2_gt_dir = os.path.join(parent_dir, f"s2_{scene_id}")
                        
                        for f in sorted(files):
                            if f.endswith('.tif'):
                                s1_path = os.path.join(dirpath, f)
                                s2_c_path = os.path.join(s2_c_dir, f.replace('s1_', 's2_cloudy_', 1))
                                s2_gt_path = os.path.join(s2_gt_dir, f.replace('s1_', 's2_', 1))
                                
                                if os.path.exists(s2_c_path) and os.path.exists(s2_gt_path):
                                    self.samples.append({
                                        's1': s1_path, 
                                        's2_cloudy': s2_c_path, 
                                        's2_clear': s2_gt_path
                                    })
                                    if len(self.samples) >= max_patches:
                                        stop_indexing = True
                                        break
        
        print(f"[Dataset] Indexed {len(self.samples)} patch triplets (Countries: {target_countries}).")

    def __len__(self): 
        return len(self.samples)

    def _norm_s1(self, img):
        img = np.nan_to_num(img, nan=0.0)
        if self.apply_speckle:
            filtered = np.zeros_like(img)
            for c in range(img.shape[0]):
                filtered[c] = lee_filter(img[c], size=3)
            img = filtered
            
        db = 10.0 * np.log10(np.clip(img, 1e-5, None))
        clipped = np.clip(db, -30.0, 0.0)
        return (clipped - (-15.0)) / 15.0

    def _norm_s2(self, img):
        img = np.nan_to_num(img, nan=0.0)
        clipped = np.clip(img / 10000.0, 0.0, 1.0)
        return (clipped * 2.0) - 1.0

    def __getitem__(self, idx):
        item = self.samples[idx]
        with rasterio.open(item['s1']) as src: 
            s1 = src.read()[:2]
        with rasterio.open(item['s2_cloudy']) as src: 
            s2_c = src.read()[:4]
        with rasterio.open(item['s2_clear']) as src: 
            s2_gt = src.read()[:3]
            
        x = np.concatenate([self._norm_s1(s1), self._norm_s2(s2_c)], axis=0)
        y = self._norm_s2(s2_gt)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)