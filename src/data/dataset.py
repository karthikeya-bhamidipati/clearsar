import os
import torch
from torch.utils.data import Dataset
import numpy as np
import rasterio

class DirectSEN12MSCRDataset(Dataset):
    def __init__(self, input_roots, max_samples=None):
        self.samples = []
        for root in input_roots:
            if not os.path.exists(root): continue
            for dirpath, _, files in os.walk(root):
                if 's1_' in dirpath and os.path.basename(dirpath).startswith('s1_'):
                    s1_dir = dirpath
                    scene_folder = os.path.basename(s1_dir).replace('s1_', '')
                    parent_dir = os.path.dirname(s1_dir)
                    s2_cloudy_dir = os.path.join(parent_dir, f"s2_cloudy_{scene_folder}")
                    s2_clear_dir = os.path.join(parent_dir, f"s2_{scene_folder}")
                    
                    if os.path.exists(s2_cloudy_dir) and os.path.exists(s2_clear_dir):
                        for f in files:
                            if f.endswith('.tif'):
                                s1_path = os.path.join(s1_dir, f)
                                s2_c_path = os.path.join(s2_cloudy_dir, f.replace('s1_', 's2_cloudy_'))
                                s2_gt_path = os.path.join(s2_clear_dir, f.replace('s1_', 's2_'))
                                if os.path.exists(s2_c_path) and os.path.exists(s2_gt_path):
                                    self.samples.append({'s1': s1_path, 's2_cloudy': s2_c_path, 's2_clear': s2_gt_path})
                                    if max_samples and len(self.samples) >= max_samples: break
                if max_samples and len(self.samples) >= max_samples: break
        print(f"[Dataset] Indexed {len(self.samples)} patch triplets.")

    def __len__(self):
        return len(self.samples)

    def _norm_s1(self, img):
        img = np.nan_to_num(img, nan=0.0)
        db = 10.0 * np.log10(np.clip(img, 1e-5, None))
        return (np.clip(db, -30.0, 0.0) - (-15.0)) / 15.0

    def _norm_s2(self, img):
        img = np.nan_to_num(img, nan=0.0)
        return (np.clip(img / 10000.0, 0.0, 1.0) * 2.0) - 1.0

    def __getitem__(self, idx):
        item = self.samples[idx]
        with rasterio.open(item['s1']) as src: s1 = src.read()[:2]
        with rasterio.open(item['s2_cloudy']) as src: s2_c = src.read()[:4]
        with rasterio.open(item['s2_clear']) as src: s2_gt = src.read()[:3]
        x = np.concatenate([self._norm_s1(s1), self._norm_s2(s2_c)], axis=0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self._norm_s2(s2_gt), dtype=torch.float32)