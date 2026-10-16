import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

def compute_metrics(pred_tensor, target_tensor):
    pred = ((pred_tensor + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    target = ((target_tensor + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    
    psnr_list, ssim_list, sam_list = [], [], []
    
    for i in range(pred.shape[0]):
        p_img = pred[i].transpose(1, 2, 0)
        t_img = target[i].transpose(1, 2, 0)
        
        # PSNR
        mse = np.mean((p_img - t_img) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100.0
        psnr_list.append(psnr)
        
        # SSIM
        ssim_val = ssim_fn(p_img, t_img, channel_axis=2, data_range=1.0)
        ssim_list.append(ssim_val)
        
        # Spectral Angle Mapper (SAM)
        dot_product = np.sum(p_img * t_img, axis=2)
        p_norm = np.linalg.norm(p_img, axis=2)
        t_norm = np.linalg.norm(t_img, axis=2)
        denom = p_norm * t_norm
        denom[denom == 0] = 1e-5
        cos_sam = np.clip(dot_product / denom, -1.0, 1.0)
        sam_deg = np.mean(np.arccos(cos_sam) * (180.0 / np.pi))
        sam_list.append(sam_deg)
        
    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), float(np.mean(sam_list))