import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

def compute_metrics(pred_tensor, target_tensor):
    pred = ((pred_tensor + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    target = ((target_tensor + 1.0) / 2.0).clamp(0, 1).cpu().numpy()
    psnr_list, ssim_list, sam_list = [], [], []
    
    for i in range(pred.shape[0]):
        p_img, t_img = pred[i].transpose(1, 2, 0), target[i].transpose(1, 2, 0)
        mse = np.mean((p_img - t_img) ** 2)
        psnr_list.append(20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100.0)
        ssim_list.append(ssim_fn(p_img, t_img, channel_axis=2, data_range=1.0))
        
        dot_product = np.sum(p_img * t_img, axis=2)
        denom = np.linalg.norm(p_img, axis=2) * np.linalg.norm(t_img, axis=2)
        denom[denom == 0] = 1e-5
        sam_list.append(np.mean(np.arccos(np.clip(dot_product / denom, -1.0, 1.0)) * (180.0 / np.pi)))
        
    return float(np.mean(psnr_list)), float(np.mean(ssim_list)), float(np.mean(sam_list))