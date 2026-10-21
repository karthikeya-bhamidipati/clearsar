import torch
import torch.nn.functional as F
import numpy as np
import scipy.linalg
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

def compute_fid_score(pred_tensor, target_tensor):
    """
    Spatially downsamples features using adaptive pooling to 8x8 (192 features)
    to calculate FID distribution distance without exceeding system RAM.
    """
    p_pooled = F.adaptive_avg_pool2d(pred_tensor.detach().cpu(), (8, 8))
    t_pooled = F.adaptive_avg_pool2d(target_tensor.detach().cpu(), (8, 8))
    
    p_flat = p_pooled.view(p_pooled.size(0), -1).numpy()
    t_flat = t_pooled.view(t_pooled.size(0), -1).numpy()
    
    mu1, sigma1 = np.mean(p_flat, axis=0), np.cov(p_flat, rowvar=False)
    mu2, sigma2 = np.mean(t_flat, axis=0), np.cov(t_flat, rowvar=False)
    
    diff = mu1 - mu2
    eps = np.eye(sigma1.shape[0]) * 1e-6
    covmean = scipy.linalg.sqrtm((sigma1 + eps).dot(sigma2 + eps))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
        
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(np.clip(fid, 0.0, 1000.0))