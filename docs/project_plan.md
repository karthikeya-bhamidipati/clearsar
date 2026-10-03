# SAR-to-Optical Cloud Removal: Asia Subset — Final Project Plan (v3)

**Goal:** a complete, hands-on understanding of deep learning, through one well-controlled comparison of three architectures. Publishing a paper is not a goal.

**Locked decisions**

- **Data:** SEN12MS-CR, only the scenes that fall inside Asia (which includes India/South Asia). Optionally spotlight the Indian scenes specifically in the write-up.
- **Models** (same data, same protocol, only the backbone changes):
  - **A.** Convolutional U-Net, trained with L1 loss. A Pix2Pix GAN version is an ablation.
  - **B.** U-Net with a Swin-Transformer bottleneck (hybrid CNN + windowed attention).
  - **C.** Pure ViT encoder-decoder (no convolutions).
- **Uncertainty:** on Model A only.
- **ViViT:** dropped.
- **Compute:** Kaggle T4 with mixed precision (AMP) for all training. RTX 3050 (4 GB) for debugging and visualisation only.

---

## 1. The Asia data reality — read this first

SEN12MS-CR has **169 scenes** spread around the globe. Each is roughly 52 × 40 km and gives about 700 patches of 256 × 256 px. No official list says which country or continent each scene is in — but the published *parent* dataset, SEN12MS (252 scenes, of which SEN12MS-CR kept 169), gives a rough continent breakdown: Asia had **59 scenes** across all seasons, the largest or second-largest continent group. After SEN12MS-CR's cleanup, expect something like **35–45 Asian scenes**, i.e. **roughly 25,000–30,000 patches**. That is an estimate, not a fact — day 1 replaces it with the real number.

- **Day 1 task:** run `find_region_scenes.py scan --continent Asia`. It streams only the Sentinel-1 archives and prints every scene's country, continent, state, location and patch count.
- **You will have far more data than you need.** Don't train on all of it — cap the training set at a manageable size (see §2.1) so downloads, disk, and per-run training time stay small. More scenes just means you get to *choose* a good, diverse subset rather than being stuck with whatever a single country offers.

**What generous data means for you:**

- **The comparison is fairer to the ViT.** Its whole weakness is needing lots of data; a bigger, more diverse pool gives it a genuine chance, which makes the learning-curve experiment (E1) more informative.
- **You can still spotlight India.** Within your Asia subset, tag which patches are Indian (the scan output already gives you `country` per scene) and report an India-specific breakdown in your results, even though training used the wider Asian set.
- **Domain shift is now partly built in for free.** Asia alone spans monsoon tropics, deserts, and Siberia — training on a diverse Asian subset and testing on a held-out, *different* climate zone within Asia is itself a mini domain-shift experiment, on top of the separate out-of-continent test described in §2.1.
- **Your scores still won't be comparable to published SEN12MS-CR numbers.** Those use different, global test data. Compare your three models against each other.

---

## 2. Data pipeline (Week 1)

### 2.1 Get the data
1. `python find_region_scenes.py scan --out scenes.csv --continent Asia`
2. Look at the printed per-country breakdown inside Asia. Pick a training subset capped at a sane size with `--max-scenes` — e.g. the **15–20 scenes with the most patches** (roughly 10,000–15,000 patches total), spread across as many seasons as you can, to keep download/disk/training time manageable. Note which of these scenes are Indian, for the India-specific breakdown later.
3. Pick **2 out-of-continent scenes** from the CSV (e.g. one from Europe, one from Africa or the Americas), used **only** for testing domain shift. These cost extra bandwidth since they're outside Asia — keep it to 2.
4. Extract the scenes:
   `python find_region_scenes.py extract --csv scenes.csv --continent Asia --max-scenes 18 --extra ROIs1868_summer:7 ROIs1970_fall:23 --out data/`
5. **Tip:** run steps 1 and 4 in a Kaggle *CPU* notebook with internet enabled. The S2 archives are large and must be streamed in full even though you keep only a small part, so it's better to use Kaggle's bandwidth than yours. Save `data/` as a private Kaggle dataset.

### 2.2 Preprocessing (standard for SEN12MS-CR, so it matches the literature)
- **Sentinel-1:** the values are *already in dB*, so don't convert again. Clip VV to [−25, 0] and VH to [−32.5, 0], then rescale each to [0, 1].
- **Sentinel-2 (cloudy and clear):** clip to [0, 10000] and rescale to [0, 1]. Keep all 13 bands.
- **No speckle filter in the main runs.** The published baselines don't use one. (Optional extra: a refined Lee filter, as one ablation.)
- **Cloud masks:** compute an s2cloudless mask on every cloudy image. You need it to report results by cloud-cover bin (0–20%, …, 80–100%).
- **Storage:** convert the GeoTIFFs once into NumPy shards (int16/uint16). Also save each patch's map coordinates, which the spatial split needs. ~3,000 triplets is roughly 11 GB, small enough to upload easily.

### 2.3 Splits — where most student projects quietly cheat
Neighbouring patches look almost identical (**spatial autocorrelation**). A random patch-level split leaks test information into training and inflates every score.
- **By scene (≥5 scenes):** about 60/20/20 of the scenes go to train/val/test, mixing seasons where possible.
- **Spatial blocks (fewer scenes):** within each scene, sort patches north to south. The top ~70% goes to train and the next ~15% to val, then drop one row of patches (a 2.56 km buffer). The last ~15% goes to test.
- **Out-of-region set:** the 2 foreign scenes, used for testing only.
- **Learning check:** once, train with a *random* patch split as well and compare the test score. The inflated number shows you what leakage looks like.

### 2.4 Baselines before any deep learning
- **Identity:** output = the cloudy input. This is surprisingly strong on lightly clouded patches.
- **Tiny CNN:** about 3 conv layers.

If a big model can't clearly beat these, something is wrong.

---

## 3. The three models (sized for small data)

All models take **15 input channels** (VV, VH and 13 cloudy bands, concatenated) and output **13 bands in [0, 1]** through a sigmoid.

| | Model A — U-Net | Model B — Swin-bottleneck U-Net | Model C — Pure ViT |
|---|---|---|---|
| Encoder | 4 conv down-blocks, base width 32 (32→64→128→256) | Same as A | Patch embedding, patch 8 (128² crop → 16×16 = 256 tokens), dim 192 |
| Middle | Conv bottleneck | 4 Swin blocks at 16×16 tokens, dim 256, window 4, shifted windows | 8 transformer layers, 3 heads |
| Decoder | Conv up-blocks + skip connections | Same as A | 2 transformer layers → linear "unpatchify" to 8×8×13 per token |
| Rough size | ~8M params | ~10M | ~5M |
| Inductive bias | Strong (locality, translation equivariance) | Mixed | Weak |

- **Model A (GAN version):** add a 70×70 PatchGAN discriminator. Loss = L1 + λ·adversarial, with λ = 100.
- **Parameter counts:** they differ. Report them, and treat "is it the architecture or just the size?" as a question you discuss in your write-up.
- **Evaluation:** cut every 256×256 test patch into four 128×128 tiles for *all* models. This keeps the comparison identical and avoids resizing the ViT's positional embeddings.

---

## 4. Fixed protocol (identical for every model)

| Item | Value |
|---|---|
| Training data | Random 128×128 crops, with random flips and 90° rotations |
| Loss | L1 (the GAN loss only in the A ablation) |
| Optimiser | AdamW, weight decay 0.01. For GAN runs, β1 = 0.5 |
| Learning rate | Pick per model with an LR range test (§6), then keep it fixed |
| Schedule | 500-step warmup, then cosine decay |
| Budget | 15,000 steps × batch 16 (≈100 epochs on ~2.5k patches); keep the checkpoint with the best validation L1 |
| Precision | AMP (autocast + GradScaler) |
| Seeds | 2 per main model; report the mean and the spread |
| Metrics | PSNR, SSIM, SAM, MAE, RMSE: overall, by cloud bin, Asia test vs. out-of-continent test, plus an India-only slice of the test set |
| Also log | Parameter count, training time, peak GPU memory |

**Rule of thumb:** with 2 seeds, treat differences under ~0.3 dB PSNR as noise.

---

## 5. The learning experiments (the heart of the project)

For each experiment, **write your prediction and the reason for it in your notes before you run it.** After the run, explain why you were right or wrong. A wrong prediction you can explain is worth more than a good score.

| # | Experiment | What to predict | DL lesson |
|---|---|---|---|
| E1 | **Learning curves:** train A, B and C on 25%, 50% and 100% of the training data | A is best at 25%; C improves fastest as data grows but stays behind | Inductive bias vs. data hunger. **The single most important experiment** |
| E2 | **Domain shift:** evaluate every model on the out-of-region scenes | All models drop; which drops most? | Generalisation, distribution shift |
| E3 | **U-Net probes:** (a) remove the skip connections; (b) L2 instead of L1; (c) visualise the effective receptive field | (a) blurry, lost fine detail; (b) blurrier than L1; (c) the effective field is much smaller than the theoretical one | Why skips work; what loss functions really optimise |
| E4 | **GAN ablation:** λ = ∞ (L1 only), 100, 10 | Sharper images, *lower* PSNR, better FID | Adversarial training; the perception–distortion trade-off |
| E5 | **Swin probes:** (a) shifting on vs. off; (b) window 4 vs. 8 vs. 16 (16 = full global attention at the bottleneck); (c) plot the attention maps over cloudy pixels | (a) Without shifting, information can't cross window borders; (b) does global attention help with large clouds? | Local vs. global context; the cost of attention |
| E6 | **ViT probes:** (a) remove the positional embeddings; (b) patch size 8 vs. 16 | (a) The model loses spatial layout because attention ignores token order; (b) patch 16 gives visible blocky artefacts | Permutation invariance; tokenisation trade-offs |
| E7 | **Uncertainty on A:** (a) variance head trained with Gaussian NLL; (b) MC dropout (20 passes); (c) ensemble of your 3 seeds | Uncertainty is highest under thick cloud. MC dropout and the ensemble should report *more* uncertainty on the out-of-region scenes; the variance head may not | Aleatoric vs. epistemic uncertainty; calibration |

**Evaluating uncertainty (E7):**
- **Calibration error (UCE):** do the predicted variances match the actual errors?
- **Sparsification curve:** does throwing away the most-uncertain pixels lower the remaining error?
- **Coverage:** do about 95% of true values fall within ±1.96σ?
- **Domain shift:** does uncertainty rise on the out-of-region scenes?

**Rough compute** (estimate; measure in week 1): about 25–30 runs at under 1 T4-hour each, so **~20–35 GPU-hours in total**. That is roughly one week of Kaggle quota.

---

## 6. Habits that build real understanding (do these every time)

1. **Overfit 16 samples first.** If the loss doesn't go to nearly zero, there's a bug. Do this on the RTX 3050.
2. **LR range test:** raise the learning rate exponentially over ~300 steps and plot loss against LR. Pick a value about 10× below where the loss blows up.
3. **Log per-layer gradient norms.** Vanishing or exploding gradients show up here before they show up in the loss.
4. **Save image grids every epoch** (cloudy input | SAR | prediction | target). Your eyes catch bugs that metrics miss.
5. **Write the core pieces yourself in plain PyTorch:** the training loop, U-Net, attention, Swin block and ViT. Check each against a reference only *after* it works. Example: write attention in ~20 lines and confirm it matches `torch.nn.functional.scaled_dot_product_attention` to about 1e-5.
6. **Checkpoint everything** every 15–20 minutes (model, optimiser, scaler, scheduler, step, random states), so a dropped Kaggle session costs you nothing.

---

## 7. Week-by-week schedule (8 weeks)

| Week | Build | Experiments | Milestone |
|---|---|---|---|
| 1 | Scan, apply the decision rule, extract; shards; splits; preprocessing; metrics with unit tests; identity + tiny-CNN baselines; time a T4 run | Leakage check (random vs. spatial split) | Baseline table |
| 2 | U-Net from scratch; training loop with AMP and resume; overfit test; LR range test | Model A seed 1; E3 (a, b) | A trained |
| 3 | PatchGAN + GAN loss; gradient-norm logging | E4; E3 (c) | GAN ablation done |
| 4 | Attention from scratch (matched to PyTorch); window partition, shift mask (unit-tested), relative position bias; Swin bottleneck | Model B; E5 | B trained, attention maps |
| 5 | Patch embedding, ViT encoder-decoder, unpatchify | Model C; E6 | C trained, artefact analysis |
| 6 | Run seeds in the background; meanwhile build the variance head, MC dropout and calibration metrics | E1 (all models × 3 data fractions); second seeds | Learning-curve plot |
| 7 | Full evaluation by cloud bin; out-of-region evaluation | E2; E7 | Final results table + uncertainty plots |
| 8 | Write-up (prediction → result → explanation for every experiment); failure gallery | Buffer | Report done |

**Minimum version (if you only have ~5 weeks):**

| Week | Work |
|---|---|
| 1 | As above |
| 2 | Model A + E3 (a) |
| 3 | Model B + GAN ablation (λ = 100 only) |
| 4 | Model C + E1 with 2 data fractions |
| 5 | E2 + variance head + write-up |

Use 1 seed throughout.

**If you fall behind, cut in this order:** FID → E5 (b, c) → E6 (b) → the ensemble in E7 → second seeds (keep them for A) → E3 (c).

**Never cut:** the baselines, the overfit test, the spatial split, the cloud-bin breakdown, E1, E2.

---

## 8. Write-up structure

1. **Problem and physics:** why SAR sees through clouds and why translating it to optical is ambiguous.
2. **Data:** the Asian scenes used, with the India-only subset highlighted on a map from `scenes.csv`, plus splits and the leakage check.
3. **Models:** one diagram each, and its inductive bias in one sentence.
4. **Results:** the main table (PSNR, SSIM, SAM, MAE, RMSE; Asia test vs. out-of-continent; India-only slice; by cloud bin).
5. **Experiments E1–E7:** each with *prediction → result → explanation*.
6. **Uncertainty:** maps, calibration plot, sparsification curve.
7. **Limitations:** small data, one country, different test set from published work, the GAN's tendency to invent detail.
8. **What I learned:** the most valuable section for your goal.

---

## 9. Reading list (one or two per week, matched to what you're building)

- **Wk 1–2:**
  - Karpathy, *A Recipe for Training Neural Networks* (blog, 2019)
  - Ronneberger et al., *U-Net* (2015)
  - Meraner et al., *DSen2-CR* (2020)
  - Ebel et al., *SEN12MS-CR* (2021)
- **Wk 3:**
  - Isola et al., *Pix2Pix* (2017)
  - Blau & Michaeli, *The Perception-Distortion Tradeoff* (2018)
- **Wk 4:**
  - Vaswani et al., *Attention Is All You Need* (2017)
  - Liu et al., *Swin Transformer* (2021)
- **Wk 5:** Dosovitskiy et al., *An Image is Worth 16x16 Words* (ViT, 2021)
- **Wk 6–7:**
  - Kendall & Gal, *What Uncertainties Do We Need…?* (2017)
  - Gal & Ghahramani, *Dropout as a Bayesian Approximation* (2016)
  - Lakshminarayanan et al., *Deep Ensembles* (2017)
  - Ebel et al., *UnCRtainTS* (2023)
- **Any time:** Luo et al., *Understanding the Effective Receptive Field* (2016)

---

## 10. New terms introduced in this version

Everything else is in the earlier glossary.

- **Domain shift (distribution shift):** test data comes from a different distribution than training data, e.g. a different country. Models usually get worse, often without warning.
- **Spatial autocorrelation:** nearby places look alike. In geospatial machine learning, random splits put near-duplicates in both train and test.
- **Spatial block split / buffer:** splitting data by contiguous geographic areas, with an unused gap between them, so train and test are truly separate.
- **Data leakage:** information from the test set influencing training or model selection, which makes scores look better than they really are.
- **Learning curve:** performance plotted against training-set size. It shows how much a model depends on data.
- **Ablation:** removing or changing one component to measure what it contributes.
- **Pre-registered hypothesis:** writing down your prediction *before* an experiment, so you can't explain away the result afterwards.
- **Effective receptive field:** the input region that *actually* influences an output pixel, measured from gradients. It is usually much smaller and more centre-weighted than the theoretical receptive field.
- **LR range test:** a short run with an exponentially rising learning rate, used to find a sensible learning rate.
- **Gradient norm:** the size of a layer's gradient vector. A healthy value is roughly steady; values near zero mean vanishing gradients, and spikes mean instability.
- **Permutation invariance:** shuffling the input tokens doesn't change the output (apart from shuffling it too). Self-attention without positional encoding is permutation-invariant.
- **Tiling evaluation:** cutting test images into fixed-size tiles so every model sees exactly the same input size.
- **Best-checkpoint selection (early stopping):** keeping the weights with the lowest *validation* loss rather than the final weights, to limit overfitting.
- **Seed variance:** the spread in results from runs that differ only in their random seed. It is the noise floor for comparing models.
- **Parameter matching:** making compared models roughly equal in size, so differences come from the design rather than capacity.
