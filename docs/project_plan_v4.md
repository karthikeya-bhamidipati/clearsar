# ClearSAR — Plan v4: one model, built properly

**Why a new plan.** The first phase (plan v3) compared 26 trainings of three architecture families. It worked
technically and taught a lot, but no single model produces reconstructions that *look* close to the true clear
image. Measured on the test gallery (2026-10-22):

| | Sharpness (1.0 = as sharp as the truth) | Colour error (16×16-blurred) | Detail error |
|---|---|---|---|
| U-Net | 0.50 | 0.069 | 0.027 |
| ViT | 0.91 (partly block artefacts) | 0.074 | 0.027 |
| GAN | 1.05 | 0.082 | 0.031 |

Two problems: **blur** (the U-Net is half as sharp as reality) and, larger, **wrong colour/brightness at coarse
scale** (colour error ≈ 2.5 × detail error). The first phase also told us why and where to look:

* scene diversity limits generalisation more than patch count or architecture (E1 flat; two test scenes
  decided the ranking; only 23 training scenes were used out of 175);
* context from far away matters under thick cloud (the ViT, receptive field 56 px, won at 60–100 % cloud);
* the models re-generate every pixel instead of keeping what the satellite actually saw;
* the L1 loss and L1-based checkpoint selection favour smooth, cautious outputs.

**Goal of v4.** Build **one** model, step by step, until its reconstructions are as close to the truth as the
data allows — and understand *why* each step helps. Not a publication: a complete, hands-on understanding of
deep learning. Phase 1 (plan v3) becomes **Stage 0: exploration** in the write-up.

**Decisions (2026-10-22).** Time: more than 3 weeks. Data: **train on all continents** (~120 scenes), test on
held-out scenes from every continent, Asia and India reported separately.

---

## How every stage works

1. **Concept** — read the short explanation (and the paper section, if any) before touching code.
2. **Prediction** — write down what you expect and why, *before* running anything (in `docs/lab_notes.md`).
3. **Build** — change exactly one thing. Parts marked ✍ are yours to write; Claude reviews and tests them.
4. **Run** — one Kaggle run (30–60 min), same evaluation every time.
5. **Look** — at the images first, then at the numbers.
6. **Explain** — was the prediction right? Why / why not? One paragraph in the lab notes.

Every stage reports the same six numbers on the validation scenes: **PSNR, SSIM, SAM, sharpness ratio,
colour error, detail error** — plus the same 8 gallery images. The test scenes are used **once**, in stage 7.

---

## Stage 0 — Understand the data and what is achievable  (week 1)
**Concept.** Irreducible error (Bayes error): some of the difference between input and target cannot be
learned, because the information is not in the input. The cloudy and clear images are taken on *different
dates*: fields change, sun angle and atmosphere differ. Under thick cloud, the colours are simply not observed.
**Build.**
* A data-study notebook: view ~50 triplets across continents and cloud levels; plot cloud-cover distribution.
* The **ceiling**: on cloud-free pixels, how different is the cloudy image from the clear one? (Phase 1 already
  suggests ~26 dB for nearly cloud-free patches.) This is the best any model can do on visible ground without
  learning date differences.
* Add the three "eye" metrics to `src/evaluation/metrics.py`: sharpness ratio, colour error, detail error. ✍
* The new global split: ~120 training scenes, ~20 validation, ~25 test, stratified by continent; ≥ 1 Indian
  and several Asian scenes in test.
**Prediction to write.** What PSNR do you expect the perfect model to reach on 0–20 % vs 80–100 % cloud?
**Done when.** You can state a realistic target per cloud level and explain where the error comes from.

## Stage 1 — Training mechanics on a plain U-Net  (week 1)
**Concept.** What a training step does (forward, loss, backward, optimiser update); learning rate; batch size;
initialisation; train vs validation curves; underfitting vs overfitting.
**Build.**
* Overfit one batch of 8 patches until the loss is ~0 (proves the pipeline can learn). ✍ the loop yourself.
* LR range test: raise the learning rate exponentially over 300 steps, plot loss vs LR, pick ~10× below the
  blow-up point.
* Train the plain U-Net on the new global data; read the curves.
**Prediction to write.** Will the global U-Net beat Phase 1's 26.3 dB on Asian validation scenes? By how much?
**Done when.** You can read a train/val curve and say whether a model under- or over-fits, and why.

## Stage 2 — Keep what the satellite saw: residual pass-through  (week 2)
**Concept.** Residual learning (He et al., 2016; DSen2-CR, Meraner et al., 2020): predicting a *correction* to
the input is easier than predicting the image from scratch. A learned soft cloud mask decides where to trust
the input and where to fill in.
**Build.** `output = cloudy + correction` and/or `output = m · cloudy + (1 − m) · generated`, with the mask m
predicted by the network. ✍ the gating module.
**Prediction to write.** Which of the six numbers improves most? On which cloud levels?
**Done when.** Visible areas are reproduced almost exactly and sharpness rises.

## Stage 3 — Data beats architecture: the scene-count learning curve  (week 2)
**Concept.** Generalisation depends on how well the training distribution covers the test distribution.
More *different places* teach more than more patches of the same place.
**Build.** Train the stage-2 model on 25 %, 50 % and 100 % of the training **scenes** (same total steps).
**Prediction to write.** Does the colour error fall with more scenes? Does PSNR rise linearly or saturate?
**Done when.** You have the curve and can explain it with the Phase-1 E1 result (flat in patches).

## Stage 4 — What the loss optimises: towards sharp *and* correct  (week 3)
**Concept.** L1 is minimised by the median of plausible answers → smooth. SSIM rewards structure; a perceptual
loss (VGG features, Johnson et al., 2016) rewards texture; an adversarial loss rewards realism but may invent
detail. The perception–distortion trade-off (Blau & Michaeli, 2018). Checkpoint selection must match the goal.
**Build.** L1 → L1 + SSIM → + perceptual → + small adversarial term (λ swept). ✍ the SSIM/perceptual loss.
Select checkpoints by a combined score, not L1 alone.
**Prediction to write.** Which loss raises sharpness towards 1.0 without lowering PSNR by more than 0.5 dB?
**Done when.** You can show the trade-off on one plot (sharpness vs PSNR) and pick a point on purpose.

## Stage 5 — Context: global attention where clouds are large  (week 3)
**Concept.** Effective receptive field; self-attention; why the ViT won under thick cloud in Phase 1.
**Build.** A global self-attention bottleneck at 16 × 16 in the stage-4 model (cheap at that resolution). ✍ the
attention block (≈ 20 lines), checked against `torch.nn.functional.scaled_dot_product_attention`.
**Prediction to write.** Gains only under 60–100 % cloud, or everywhere?
**Done when.** The cloud-bin plot shows where attention helps.

## Stage 6 — Train it properly  (week 4)
**Concept.** Scaling, regularisation, EMA (exponential moving average of weights), full-resolution training.
**Build.** Full 256 × 256 inputs, longer schedule, EMA, 2–3 seeds of the final configuration.
**Done when.** Validation improvements have flattened and seed spread is known.

## Stage 7 — One honest final test, and the expert system on top  (week 4)
**Build.** Evaluate once on the held-out test scenes (overall, per continent, Asia, India, per cloud level).
Compare against the Phase-1 models and the published range for single-image SEN12MS-CR methods (~27–29 dB
PSNR, check the exact table in Ebel et al. 2021). Plug the final model into the Advisor and re-run its
evaluation on all test patches (not only 8).
**Done when.** The final model, its numbers and its limitations are written up — with every stage's
prediction → result → explanation.

---

## What stays
The Kaggle pipeline, caching, metrics, frontend and expert system are reused. New runs use
`notebooks/kaggle_pipeline.py`-style scripts with a single job each, so every stage is one short run.

## Compute
Each stage: 1–3 runs of 30–60 min on a T4 → ~10–15 GPU-hours in total (Kaggle gives ~30 per week). The global
training cache (~10 000 patches ≈ 37 GB in `/tmp`) is rebuilt once per session (~25 min).

## Where things are recorded
* `docs/lab_notes.md` — your predictions and explanations, one section per stage (the core of the write-up).
* `docs/progress_log.md` — what was run, when, with which result.
* `docs/ClearSAR_Project_Guide.docx` — updated at the end with the v4 results.
