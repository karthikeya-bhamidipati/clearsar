import streamlit as st

from frontend.common import card, hero, load_final, section

fin = load_final()
hero("How it works", "From radar and clouds <span class='cs-grad'>to a decision</span>",
     "ClearSAR is two projects in one: a deep-learning model that rebuilds cloud-free Sentinel-2 images from "
     "Sentinel-1 radar plus the cloudy image, and an explainable expert system that decides how far each "
     "reconstruction can be trusted.")

section("1 · Data", "SEN12MS-CR: co-registered Sentinel-1 radar, a cloudy Sentinel-2 image and a clear Sentinel-2 "
        "image of the same 2.56 km patch - 122 218 patches from 175 scenes on every continent, four seasons.")
c = st.columns(3, gap="medium")
card(c[0], "Input (15 channels)", "Radar VV + VH (in dB, clipped to [−25, 0] / [−32.5, 0], scaled to [0, 1]) and "
     "all 13 bands of the cloudy optical image (reflectance = value / 10 000).")
card(c[1], "Target (13 channels)", "The clear optical image of the same place, taken on a different day. Because "
     "of that date difference even a cloud-free input only scores ~26-27 dB against it - the ceiling of a plain "
     "pixel match. 'PSNR, date shift removed' measures past it.")
card(c[2], "Split by scene", "147 training scenes (all 102 871 patches), 10 validation scenes (600 patches), 18 "
     "test scenes (up to 300 patches each, 5 400 in total). Neighbouring patches never cross splits, so the test "
     "measures generalisation to new places.")

section("2 · The model: ClearSAR-Net + PatchGAN", "One generator, trained in two parts on Kaggle's 2 × T4 GPUs.")
c = st.columns(3, gap="medium")
card(c[0], "Generator", "U-Net encoder-decoder (3 levels, width 64) with skip connections · 2 global self-attention "
     "blocks on the 32×32 bottleneck (every region sees the whole patch) · bilinear upsampling (no checkerboard "
     "artefacts) · <b>hybrid output</b>: a learned gate m mixes 'keep and colour-correct the input' with 'generate "
     "from radar': out = m·corrected + (1 − m)·generated. 12.8 M parameters.")
card(c[1], "Part A · reconstruction (55 %)", "Loss L1 + 0.05·(1 − SSIM) - weights chosen by measuring both "
     "gradients on real patches (SSIM's is 14× L1's) · AdamW, lr 3e-4, warm-up then cosine · mixed precision · "
     "EMA of the weights (0.999) · best validation L1 → <i>before GAN</i>.")
card(c[2], "Part B · PatchGAN (45 %)", "Conditional 70×70 PatchGAN with spectral normalisation, least-squares loss · "
     "the adversarial weight adapts so its gradient is 20 % of the pixel loss's · generator lr 5e-5, discriminator "
     "2e-4 · final model chosen by L1 × (1 + |1 − sharpness|), i.e. accurate <i>and</i> sharp.")
c = st.columns(3, gap="medium")
card(c[0], "Efficiency", "Both GPUs train the same model (DistributedDataParallel, fp16 gradient exchange) · batch "
     "size probed to fit the T4 memory · patches read straight from the GeoTIFFs by 8 processes × 3 threads · "
     "when the disk is slower than the GPUs, recent patches are re-used with a new flip/rotation (data echoing).")
card(c[1], "Robustness", "Time-based schedule: always finishes inside Kaggle's 12 h · NaN guard: snapshots every 1 000 "
     "good steps and rolls back automatically if training goes non-finite · the attention residual stream runs in "
     "fp32 (the first full run overflowed fp16 at step ~5 000 - the reason for this fix).")
card(c[2], "Evaluation", "Once, on the 18 test scenes: the cloudy input, the Phase 1 U-Net, before-GAN and final · "
     "PSNR, SSIM, SAM, MAE on all 13 bands, sharpness, colour / detail error, date-shift-free PSNR · by cloud "
     "cover, continent and country · uncertainty from 4-way flip test-time augmentation.")

section("3 · Expert system", "Decision support on top of the model - see the Advisor and Knowledge base pages.")
c = st.columns(3, gap="medium")
card(c[0], "Facts", "Cloud cover, cirrus, radar gaps, open water (from the input) · disagreement of independent "
     "models, error on visible pixels, uncertainty, radar reliance (from the models) · the most accurate model per "
     "cloud level (from the test results).")
card(c[1], "Reasoning", "IF-THEN rules with certainty factors, chained group by group: observations → trust → model "
     "→ suitability → action. The knowledge base is a plain YAML file.")
card(c[2], "Explanation", "Every decision lists the rules that produced it, the alternatives that were considered, "
     "why other rules did not fire, and visual evidence (cloud mask, radar-reliance and uncertainty maps, band saliency).")

section("4 · How we got here", "Phase 1 - kept under Earlier experiments - compared 26 trainings (U-Net, Swin U-Net, "
        "ViT, ablations, GAN, uncertainty) on 3 000 Asian patches.")
st.markdown("""
- No architecture was clearly best (26.3-27.0 dB); global context (ViT) helped most under thick cloud - hence the
  attention bottleneck.
- The biggest limit was data: 3 000 patches are 2.5 % of SEN12MS-CR - hence training on everything.
- The Phase 1 GAN lost 1.4 dB because its checkpoint was chosen by L1 after 1-4 k steps - hence the two-part training
  and the sharpness-aware selection.
""")

section("Limitations")
st.markdown("""
- The clear target is from another date, so PSNR has a ceiling; read 'PSNR, date shift removed' next to it.
- A GAN can add plausible but invented detail - the advisor recommends the before-GAN model for quantitative uses.
- The cloud mask of the expert system is a brightness/whiteness heuristic; bright deserts or snow can look like cloud.
- The uncertainty scale is calibrated on the test patches (median spread = median error) - fine for a demo, a real
  deployment would calibrate on separate data.
""")
if fin is not None and fin["config"]:
    with st.expander("Settings of the loaded final run"):
        st.json(fin["config"].get("cfg", {}))
