import streamlit as st

from frontend.common import card, hero, load, section

root, data = load()
hero("How it works", "From radar and clouds <span class='cs-grad'>to a decision</span>",
     "ClearSAR is two projects in one: a controlled deep-learning comparison of three architectures for "
     "SAR-to-optical cloud removal, and an explainable expert system that turns their outputs into decisions.")

section("1 · Data", "SEN12MS-CR: co-registered Sentinel-1 radar, a cloudy Sentinel-2 image and a clear Sentinel-2 "
        "image of the same 2.56 km patch.")
c = st.columns(3, gap="medium")
card(c[0], "Inputs (15 channels)", "Radar VV + VH (in dB, clipped to [−25, 0] / [−32.5, 0]) and all 13 bands of "
     "the cloudy optical image (reflectance in [0, 1]).")
card(c[1], "Target (13 channels)", "The clear optical image taken close in time - the answer the models must "
     "reproduce. It is never shown to the expert system.")
card(c[2], "Split by scene", "Asian scenes go to train / validation / test; neighbouring patches never cross "
     "splits. One scene from each other continent is used only for testing domain shift.")

section("2 · Deep learning", "Three architectures, one training loop, one budget.")
c = st.columns(3, gap="medium")
card(c[0], "Training", "Random 128×128 crops with flips and rotations · L1 loss · AdamW with warm-up and cosine decay "
     "· mixed precision · output initialised at the mean reflectance · best checkpoint by validation L1.")
card(c[1], "Evaluation", "Every 256×256 test patch is cut into four tiles so all models see the size they were "
     "trained on. PSNR, SSIM, SAM, MAE, RMSE on all 13 bands, FID on RGB.")
card(c[2], "Experiments", "Learning curves, leakage check, U-Net / Swin / ViT ablations, GAN loss, three kinds of "
     "uncertainty and repeated seeds - each changing exactly one thing.")

section("3 · Expert system", "Decision support on top of the models - see the Advisor and Knowledge base pages.")
c = st.columns(3, gap="medium")
card(c[0], "Facts", "Cloud cover, cirrus, radar gaps, open water (from the input) · model disagreement, error on "
     "visible pixels, uncertainty, radar reliance (from the models) · best model per cloud level and domain-shift "
     "drop (from the experiments).")
card(c[1], "Reasoning", "IF-THEN rules with certainty factors, chained group by group: observations → trust → model "
     "→ suitability → action. The knowledge base is a plain YAML file.")
card(c[2], "Explanation", "Every decision lists the rules that produced it, the alternatives that were considered, "
     "why other rules did not fire, and visual evidence (cloud mask, radar-reliance and uncertainty maps, band saliency).")

section("Limitations")
st.markdown("""
- Trained on a 3,000-patch subset of Asian scenes with a fixed GPU budget; scores are not comparable to published
  SEN12MS-CR benchmarks, which use different, global test sets.
- The cloud mask is a brightness/whiteness heuristic (no s2cloudless); bright deserts or snow can be mistaken for cloud.
- FID is computed on a few hundred images and is noisy - compare models with it, don't read its absolute value.
- The expert system's thresholds and certainty factors are expert judgements encoded in the knowledge base; they can be
  tuned against the experiment results.
""")
if data is not None and data["config"]:
    cfg = data["config"]
    with st.expander("Settings of the loaded run"):
        st.json({k: cfg[k] for k in ("steps_per_run", "slots", "minutes_total", "status") if k in cfg})
