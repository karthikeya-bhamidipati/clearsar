# ClearSAR Advisor — an explainable expert system for cloud-removal decisions

The deep-learning part of ClearSAR reconstructs cloud-free Sentinel-2 images from Sentinel-1 radar and a
cloudy optical image. A reconstruction, however, is only an *estimate*: under thick cloud most of it is
inferred from radar. The **ClearSAR Advisor** is a rule-based expert system that sits on top of the
networks and answers the questions a user actually has:

1. **How far can I trust this reconstruction?**
2. **Which model should I use for it?**
3. **Is it fit for what I want to do with it?** (visual inspection, vegetation monitoring, land cover,
   change detection, flood mapping)
4. **What should I do next?** (accept · accept with an uncertainty mask · use radar-only products · wait
   for a clearer acquisition)

Every answer comes with a certainty factor and an explanation. Code: `src/expert/`; frontend: **Advisor**
and **Knowledge base** pages; tests: `tests/test_expert.py`.

---

## 1. Architecture

```
 patch (radar + cloudy optical) ──► deep-learning models (U-Net · Swin U-Net · ViT · variance head)
        │                                   │                        │
        │                                   ▼                        ▼
        │                        XAI: radar reliance,         predictions, uncertainty
        │                             band saliency                  │
        ▼                                   ▼                        ▼
   observation facts ──────────────►  FACT EXTRACTION + FUZZIFICATION  ◄── experience facts
                                              │                        (experiment results CSVs)
                                              ▼
            knowledge base (YAML) ──►  INFERENCE ENGINE  (forward chaining, certainty factors)
                                              │
                         ┌────────────────────┴───────────────────┐
                         ▼                                        ▼
            decision: trust · model · fit · action      explanation: how · why not · report
```

| Component | File | Role |
|---|---|---|
| Knowledge base | `src/expert/knowledge_base.yaml` | thresholds, 35 IF-THEN rules with certainty factors, labels, conflict resolution |
| Fact extraction | `src/expert/facts.py` | measures the patch, the model outputs and the experiment results |
| Explainability | `src/expert/xai.py` | occlusion (radar vs optical reliance), input×gradient band saliency |
| Inference engine | `src/expert/engine.py` | fuzzification, stratified forward chaining, MYCIN certainty factors, trace |
| Advisor | `src/expert/advisor.py` | runs the models, gathers facts, calls the engine, packages the report |

The knowledge is **separate from the inference code**: changing the YAML file changes the system's
behaviour without touching Python — the defining property of an expert system.

## 2. Knowledge representation

### 2.1 Facts
None of the facts uses the clear target image; it does not exist when the system is used for real.

| Source | Fact | How it is measured |
|---|---|---|
| Input patch | `cloud_cover`, `thick_cloud` | pixels that are bright (mean B2–B4 > 0.20) **and** white (low colour spread); thick = > 0.30 |
| | `cirrus_fraction` | cirrus band B10 > 0.03 |
| | `sar_gaps` | radar pixels at the no-signal floor |
| | `water_fraction` | VV backscatter < −18 dB (open water reflects almost no radar) |
| Model outputs | `model_disagreement` | mean absolute difference between the three architectures' predictions |
| | `clear_pixel_error` | error on the *visible* (cloud-free) pixels, where the input is the truth |
| | `uncertainty` | median Laplace scale predicted by the variance-head U-Net under the clouds |
| | `sar_reliance` | share of the prediction change caused by removing the radar (occlusion, §4) |
| Experiments | `best_model_for_cloud_level` | best PSNR in `metrics_by_cloud_bin.csv` for this cloud bin |
| | `ood_drop` | PSNR lost on other continents (`metrics_main.csv`) |
| User | `use_case`, `region` | chosen in the interface; region is auto-detected from the scene or GeoTIFF |

### 2.2 Fuzzification
Numeric facts are mapped to symbolic levels by the thresholds in the knowledge base, e.g.
`cloud_cover: low [0, 0.2) · moderate [0.2, 0.6) · heavy [0.6, 1]`. A value within 10 % of a threshold is
*borderline* and enters the reasoning with certainty 0.7 instead of 1.0, so a patch with 19.5 % cloud is not
treated as confidently "low" as one with 5 %.

### 2.3 Rules
```yaml
- id: T4
  group: trust
  if: [{fact: cloud_cover, is: heavy}, {fact: sar_reliance, is: high},
       {fact: model_disagreement, in: [low, moderate]}]
  then: {fact: trust, is: medium}
  cf: 0.5
  because: "Heavy cloud ({cloud_pct:.0f}%), but the model draws most of its information from the
            radar ({sar_rel_pct:.0f}% reliance) and the architectures largely agree."
```
35 rules in five groups: **observations** (warnings) → **trust** → **model** → **suitability** → **action**.
The `because` template is filled with the measured values, which produces patch-specific explanations.

## 3. Inference

* **Stratified forward chaining.** Groups run in order; inside a group every rule is tried once.
* **Firing condition.** All premises hold with CF ≥ 0.2.
* **Certainty propagation.** Conclusion CF = rule CF × min(premise CFs).
* **Evidence combination** (MYCIN). Two rules supporting the same conclusion: CF = CF₁ + CF₂(1 − CF₁).
* **Conflict resolution.** Single-valued conclusions (trust, model, suitability, action) are resolved after
  their group: the highest CF wins; more specific rules carry higher CFs (*specificity*); exact ties go to
  the more cautious value (*conservatism*, `tie_break` in the KB). Later groups reason about that verdict.

## 4. Explanation facility

| Question | Answer |
|---|---|
| **How** was this concluded? | every fired rule, its premises, its CF contribution and the filled-in explanation |
| **What else** was considered? | all competing values with their combined CF |
| **Why not** another answer? | rules that did not fire and the first premise that failed |
| **What did the network look at?** | radar-reliance map (occlusion), band saliency (input × gradient), uncertainty map, disagreement map |
| **What if** the facts were different? | the *What-if simulator* on the Knowledge-base page re-runs the engine live |

**Occlusion.** The radar (or the cloudy optical image) is replaced by its spatial mean, which removes its
spatial information, and the change in the prediction is measured. Radar reliance =
Δ_radar / (Δ_radar + Δ_optical). Under thick cloud a sound model should rely on the radar; rule **T12**
lowers trust when it does not.

**Saliency.** |input × ∂(predicted RGB under the clouds)/∂input| summed per input channel shows which of the
15 bands (VV, VH, B1 … B12) drove the reconstruction.

## 5. Knowledge acquisition
Two sources:
1. **Domain knowledge** (remote sensing): radar detects open water directly, NDVI needs observed red/NIR,
   GAN outputs can contain invented detail, reconstruction errors look like change in change detection.
2. **Empirical knowledge from the deep-learning experiments**: which architecture is best at each cloud
   level, how much accuracy drops out of region, whether the uncertainty estimates are calibrated. The
   advisor reads these from the results CSVs, so retraining updates the expert system automatically.

## 6. Example consultation
Patch with 87 % cloud, models disagreeing (0.073), radar reliance 15 %, intended use: vegetation monitoring.

* W3 → warning: possible cirrus (0.70)
* T5, T6, T7, T8, T12 → trust = **low**, combined CF 0.99
* M1 → recommended model: best model for 80–100 % cloud (0.80); M3 → warning: avoid the GAN (0.90)
* U4 (thick cloud invalidates NDVI), U7 (low trust) → suitability = **not suitable** (0.88)
* A3 → action = **wait for a clearer acquisition** (0.83)

## 7. Evaluating the expert system (suggestions for the report)
* **Unit scenarios** (`tests/test_expert.py`): clear scene → accept; cloudy disagreeing scene → reject;
  flood mapping → radar-only; ties resolve conservatively; KB consistency checks.
* **Agreement with ground truth.** On the test gallery the true PSNR of each reconstruction is known:
  check that patches rated *high trust* really have higher PSNR than those rated *low trust*
  (e.g. mean PSNR per trust level, or a rank correlation between trust CF and PSNR).
* **Sensitivity analysis.** Use the What-if simulator to show how the decision changes with each fact and
  where the decision boundaries lie.
* **Expert review.** Ask a remote-sensing user whether the explanations are understandable and correct.

## 7a. Calibration and pilot evaluation on the real outputs (2026-10-22)
* **Cloud test:** on the 8 gallery patches (true cloud fraction known), the brightness threshold 0.17 gave a mean
  absolute error of 0.081 in cloud fraction; 0.20 gives 0.045. **Cirrus:** clear-sky B10 reaches 0.012-0.025 in
  some scenes, so the original threshold 0.01 flagged cirrus almost everywhere; raised to 0.03. Both were tuned on
  only 8 patches - an honest limitation.
* **Trust vs true PSNR (pilot, n = 8, intended use: land cover):** see the project guide, chapter 13.

## 8. Limitations and extensions
* Thresholds and certainty factors are expert judgements; they could be tuned on the test-gallery
  agreement study above.
* The cloud mask is a heuristic (bright deserts or snow look like cloud); s2cloudless would be better.
* Certainty factors are not probabilities; a Bayesian network over the same facts is a natural extension.
* New use cases or rules can be added to the YAML file without code changes.
