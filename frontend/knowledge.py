import pandas as pd
import streamlit as st

from frontend.common import final_root, hero, pill, section, verdict_grid
from src.expert.engine import InferenceEngine, KnowledgeBase

hero("Expert system · knowledge base",
     "How the advisor <span class='cs-grad'>reasons</span>",
     "A classic rule-based expert system: facts measured from the image and the deep-learning outputs, a knowledge "
     "base of IF-THEN rules with certainty factors, and an inference engine that chains them into a decision - "
     "keeping a full trace so every conclusion can be explained.")

kb = KnowledgeBase.load()

# the pipeline as it really runs: with the final model's outputs (outputs/final) or with the Phase 1 models only
FINAL = final_root() is not None
if FINAL:
    evidence = [("teal", "ClearSAR-Net", "reconstruction by the final (GAN) and before-GAN models; flip-test uncertainty"),
                ("", "Phase 1 models", "U-Net · Swin · ViT, run only to measure how much independent models disagree"),
                ("", "XAI", "radar reliance (occlusion) and band saliency of the most accurate model"),
                ("", "Test-set experience", "final run, 18 unseen scenes: PSNR per cloud level, cost of the GAN")]
else:
    evidence = [("teal", "Phase 1 models", "U-Net · Swin · ViT reconstructions, variance-head uncertainty, disagreement"),
                ("", "XAI", "radar reliance (occlusion) and band saliency of the best model"),
                ("", "Phase 1 results", "best model per cloud level, drop outside the training region")]
stages = [
    ("1 · Measure", [("", "Observations", "cloud cover, thin cirrus, open water and radar gaps, from the patch itself")]
     + evidence),
    ("2 · Fuzzify", [("", "Fact extraction", ("every measurement becomes a level (low / medium / high) using the "
                                       "thresholds below; values near a threshold enter as borderline (CF 0.7)"))]),
    ("3 · Reason", [("violet", "Knowledge base", f"{len(kb.rules)} IF-THEN rules in {len(kb.groups)} groups: "
                     + " → ".join(kb.groups)),
                    ("violet", "Inference engine", ("forward chaining with MYCIN certainty factors; a full trace of "
                                                  "every rule that fired or failed"))]),
    ("4 · Decide & explain", [("teal", "Decision", "trust · recommended model · fit for purpose · action"),
                              ("", "Explanation", "how (rules that fired), why not (rules that failed), report")]),
]
html = "".join(
    f"<li class='cs-stage'><div class='cs-stage-h'>{title}</div>"
    + "".join(f"<div class='cs-node {tone}'><b>{name}</b><span>{text}</span></div>" for tone, name, text in nodes)
    + "</li>" for title, nodes in stages)
st.markdown("""<style>
.cs-flow { list-style: none; padding: 0; margin: .2rem 0 .4rem 0; display: grid; gap: 30px;
  grid-template-columns: 1.25fr 1fr 1fr 1fr; }
.cs-stage { position: relative; display: flex; flex-direction: column; gap: 10px; margin: 0; }
.cs-stage:not(:last-child)::after { content: "→"; position: absolute; right: -24px; top: 50%; color: var(--sky);
  font-size: 1.3rem; transform: translateY(-50%); }
.cs-stage-h { color: #99F6E4; font-size: .74rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
.cs-node { border-radius: 14px; padding: 11px 14px; background: var(--panel-solid); border: 1px solid var(--line); }
.cs-node.teal { border-color: rgba(45, 212, 191, .55); background: linear-gradient(140deg, rgba(45,212,191,.12), var(--panel-solid) 60%); }
.cs-node.violet { border-color: rgba(167, 139, 250, .55); background: linear-gradient(140deg, rgba(167,139,250,.14), var(--panel-solid) 60%); }
.cs-node b { display: block; font-size: .93rem; margin-bottom: .15rem; }
.cs-node span { color: var(--muted); font-size: .82rem; line-height: 1.45; display: block; }
@media (max-width: 900px) { .cs-flow { grid-template-columns: 1fr; gap: 34px; }
  .cs-stage:not(:last-child)::after { content: "↓"; right: auto; left: 50%; top: auto; bottom: -30px; transform: none; } }
</style>""" + f"<ol class='cs-flow' aria-label='How the advisor reaches a decision'>{html}</ol>", unsafe_allow_html=True)

# ---------------------------------------------------------------- what-if simulator
section("What-if simulator", "Change the facts and watch the decision change - no image needed.")
c = st.columns(4)
num = dict(
    cloud_cover=c[0].slider("Cloud cover", 0.0, 1.0, 0.45, 0.01),
    model_disagreement=c[1].slider("Disagreement between models", 0.0, 0.08, 0.015, 0.001, format="%.3f"),
    clear_pixel_error=c[2].slider("Error on visible pixels", 0.0, 0.12, 0.025, 0.001, format="%.3f"),
    sar_reliance=c[3].slider("Radar reliance under cloud", 0.0, 1.0, 0.55, 0.01),
)
c = st.columns(4)
num.update(
    uncertainty=c[0].slider("Predicted uncertainty", 0.0, 0.1, 0.03, 0.001, format="%.3f"),
    water_fraction=c[1].slider("Open water", 0.0, 1.0, 0.02, 0.01),
    sar_gaps=c[2].slider("Radar gaps", 0.0, 0.5, 0.0, 0.01),
    cirrus_fraction=c[3].slider("Cirrus", 0.0, 1.0, 0.05, 0.01),
)
c = st.columns(4)
sym = dict(use_case=c[0].selectbox("Intended use", list(kb.labels["use_case"]), format_func=kb.labels["use_case"].get),
           region=c[1].selectbox("Region", ["in_region", "out_of_region", "unknown"]),
           gan_available="yes")
best = c[2].selectbox("Most accurate model at this cloud level (from the test)", ["before_gan", "A_unet", "B_swin", "C_vit"],
                      format_func=lambda r: {"before_gan": "ClearSAR-Net · before GAN"}.get(r, r))
num["gan_psnr_cost"] = c[3].slider("PSNR cost of the GAN model here (dB)", -0.5, 3.0, 0.1, 0.01, format="%.2f",
                                   help="How far the GAN-trained model trails the most accurate one at this cloud level in "
                                        "the test results. Final model: at most 0.14 dB; Phase 1 GAN: 2-3 dB.")
gan = "final" if best == "before_gan" else "A_gan"
nongan = best
ctx = dict(num, cloud_pct=100 * num["cloud_cover"], sar_rel_pct=100 * num["sar_reliance"],
           disagreement=num["model_disagreement"], clear_err=num["clear_pixel_error"], unc=num["uncertainty"],
           water_pct=100 * num["water_fraction"], sar_gaps_pct=100 * num["sar_gaps"], cirrus_pct=100 * num["cirrus_fraction"],
           best_model_for_cloud_level=best, best_model_label=best, best_model_psnr=float("nan"), cloud_bin="this",
           ood_drop=float("nan"), gan_model=gan, gan_model_label=gan, nongan_model=nongan, nongan_model_label=nongan,
           gan_cost=num["gan_psnr_cost"])
res = InferenceEngine(kb).run(num, sym, ctx)
items = []
for fact in ["trust", "recommended_model", "suitability", "action"]:
    v, cf = res.value(fact)
    items.append((fact.replace("_", " "), kb.labels.get(fact, {}).get(v, {}).get("title", v), "#E6EAF2", "", cf))
verdict_grid(items)
fired = ", ".join(f"{r.id}" for r in res.fired)
st.markdown(f"<div style='margin-top:.6rem;color:#8B95A7'>Rules fired: <span class='cs-mono'>{fired}</span></div>",
            unsafe_allow_html=True)

# ---------------------------------------------------------------- rules
section("The rules", "Edit src/expert/knowledge_base.yaml to change them - the engine needs no code changes.")
group = st.segmented_control("Group", kb.groups, default="trust", key="kb_group") or "trust"
for r in [r for r in kb.rules if r["group"] == group]:
    cond = " AND ".join(f"{p['fact']} {'∈ {' + ', '.join(p['in']) + '}' if 'in' in p else '= ' + str(p['is'])}"
                        for p in r["if"])
    then = r["then"]
    concl = f"{then['fact']} = {then.get('is', '⟨' + then.get('value_from', '?') + '⟩')}"
    hit = pill("fired in simulator", "good") if any(fr.id == r["id"] for fr in res.fired) else ""
    st.markdown(f"<div class='cs-rule'><span class='id'>{r['id']} · CF {r['cf']}</span> {hit}"
                f"<div class='cs-mono' style='font-size:.82rem;color:#C7D0DF;margin:.25rem 0'>IF {cond}<br>THEN {concl}</div>"
                f"<div class='txt' style='color:#8B95A7'>{r.get('because', '')}</div></div>", unsafe_allow_html=True)

with st.expander("Thresholds (fuzzification of numeric facts)"):
    rows = [dict(fact=f, level=str(lv), **{"from": lo, "to": hi}) for f, levels in kb.thresholds.items()
            for lv, (lo, hi) in levels.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
with st.expander("Certainty-factor algebra"):
    st.markdown(
        "- A rule fires when every premise holds with certainty ≥ "
        f"{kb.meta['min_premise_cf']}.\n"
        "- Its conclusion receives **CF = rule CF × min(premise CFs)**.\n"
        "- Two rules supporting the same conclusion combine as **CF₁ + CF₂ · (1 − CF₁)** (MYCIN).\n"
        f"- Rules run in groups ({' → '.join(kb.groups)}); after each group the single-valued conclusions "
        "(trust, model, suitability, action) keep their highest-CF value, so later rules reason about one verdict.\n"
        f"- A measured value within {int(100 * kb.meta['borderline'])}% of a threshold is *borderline* and enters "
        "with CF 0.7.")
