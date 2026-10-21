import pandas as pd
import streamlit as st

from frontend.common import hero, pill, section, verdict_grid
from src.expert.engine import InferenceEngine, KnowledgeBase

hero("Expert system · knowledge base",
     "How the advisor <span class='cs-grad'>reasons</span>",
     "A classic rule-based expert system: facts measured from the image and the deep-learning outputs, a knowledge "
     "base of IF-THEN rules with certainty factors, and an inference engine that chains them into a decision - "
     "keeping a full trace so every conclusion can be explained.")

kb = KnowledgeBase.load()

st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fontname="Inter", fontsize=11, color="#1F2A44", fillcolor="#121A2E", fontcolor="#E6EAF2", penwidth=1.2];
  edge [color="#2DD4BF", arrowsize=0.7, penwidth=1.2];
  in  [label="Patch\\nradar + cloudy optical"];
  dl  [label="Deep-learning models\\nU-Net · Swin · ViT\\n+ variance head", fillcolor="#0F2A2A"];
  xai [label="XAI\\nradar reliance\\nband saliency"];
  ex  [label="Experiment results\\nbest model per cloud level\\ndomain-shift drop"];
  f   [label="Fact extraction\\n+ fuzzification"];
  kb  [label="Knowledge base\\n%d rules · thresholds", fillcolor="#1E1B3A"];
  ie  [label="Inference engine\\nforward chaining\\ncertainty factors", fillcolor="#1E1B3A"];
  out [label="Decision\\ntrust · model · fit · action", fillcolor="#0F2A2A"];
  exp [label="Explanation\\nhow · why not · report"];
  in -> dl -> f; dl -> xai -> f; ex -> f; in -> f; f -> ie; kb -> ie; ie -> out; ie -> exp;
}""" % len(kb.rules))

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
c = st.columns(3)
sym = dict(use_case=c[0].selectbox("Intended use", list(kb.labels["use_case"]), format_func=kb.labels["use_case"].get),
           region=c[1].selectbox("Region", ["in_region", "out_of_region", "unknown"]),
           gan_available="yes")
best = c[2].selectbox("Best model at this cloud level (from experiments)", ["A_unet", "B_swin", "C_vit"])
ctx = dict(num, cloud_pct=100 * num["cloud_cover"], sar_rel_pct=100 * num["sar_reliance"],
           disagreement=num["model_disagreement"], clear_err=num["clear_pixel_error"], unc=num["uncertainty"],
           water_pct=100 * num["water_fraction"], sar_gaps_pct=100 * num["sar_gaps"], cirrus_pct=100 * num["cirrus_fraction"],
           best_model_for_cloud_level=best, best_model_label=best, best_model_psnr=float("nan"), cloud_bin="this",
           ood_drop=float("nan"))
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
