"""Tests for the ClearSAR Advisor expert system: python -m pytest tests/test_expert.py -q"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.expert.engine import InferenceEngine, KnowledgeBase  # noqa: E402
from src.expert.facts import cloud_bin, cloud_masks, model_facts, observation_facts  # noqa: E402

KB = KnowledgeBase.load()
SYMBOLIC = {"use_case", "region", "gan_available"}


def run(num, **sym):
    s = dict(use_case="visual_inspection", region="in_region", gan_available="no")
    s.update(sym)
    ctx = dict(num, best_model_for_cloud_level="A_unet")
    return InferenceEngine(KB).run(num, s, ctx)


GOOD = dict(cloud_cover=0.1, model_disagreement=0.01, clear_pixel_error=0.02, sar_reliance=0.5,
            uncertainty=0.01, water_fraction=0.0, sar_gaps=0.0, cirrus_fraction=0.0, thick_cloud=0.0)
BAD = dict(GOOD, cloud_cover=0.9, thick_cloud=0.6, model_disagreement=0.06, uncertainty=0.08, sar_reliance=0.1)


def test_kb_is_consistent():
    produced = set(KB.thresholds) | SYMBOLIC | {r["then"]["fact"] for r in KB.rules}
    for r in KB.rules:
        for p in r["if"]:
            assert p["fact"] in produced, f"{r['id']} uses unknown fact {p['fact']}"
            if p["fact"] in KB.thresholds:
                wanted = p.get("in", [p.get("is")])
                assert set(map(str, wanted)) <= set(map(str, KB.thresholds[p["fact"]])), r["id"]
    assert len({r["id"] for r in KB.rules}) == len(KB.rules), "duplicate rule ids"
    for fact in ("trust", "suitability", "action"):
        concluded = {r["then"].get("is") for r in KB.rules if r["then"]["fact"] == fact}
        assert concluded <= set(KB.labels[fact]), f"unlabelled {fact} values {concluded - set(KB.labels[fact])}"


def test_cf_combination_and_fuzzify():
    assert InferenceEngine._combine(0.6, 0.5) == pytest.approx(0.8)
    assert KB.fuzzify("cloud_cover", 0.05)[:2] == ("low", 1.0)
    level, cf, border = KB.fuzzify("cloud_cover", 0.195)           # just below the 0.2 threshold
    assert level == "low" and border and cf == 0.7
    assert KB.fuzzify("cloud_cover", 0.9)[0] == "heavy"


def test_clear_scene_is_trusted_and_accepted():
    r = run(GOOD)
    assert r.value("trust")[0] == "high"
    assert r.value("action")[0] == "accept"
    assert any(f.id == "T1" for f in r.fired)


def test_cloudy_disagreeing_scene_is_rejected():
    r = run(BAD, use_case="vegetation_monitoring")
    assert r.value("trust")[0] == "low"
    assert r.value("suitability")[0] == "unsuitable"
    assert r.value("action")[0] in ("reacquire", "sar_only")
    assert "avoid_gan" in r.warnings()


def test_flood_mapping_uses_radar_and_domain_shift_warns():
    r = run(GOOD, use_case="flood_mapping", region="out_of_region")
    assert r.value("action")[0] == "sar_only"
    assert "domain_shift" in r.warnings()


def test_ties_resolve_conservatively_and_why_not_is_recorded():
    r = run(dict(GOOD, cloud_cover=0.45, clear_pixel_error=0.045))   # T3 (medium) fires, T2 (high) does not
    assert r.value("trust")[0] == "medium"
    assert any(n[0] == "T2" and "clear_pixel_error" in n[3] for n in r.not_fired)
    assert KB.best("trust", {"high": 0.6, "low": 0.6}) == "low"


def test_fact_extraction_from_patch():
    x = np.zeros((15, 128, 128), np.float32)
    x[0] = 0.6                                   # VV ~ -10 dB: land
    x[2:] = 0.08                                 # dark ground
    x[2:, :64] = 0.5                             # left half: bright white cloud
    cloud, thick = cloud_masks(x)
    assert abs(cloud.mean() - 0.5) < 0.01 and thick.mean() > 0.4
    obs, cloud = observation_facts(x)
    assert obs["sar_gaps"] == 0 and obs["water_fraction"] == 0
    preds = {"A_unet": x[2:].copy(), "B_swin": x[2:] + 0.01, "C_vit": x[2:] - 0.01}
    mf = model_facts(x, preds, cloud, primary="A_unet")
    assert mf["clear_pixel_error"] == pytest.approx(0.0, abs=1e-6)
    assert mf["model_disagreement"] == pytest.approx(0.0133, abs=1e-3)
    assert cloud_bin(0.5) == "40-60%" and cloud_bin(0.95) == "80-100%"


@pytest.mark.skipif(not os.path.isdir(os.path.join(ROOT, "outputs", "checkpoints")), reason="no trained outputs")
def test_advisor_end_to_end_on_outputs():
    import json
    from src.expert.advisor import Advisor
    z = np.load(os.path.join(ROOT, "outputs", "results", "samples.npz"))
    adv = Advisor(os.path.join(ROOT, "outputs", "checkpoints"), os.path.join(ROOT, "outputs", "results"))
    a = adv.advise(z["x"][0].astype(np.float32), use_case="land_cover", region="in_region")
    for fact in ("trust", "recommended_model", "suitability", "action"):
        assert a.verdict(fact)[0] is not None
    assert a.reconstruction.shape[0] == 13 and 0 <= a.numeric["sar_reliance"] <= 1
    assert a.saliency.shape == (15,) and abs(a.saliency.sum() - 1) < 1e-5
