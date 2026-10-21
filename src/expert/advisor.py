"""
ClearSAR Advisor - an explainable decision-support system on top of the deep-learning models.

    advisor = Advisor("outputs/checkpoints", "outputs/results", labels)
    report = advisor.advise(x, use_case="vegetation_monitoring", region="in_region")

Pipeline for one patch
  1. run the deep-learning models (U-Net, Swin U-Net, ViT, + variance-head U-Net if available)
  2. extract facts: observations, model behaviour, experience from the experiments  (facts.py)
  3. explain the recommended model: radar-vs-optical reliance and band saliency        (xai.py)
  4. reason over the knowledge base with certainty factors                            (engine.py)
  5. return the decision (trust, model, suitability, action), the warnings and the full trace
"""
import os
from dataclasses import dataclass, field

import numpy as np
import torch

from src.data.dataset import OUT_CHANNELS
from src.evaluation.evaluate import predict_tiled
from src.expert.engine import InferenceEngine, KnowledgeBase
from src.expert.facts import (ResultsKnowledge, cloud_bin, model_facts, observation_facts)
from src.expert.xai import channel_saliency, occlusion, sar_reliance

CORE = ("A_unet", "B_swin", "C_vit")


def region_from_lonlat(lon, lat, train_region="Asia"):
    """'in_region' / 'out_of_region' from coordinates (offline reverse geocoding)."""
    try:
        import reverse_geocoder as rg

        from src.data.continents import to_continent
        cc = rg.search([(lat, lon)], mode=1, verbose=False)[0]["cc"]
        return ("in_region" if to_continent(cc) == train_region else "out_of_region"), cc
    except Exception:
        return "unknown", None


@dataclass
class Advice:
    use_case: str
    numeric: dict
    symbolic: dict
    context: dict
    result: object                       # InferenceResult
    preds: dict                          # run -> [13, H, W]
    reconstruction_run: str
    cloud_mask: np.ndarray
    sigma: np.ndarray = None
    occlusion: dict = field(default_factory=dict)
    saliency: np.ndarray = None
    kb: object = None

    @property
    def reconstruction(self):
        return self.preds[self.reconstruction_run]

    def verdict(self, fact):
        return self.result.value(fact)


class Advisor:
    def __init__(self, checkpoint_dir, results_dir=None, labels=None, kb_path=None, device="cpu"):
        self.ckpt_dir = checkpoint_dir
        self.device = torch.device(device)
        self.kb = KnowledgeBase.load(kb_path) if kb_path else KnowledgeBase.load()
        self.engine = InferenceEngine(self.kb)
        self.labels = labels or {}
        self.experience = ResultsKnowledge(results_dir, self.labels)
        self.available = sorted(f[5:-3] for f in os.listdir(checkpoint_dir)
                                if f.startswith("best_") and f.endswith(".pt")) if os.path.isdir(checkpoint_dir) else []
        self._models = {}

    def model(self, run):
        if run not in self._models:
            from src.models.architectures import load_model_from_checkpoint
            self._models[run] = load_model_from_checkpoint(os.path.join(self.ckpt_dir, f"best_{run}.pt"),
                                                           self.device)
        return self._models[run]

    @torch.no_grad()
    def _predict(self, run, x):
        model, meta = self.model(run)
        out = predict_tiled(model, torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(self.device))[0]
        pred = out[:OUT_CHANNELS].cpu().numpy()
        sigma = torch.exp(out[OUT_CHANNELS:].clamp(-10, 10)).mean(0).cpu().numpy() if meta.get("uncertainty") else None
        return pred, sigma

    def advise(self, x, use_case="visual_inspection", region="unknown"):
        if not self.available:
            raise RuntimeError("No checkpoints found - run the Kaggle pipeline and extract its output first.")
        obs, cloud = observation_facts(x)
        preds, sigma = {}, None
        for run in [r for r in CORE + ("A_unc",) if r in self.available]:
            preds[run], s = self._predict(run, x)
            if s is not None:
                sigma = s
        bin_name = cloud_bin(obs["cloud_cover"])
        best, best_psnr = self.experience.best_model(bin_name, self.available)
        best = best or next(iter(preds))
        if best not in preds:
            preds[best], _ = self._predict(best, x)

        mfacts = model_facts(x, preds, cloud, core=CORE, primary=best, sigma=sigma)
        primary_model = self.model(best)[0]
        occ = occlusion(primary_model, x)
        rel = sar_reliance(occ, cloud)
        sal = channel_saliency(primary_model, x, cloud)

        numeric = {**obs, **mfacts, "sar_reliance": rel}
        symbolic = {"use_case": use_case, "region": region,
                    "gan_available": "yes" if "A_gan" in self.available else "no"}
        ctx = dict(numeric)
        ctx.update(
            cloud_pct=100 * obs["cloud_cover"], sar_gaps_pct=100 * obs["sar_gaps"],
            water_pct=100 * obs["water_fraction"], cirrus_pct=100 * obs["cirrus_fraction"],
            sar_rel_pct=100 * rel, disagreement=mfacts.get("model_disagreement", float("nan")),
            clear_err=mfacts.get("clear_pixel_error", float("nan")), unc=mfacts.get("uncertainty", float("nan")),
            cloud_bin=bin_name, best_model_for_cloud_level=best,
            best_model_label=self.experience.label(best), best_model_psnr=best_psnr,
            ood_drop=self.experience.ood_drop(best),
        )
        result = self.engine.run(numeric, symbolic, ctx)

        rec, _ = result.value("recommended_model")
        rec = rec if rec in self.available else best
        if rec not in preds:
            preds[rec], _ = self._predict(rec, x)
        return Advice(use_case=use_case, numeric=numeric, symbolic=symbolic, context=ctx, result=result,
                      preds=preds, reconstruction_run=rec, cloud_mask=cloud, sigma=sigma, occlusion=occ,
                      saliency=sal, kb=self.kb)
