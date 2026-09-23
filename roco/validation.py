"""Internal validation used for early stopping and checkpoint selection.

49 clean held-out pairs plus four synthetic corruptions of the first pair of
each held-out scene. The selection score is

    N(c) = D1Abs/7.466 + D2Abs/7.5935 + FlowEPE/2.527
    J    = 0.5 * (N(clean) + mean over corruptions of N(c))     (lower is better)
"""
import collections

import numpy as np
import torch

from . import augment, data, metrics, models

CONDITIONS = ("clean", "brightness", "noise", "blur", "jpeg")


def corrupt(raw, name, seed):
    if name == "clean":
        return raw
    out = {k: v.copy() for k, v in raw.items()}
    nr = np.random.default_rng(seed)
    for k in data.IMAGES:
        if name == "brightness":
            out[k] = np.clip(out[k].astype(np.float32) * .65, 0, 255).astype(np.uint8)
        else:
            out[k] = augment.degrade(out[k], name, .5, nr)
    return out


def selection_score(results):
    def norm(m):
        return m["d1_abs"] / 7.466 + m["d2_abs"] / 7.5935 + m["flow_epe"] / 2.527
    return .5 * (norm(results["clean"]) + np.mean([norm(results[c]) for c in CONDITIONS[1:]]))


@torch.inference_mode()
def evaluate(stereo, flow, val, seed, stereo_device="cuda:0", flow_device="cuda:0", log=print):
    stereo.eval()
    flow.eval()
    results, rows = {}, {}
    for condition in CONDITIONS:
        totals = collections.defaultdict(float)
        rows[condition] = []
        subset = val if condition == "clean" else val[::7]
        for i, pair in enumerate(subset):
            t = data.to_tensors(corrupt(data.read_pair(pair), condition, seed + i), torch.device(stereo_device))
            d1 = models.infer_stereo(stereo, t["l0"], t["r0"], 16, 6)[None, None]
            dn = models.infer_stereo(stereo, t["l1"], t["r1"], 16, 6)[None, None]
            fw = models.flow_forward(flow, t["l0"].to(flow_device), t["l1"].to(flow_device))["flows"][:, 0].to(stereo_device)
            d2, _ = models.warp_field(dn, fw)
            sums = metrics.metric_sums((d1, d2, fw), t)
            for k, v in sums.items():
                totals[k] += v
            rows[condition].append(dict(sample=f"{pair.scene.name}:{pair.frame}", metrics=metrics.metrics(sums)))
        results[condition] = metrics.metrics(totals)
        log(f"validation {condition}: SF1px {results[condition]['sf_1px']:.3f}")
    results["selection_score"] = selection_score(results)
    return results, rows
