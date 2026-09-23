"""Evaluate a checkpoint on the frozen analysis pairs, clean and corrupted.

Seven corruption families at three severities. Two corruption protocols:
  independent  each view is corrupted with its own random realization
  consistent   rain/snow/spatter use one mask drawn in L_t and moved to the other
               views with ground-truth disparity and flow (blur, JPEG and pixelation
               depend only on the severity, so both protocols agree on them)
"""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roco import augment, data, metrics, models, paths  # noqa: E402
from roco.utils import devices, dump, sha256  # noqa: E402

FAMILIES = ("blur", "noise", "jpeg", "pixelate", "rain", "snow", "spatter")
SEVERITIES = (0.25, 0.5, 0.75)
SPATIAL = ("rain", "snow", "spatter")
KEYS = ("d1_abs", "d2_abs", "flow_epe", "sf_1px", "sf_outlier")


def corrupt(raw, family, severity, seed, protocol):
    out = {k: v.copy() for k, v in raw.items()}
    nr, rng = np.random.default_rng(seed), random.Random(seed)
    if protocol == "consistent" and family in SPATIAL:
        mask = augment.weather_mask(out["l0"].shape[:2], family, severity, rng, nr)
        for k, m in zip(data.IMAGES, augment.transport(mask, raw)):
            out[k] = augment.overlay(out[k], m, nr, white=True)
        return out
    for k in data.IMAGES:
        if family in SPATIAL:
            mask = augment.weather_mask(out[k].shape[:2], family, severity, rng, nr)
            out[k] = augment.overlay(out[k], mask, nr, white=True)
        else:
            out[k] = augment.degrade(out[k], family, severity, nr)
    return out


def conditions(families, with_clean):
    if with_clean:
        yield "clean", None, None, 9610
    for fi, family in enumerate(FAMILIES):
        if family in families:
            for si, severity in enumerate(SEVERITIES):
                yield f"{family}_v{severity:.2f}", family, severity, 9610 + 1000 * fi + 100 * si


def summarize(results):
    mean = lambda names: {k: float(np.mean([results[n][k] for n in names])) for k in KEYS}
    name = lambda f, s: f"{f}_v{s:.2f}"
    return dict(clean=results["clean"],
                corruption_mean=mean([n for n in results if n != "clean"]),
                per_family={f: mean([name(f, s) for s in SEVERITIES]) for f in FAMILIES},
                per_severity={f"v{s:.2f}": mean([name(f, s) for f in FAMILIES]) for s in SEVERITIES},
                non_spatial_mean=mean([name(f, s) for f in FAMILIES if f not in SPATIAL for s in SEVERITIES]),
                spatial_mean=mean([name(f, s) for f in SPATIAL for s in SEVERITIES]))


@torch.inference_mode()
def main(a):
    S, F = devices()
    pairs = [data.SpringPair(a.data / "train" / s, int(f))
             for s, f in (line.split() for line in a.pairs.read_text().splitlines())]
    if a.limit:
        pairs = pairs[:a.limit]
    stereo, flow = models.load_pair(a.checkpoint, a.stereo_init, a.flow_init, S, F)
    out_dir = a.out / a.tag
    results = {}
    for name, family, severity, base_seed in conditions(a.families or FAMILIES, not a.no_clean):
        path = out_dir / f"{name}.json"
        if path.exists():
            results[name] = json.loads(path.read_text())["summary"]
            continue
        totals, rows = collections.defaultdict(float), []
        started = time.perf_counter()
        for i, pair in enumerate(pairs):
            raw = data.read_pair(pair)
            if family is not None:
                raw = corrupt(raw, family, severity, base_seed + i, a.protocol)
            t = data.to_tensors(raw, torch.device(S))
            d1 = models.infer_stereo(stereo, t["l0"], t["r0"], 16, 6)[None, None]
            dn = models.infer_stereo(stereo, t["l1"], t["r1"], 16, 6)[None, None]
            fw = models.flow_forward(flow, t["l0"].to(F), t["l1"].to(F))["flows"][:, 0].to(S)
            d2, _ = models.warp_field(dn, fw)
            sums = metrics.metric_sums((d1, d2, fw), t)
            for k, v in sums.items():
                totals[k] += v
            rows.append(dict(sample=f"{pair.scene.name}:{pair.frame:04d}", scene=pair.scene.name,
                             frame=pair.frame, metrics=metrics.metrics(sums), sums=sums))
        results[name] = metrics.metrics(totals)
        dump(path, dict(summary=results[name], sums=dict(totals), rows=rows, protocol=a.protocol,
                        family=family, severity=severity, seed_base=base_seed,
                        seconds=time.perf_counter() - started))
        print(f"{a.tag} {name}: SF1px {results[name]['sf_1px']:.3f}", flush=True)
    complete = len(results) == 1 + 3 * len(FAMILIES)
    dump(out_dir / "summary.json", dict(
        model=a.tag, checkpoint=str(a.checkpoint) if a.checkpoint else "public",
        checkpoint_sha256=sha256(a.checkpoint) if a.checkpoint else None, pairs=len(pairs),
        protocol=a.protocol, conditions=results, aggregate=summarize(results) if complete else None))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--checkpoint", type=Path, default=None, help="omit for the public pretrained pair")
    p.add_argument("--protocol", choices=["independent", "consistent"], default="independent")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--no-clean", action="store_true")
    p.add_argument("--data", type=Path, default=paths.SPRING)
    p.add_argument("--pairs", type=Path, default=paths.REPO / "configs/frozen_pairs.txt")
    p.add_argument("--stereo-init", type=Path, default=paths.DEFOM_INIT)
    p.add_argument("--flow-init", type=Path, default=paths.DPFLOW_INIT)
    p.add_argument("--out", type=Path, default=paths.REPO / "results/frozen")
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
