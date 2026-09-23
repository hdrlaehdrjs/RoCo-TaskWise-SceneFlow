"""Scene-clustered paired bootstrap for differences between two models.

Reads the per-pair records written by tools/frozen_eval.py. For each pair the metric
difference (A - B) is averaged over the chosen conditions; the seven held-out scenes are
then resampled with replacement. Lower is better, so a negative difference favours A.
The interval reflects variability over held-out scenes, not over training seeds.
"""
import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roco import paths  # noqa: E402

FAMILIES = ("blur", "noise", "jpeg", "pixelate", "rain", "snow", "spatter")
SPATIAL = ("rain", "snow", "spatter")
SEVERITIES = (0.25, 0.5, 0.75)
GROUPS = {
    "all": [f"{f}_v{s:.2f}" for f in FAMILIES for s in SEVERITIES],
    "spatial": [f"{f}_v{s:.2f}" for f in SPATIAL for s in SEVERITIES],
    "non_spatial": [f"{f}_v{s:.2f}" for f in FAMILIES if f not in SPATIAL for s in SEVERITIES],
    "clean": ["clean"],
}


def load(results, tag, conditions):
    out = {}
    for c in conditions:
        rows = json.loads((results / tag / f"{c}.json").read_text())["rows"]
        out[c] = {r["sample"]: r["metrics"] for r in rows}
    return out


def bootstrap(a, b, conditions, metric, replicates, seed):
    samples = sorted(a[conditions[0]])
    per_sample = {s: float(np.mean([a[c][s][metric] - b[c][s][metric] for c in conditions])) for s in samples}
    scenes = sorted({s.split(":")[0] for s in samples})
    scene_mean = {sc: np.mean([v for s, v in per_sample.items() if s.startswith(sc + ":")]) for sc in scenes}
    rng = np.random.default_rng(seed)
    draws = np.array([np.mean([scene_mean[sc] for sc in rng.choice(scenes, len(scenes), replace=True)])
                      for _ in range(replicates)])
    return dict(mean_delta=float(np.mean(list(per_sample.values()))), median_delta=float(np.median(draws)),
                ci_low=float(np.percentile(draws, 2.5)), ci_high=float(np.percentile(draws, 97.5)),
                fraction_below_zero=float(np.mean(draws < 0)))


def main(a):
    conditions = GROUPS.get(a.conditions[0], a.conditions) if len(a.conditions) == 1 else a.conditions
    rows = []
    for pair in a.compare:
        tag_a, tag_b = pair.split(":")
        da, db = load(a.results, tag_a, conditions), load(a.results, tag_b, conditions)
        for metric in a.metrics:
            # Seeded per comparison, so results do not depend on argument order.
            seed = [a.seed, zlib.crc32(f"{tag_a}:{tag_b}:{metric}".encode())]
            r = bootstrap(da, db, conditions, metric, a.replicates, seed)
            rows.append(dict(a=tag_a, b=tag_b, metric=metric, conditions=len(conditions), **r))
            print(f"{tag_a} - {tag_b} {metric}: {r['mean_delta']:+.3f} "
                  f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  P(<0)={r['fraction_below_zero']:.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=paths.REPO / "results/frozen")
    p.add_argument("--compare", nargs="+", required=True, help="pairs as TAG_A:TAG_B")
    p.add_argument("--conditions", nargs="+", default=["all"],
                   help="all, spatial, non_spatial, clean, or explicit condition names")
    p.add_argument("--metrics", nargs="+", default=["sf_1px"])
    p.add_argument("--replicates", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260919)
    p.add_argument("--out", type=Path, default=None)
    main(p.parse_args())
