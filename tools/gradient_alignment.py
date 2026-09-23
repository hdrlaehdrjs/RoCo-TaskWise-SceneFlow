"""Agreement between the D2 gradient and each network's own training gradient.

For a fixed list of training quadruplets, every checkpoint is evaluated without any
update: we measure the cosine between grad(L_D2) and grad(L_s) over the stereo
parameters, and between grad(L_D2) and grad(L_f) over the flow parameters, together
with the norm ratio |grad L_D2| / |grad L_native|.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roco import augment, data, losses, models, paths  # noqa: E402
from roco.utils import devices, seed_all  # noqa: E402


def cosine(a, b, params):
    a = [torch.zeros_like(p) if g is None else g for g, p in zip(a, params)]
    b = [torch.zeros_like(p) if g is None else g for g, p in zip(b, params)]
    dot = sum(float((x * y).sum()) for x, y in zip(a, b))
    na = np.sqrt(sum(float((x * x).sum()) for x in a))
    nb = np.sqrt(sum(float((y * y).sum()) for y in b))
    return (dot / (na * nb) if na > 0 and nb > 0 else float("nan")), (nb / na if na > 0 else float("nan"))


def main(a):
    S, F = devices()
    seed_all(a.seed)
    pairs, _ = data.splits(a.data, a.split)
    dataset = data.QuadDataset(pairs, a.seed, (320, 640), augment.RECIPES["full"])
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(a.seed)).tolist()[:a.batches]
    batches = list(DataLoader(dataset, batch_size=1, sampler=order, num_workers=2))

    rows = []
    for spec in a.checkpoint:
        label, path = spec.split("=", 1)
        stereo, flow = models.load_pair(path, a.stereo_init, a.flow_init, S, F)
        stereo.requires_grad_(True).args.detach_disparity = False
        flow.requires_grad_(True).detach_flow = False
        stereo.train()
        stereo.freeze_bn()
        flow.train()
        for module in flow.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        theta, phi = list(stereo.parameters()), list(flow.parameters())
        for i, batch in enumerate(batches):
            left = torch.cat((batch["l0"], batch["l1"])).to(S)
            right = torch.cat((batch["r0"], batch["r1"])).to(S)
            gt = torch.cat((batch["d1"], batch["dn"])).to(S)
            gt2, gtf = batch["d2"].to(S), batch["flow"].to(F)
            predictions = stereo(left, right, iters=8, scale_iters=4, test_mode=False)
            out = models.flow_forward(flow, batch["l0"].to(F), batch["l1"].to(F), gtf)
            fw, native = out["flows"][:, 0], predictions[-1][1:]
            d2, _ = models.warp_field(native, fw.to(S))
            with torch.no_grad():
                _, bounds = models.warp_field(gt[1:], torch.nan_to_num(gtf.to(S)))
                valid2 = losses.valid_disp(gt2) & bounds & torch.isfinite(gtf.to(S)).all(1, keepdim=True)
            ls = losses.stereo_loss(predictions, gt)
            lf = losses.flow_loss(out, gtf)
            ld2 = losses.d2_loss(d2, gt2, valid2)
            grad = lambda loss, params: torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            cos_s, ratio_s = cosine(grad(ls, theta), grad(ld2, theta), theta)
            cos_f, ratio_f = cosine(grad(lf, phi), grad(ld2, phi), phi)
            rows.append(dict(checkpoint=label, batch=i, sample=batch["sample"][0],
                             cos_stereo=cos_s, cos_flow=cos_f, norm_ratio_stereo=ratio_s, norm_ratio_flow=ratio_f))
        del stereo, flow
        torch.cuda.empty_cache()

    for label in dict.fromkeys(r["checkpoint"] for r in rows):
        sel = [r for r in rows if r["checkpoint"] == label]
        for space in ("stereo", "flow"):
            c = np.array([r["cos_" + space] for r in sel])
            n = np.array([r["norm_ratio_" + space] for r in sel])
            print(f"{label} {space}: mean cos {c.mean():+.3f}, median {np.median(c):+.3f}, "
                  f"negative {100 * (c < 0).mean():.1f}%, median norm ratio {np.median(n):.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True, help="LABEL=path/to/best.pt, repeatable")
    p.add_argument("--batches", type=int, default=48)
    p.add_argument("--data", type=Path, default=paths.SPRING)
    p.add_argument("--split", type=Path, default=paths.REPO / "configs/spring_holdout.json")
    p.add_argument("--stereo-init", type=Path, default=paths.DEFOM_INIT)
    p.add_argument("--flow-init", type=Path, default=paths.DPFLOW_INIT)
    p.add_argument("--seed", type=int, default=9610)
    p.add_argument("--out", type=Path, default=None)
    main(p.parse_args())
