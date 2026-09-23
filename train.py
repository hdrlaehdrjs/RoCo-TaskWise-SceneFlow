"""Task-wise fine-tuning of DEFOM-Stereo and DPFlow on Spring.

    L = L_s + L_f^EPE + 0.01 L_f^NLL + lambda_D2 * L_D2

The submitted model ("final") uses lambda_D2 = 0: the composed disparity d2 is
computed and logged but not optimized. Other experiments change one factor.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from roco import augment, data, losses, models, paths, validation
from roco.utils import devices, dump, seed_all, sha256

EXPERIMENTS = {
    "final": dict(recipe="full"),
    "geometry": dict(recipe="geometry"),
    "independent_view": dict(recipe="independent_view"),
    "q4aug": dict(recipe="q4aug"),
    "q4aug_transport": dict(recipe="q4aug_transport"),
    "random_mask": dict(recipe="q4aug_transport_randmask"),
    "stereo_only": dict(recipe="full", train_flow=False),
    "flow_only": dict(recipe="full", train_stereo=False),
    "coupled": dict(recipe="full", lambda_d2=1.0),
}


def experiment_config(name, lambda_d2=None):
    spec = dict(recipe="full", train_stereo=True, train_flow=True, lambda_d2=0.0, nll_weight=.01)
    spec.update(EXPERIMENTS[name])
    if lambda_d2 is not None:
        spec["lambda_d2"] = lambda_d2
    return spec


def main(a):
    spec = experiment_config(a.experiment, a.lambda_d2)
    recipe = augment.RECIPES[spec["recipe"]]
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    S, F = devices()
    seed_all(a.seed)
    torch.set_num_threads(4)
    pairs, val = data.splits(a.data, a.split)

    a.out.mkdir(parents=True)
    dump(a.out / "config.json", dict(
        args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        experiment=spec, recipe=recipe, train_pairs=len(pairs), devices=[S, F],
        stereo_init=str(a.stereo_init), stereo_init_sha256=sha256(a.stereo_init),
        flow_init=str(a.flow_init), flow_init_sha256=sha256(a.flow_init)))

    stereo = models.load_stereo(a.stereo_init, S).requires_grad_(spec["train_stereo"])
    stereo.args.detach_disparity = False
    flow = models.load_flow(a.flow_init, F).requires_grad_(spec["train_flow"])
    flow.detach_flow = False

    groups, min_lr = [], []
    if spec["train_stereo"]:
        mono = [p for n, p in stereo.named_parameters() if n.startswith("defomencoder.")]
        heads = [p for n, p in stereo.named_parameters() if not n.startswith("defomencoder.")]
        groups += [{"params": heads, "lr": a.lr}, {"params": mono, "lr": a.lr * .1}]
        min_lr += [3e-7, 3e-8]
    if spec["train_flow"]:
        groups += [{"params": list(flow.parameters()), "lr": a.flow_lr}]
        min_lr += [1e-7]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=.5, patience=2, threshold=.001, min_lr=min_lr)

    dataset = data.QuadDataset(pairs, a.seed, tuple(a.crop), recipe)
    state = dict(step=0, best=None, bad=0, anchor=None)
    start = time.time()

    def save(name):
        tmp = a.out / (name + ".tmp")
        torch.save(dict(stereo=stereo.state_dict(), flow=flow.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), progress=state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(),
                        python_rng=random.getstate()), tmp)
        tmp.replace(a.out / name)

    def validate(step):
        results, rows = validation.evaluate(stereo, flow, val, a.seed, S, F)
        for condition in validation.CONDITIONS:
            dump(a.out / "evaluations" / str(step) / f"{condition}.json",
                 dict(summary=results[condition], rows=rows[condition]))
        dump(a.out / "evaluations" / str(step) / "summary.json", dict(**results, step=step))
        return results

    total = a.epochs * len(pairs)
    for epoch in range(a.epochs):
        dataset.epoch = epoch
        order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(a.seed + epoch * 1000003)).tolist()
        loader = DataLoader(dataset, batch_size=1, sampler=order, num_workers=a.workers, pin_memory=True)
        for batch in loader:
            tick = time.time()
            stereo.train(spec["train_stereo"])
            stereo.freeze_bn()
            flow.train(spec["train_flow"])
            for module in flow.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
            optimizer.zero_grad(set_to_none=True)

            left = torch.cat((batch["l0"], batch["l1"])).to(S)
            right = torch.cat((batch["r0"], batch["r1"])).to(S)
            gt = torch.cat((batch["d1"], batch["dn"])).to(S)
            gt2, gtf = batch["d2"].to(S), batch["flow"].to(F)
            with torch.set_grad_enabled(spec["train_stereo"]):
                predictions = stereo(left, right, iters=8, scale_iters=4, test_mode=False)
            with torch.set_grad_enabled(spec["train_flow"]):
                out = models.flow_forward(flow, batch["l0"].to(F), batch["l1"].to(F),
                                          gtf if spec["train_flow"] else None)
            fw, native = out["flows"][:, 0], predictions[-1][1:]
            d2, _ = models.warp_field(native, fw.to(S))
            with torch.no_grad():
                # Validity uses the ground-truth correspondence, so a bad flow cannot leave the crop.
                _, bounds = models.warp_field(gt[1:], torch.nan_to_num(gtf.to(S)))
                valid2 = losses.valid_disp(gt2) & bounds & torch.isfinite(gtf.to(S)).all(1, keepdim=True)

            ls = losses.stereo_loss(predictions, gt)
            ld2 = losses.d2_loss(d2, gt2, valid2)
            lf = losses.flow_loss(out, gtf, spec["nll_weight"]) if spec["train_flow"] else torch.zeros((), device=F)
            loss = (ls if spec["train_stereo"] else 0) + (spec["lambda_d2"] * ld2 if spec["lambda_d2"] > 0 else 0) + lf.to(S)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            loss.backward()
            gs = torch.nn.utils.clip_grad_norm_(stereo.parameters(), 1.)
            gf = torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.)
            if not (torch.isfinite(gs) and torch.isfinite(gf)):
                raise FloatingPointError(f"nonfinite gradient norm: stereo={float(gs)} flow={float(gf)}")
            optimizer.step()

            state["step"] += 1
            step = state["step"]
            row = dict(step=step, epoch=step / len(pairs), loss=float(loss.detach()),
                       stereo_loss=float(ls.detach()), d2_loss=float(ld2.detach()), flow_loss=float(lf.detach()),
                       grad_stereo=float(gs), grad_flow=float(gf), augmentation=batch["augmentation"][0],
                       mask=batch["mask"][0], sample=batch["sample"][0], seconds=time.time() - tick)
            with (a.out / "train.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
            del predictions, out, fw, native, d2, loss, ls, ld2, lf, left, right, gt, gt2, gtf, batch
            if step == 1 or step % 100 == 0:
                print(json.dumps(row), flush=True)
            if a.max_steps and step >= a.max_steps:
                dump(a.out / "status.json", dict(phase="stopped", step=step))
                return
            if step % 250 == 0:
                save("latest.pt")
            if step % a.eval_every == 0 or step == total:
                save("latest.pt")
                score = validate(step)["selection_score"]
                if state["best"] is None or score < state["best"]:
                    state["best"] = score
                    save("best.pt")
                    dump(a.out / "best.json", dict(step=step, score=score))
                if state["anchor"] is None or score < state["anchor"] * (1 - .001):
                    state["anchor"], state["bad"] = score, 0
                else:
                    state["bad"] += 1
                scheduler.step(score)
                save("latest.pt")
                print(f"step {step}: J={score:.4f} best={state['best']:.4f}", flush=True)
                torch.cuda.empty_cache()
                if step >= a.min_epochs * len(pairs) and state["bad"] >= a.patience:
                    break
        if state["step"] >= a.min_epochs * len(pairs) and state["bad"] >= a.patience:
            break
    save("latest.pt")
    dump(a.out / "status.json", dict(phase="complete", **state, wall_seconds=time.time() - start))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--experiment", choices=list(EXPERIMENTS), default="final")
    p.add_argument("--lambda-d2", type=float, default=None, help="override the D2 coupling weight")
    p.add_argument("--data", type=Path, default=paths.SPRING)
    p.add_argument("--split", type=Path, default=paths.REPO / "configs/spring_holdout.json")
    p.add_argument("--stereo-init", type=Path, default=paths.DEFOM_INIT)
    p.add_argument("--flow-init", type=Path, default=paths.DPFLOW_INIT)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--min-epochs", type=int, default=3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--eval-every", type=int, default=2012)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--flow-lr", type=float, default=1e-6)
    p.add_argument("--crop", type=int, nargs=2, default=[320, 640])
    p.add_argument("--seed", type=int, default=9610)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=0, help="stop after this many updates (debugging)")
    main(p.parse_args())
