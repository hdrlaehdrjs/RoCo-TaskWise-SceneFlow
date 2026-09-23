"""Spring quadruplets (L_t, R_t, L_t+1, R_t+1) with disparity and flow ground truth.

Images are 1080p while Spring ground truth is stored at 2160p, so ground truth is
read with a stride of 2 and all quantities are in 1080p pixel units.
"""
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import paths  # noqa: F401
import flow_IO  # noqa: E402

IMAGES = ("l0", "r0", "l1", "r1")


@dataclass(frozen=True)
class SpringPair:
    scene: Path
    frame: int


def build_pairs(root, excluded=()):
    """All forward pairs in root/train whose images and ground truth exist."""
    pairs = []
    for scene in sorted((Path(root) / "train").iterdir()):
        if not scene.is_dir() or scene.name in excluded:
            continue
        for path in sorted((scene / "flow_FW_left").glob("*.flo5")):
            f = int(path.stem.rsplit("_", 1)[-1])
            required = [scene / "frame_left" / f"frame_left_{f:04d}.png",
                        scene / "frame_right" / f"frame_right_{f:04d}.png",
                        scene / "frame_left" / f"frame_left_{f + 1:04d}.png",
                        scene / "frame_right" / f"frame_right_{f + 1:04d}.png",
                        scene / "disp1_left" / f"disp1_left_{f:04d}.dsp5",
                        scene / "disp1_left" / f"disp1_left_{f + 1:04d}.dsp5",
                        scene / "disp2_FW_left" / f"disp2_FW_left_{f:04d}.dsp5"]
            if all(p.exists() for p in required):
                pairs.append(SpringPair(scene, f))
    if not pairs:
        raise RuntimeError(f"no Spring pairs found under {root}")
    return pairs


def validation_pairs(root, scenes, per_scene):
    """per_scene evenly spaced forward pairs from each held-out scene."""
    pairs = []
    for name in scenes:
        scene = Path(root) / "train" / name
        flows = sorted((scene / "flow_FW_left").glob("*.flo5"))
        for i in np.linspace(0, len(flows) - 1, min(per_scene, len(flows)), dtype=int):
            pairs.append(SpringPair(scene, int(flows[i].stem.rsplit("_", 1)[-1])))
    return pairs


def splits(root, config):
    split = json.loads(Path(config).read_text())
    held_out = set(split["validation_scenes"])
    train = build_pairs(root, held_out)
    val = validation_pairs(root, split["validation_scenes"], split["samples_per_scene"])
    assert not ({p.scene.name for p in train} & {p.scene.name for p in val})
    return train, val


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()


def read_disp(path):
    return flow_IO.readDispFile(str(path)).astype(np.float32)[::2, ::2]


def read_flow(path):
    return flow_IO.readFlowFile(str(path)).astype(np.float32)[::2, ::2]


def read_pair(pair):
    s, f = pair.scene, pair.frame
    return {
        "l0": read_rgb(s / "frame_left" / f"frame_left_{f:04d}.png"),
        "r0": read_rgb(s / "frame_right" / f"frame_right_{f:04d}.png"),
        "l1": read_rgb(s / "frame_left" / f"frame_left_{f + 1:04d}.png"),
        "r1": read_rgb(s / "frame_right" / f"frame_right_{f + 1:04d}.png"),
        "d1": read_disp(s / "disp1_left" / f"disp1_left_{f:04d}.dsp5"),
        "dn": read_disp(s / "disp1_left" / f"disp1_left_{f + 1:04d}.dsp5"),
        "d2": read_disp(s / "disp2_FW_left" / f"disp2_FW_left_{f:04d}.dsp5"),
        "flow": read_flow(s / "flow_FW_left" / f"flow_FW_left_{f:04d}.flo5"),
    }


def to_tensors(item, device):
    out = {}
    for key in IMAGES:
        out[key] = torch.from_numpy(item[key]).permute(2, 0, 1).float()[None].to(device)
    for key in ("d1", "dn", "d2"):
        out[key] = torch.from_numpy(item[key]).float()[None, None].to(device)
    out["flow"] = torch.from_numpy(item["flow"]).permute(2, 0, 1).float()[None].to(device)
    return out


def resize_crop(item, crop, rng):
    """Shared log-uniform rescale in [0.9, 1.15] and random crop for all views and labels."""
    h, w = item["d1"].shape
    scale = math.exp(rng.uniform(math.log(.9), math.log(1.15)))
    nh, nw = max(crop[0], round(h * scale)), max(crop[1], round(w * scale))
    sy, sx = nh / h, nw / w
    y, x = rng.randrange(nh - crop[0] + 1), rng.randrange(nw - crop[1] + 1)
    out = {}
    for k, a in item.items():
        b = cv2.resize(a, (nw, nh), interpolation=cv2.INTER_LINEAR if k in IMAGES else cv2.INTER_NEAREST)
        if k == "flow":
            b = b * np.array([sx, sy], dtype=np.float32)
        elif k not in IMAGES:
            b = b * sx
        out[k] = np.ascontiguousarray(b[y:y + crop[0], x:x + crop[1]])
    return out


class QuadDataset(Dataset):
    """Training quadruplets; each sample is seeded from (seed, epoch, index)."""

    def __init__(self, pairs, seed, crop, recipe):
        self.pairs, self.seed, self.crop, self.recipe, self.epoch = pairs, seed, crop, recipe, 0

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        from .augment import augment
        cv2.setNumThreads(0)
        seed = self.seed + self.epoch * 1000003 + i
        rng, nr = random.Random(seed), np.random.default_rng(seed)
        clean = resize_crop(read_pair(self.pairs[i]), self.crop, rng)
        aug, label, mask = augment(clean, rng, nr, self.recipe)
        t = {k: torch.from_numpy(aug[k]).permute(2, 0, 1).float() for k in IMAGES}
        for k in ("d1", "dn", "d2"):
            t[k] = torch.from_numpy(clean[k])[None]
        t["flow"] = torch.from_numpy(clean["flow"]).permute(2, 0, 1)
        t["augmentation"], t["mask"] = label, mask
        t["sample"] = f"{self.pairs[i].scene.name}:{self.pairs[i].frame:04d}"
        return t
