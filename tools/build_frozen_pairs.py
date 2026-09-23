"""Frame-disjoint analysis pairs: up to 14 evenly spaced pairs per held-out scene,
excluding the pairs used for checkpoint selection (configs/frozen_pairs.txt)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roco import data, paths  # noqa: E402

PER_SCENE = 14


def valid_frames(root, scene_name):
    scene = root / "train" / scene_name
    frames = []
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
            frames.append(f)
    return frames


def main(a):
    split = json.loads(a.split.read_text())
    used = {}
    for pair in data.validation_pairs(a.data, split["validation_scenes"], split["samples_per_scene"]):
        used.setdefault(pair.scene.name, set()).add(pair.frame)
    lines = []
    for name in split["validation_scenes"]:
        remaining = [f for f in valid_frames(a.data, name) if f not in used[name]]
        picked = [remaining[i] for i in np.linspace(0, len(remaining) - 1, min(PER_SCENE, len(remaining)), dtype=int)]
        lines += [f"{name} {f:04d}" for f in picked]
        print(f"{name}: {len(remaining)} remaining, {len(picked)} selected")
    a.out.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=paths.SPRING)
    p.add_argument("--split", type=Path, default=paths.REPO / "configs/spring_holdout.json")
    p.add_argument("--out", type=Path, default=paths.REPO / "configs/frozen_pairs.txt")
    main(p.parse_args())
