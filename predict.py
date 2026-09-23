"""Scene-flow predictions for a Spring split in the benchmark's file layout.

For every scene: left and right disparity at every frame, forward and backward flow
for both cameras, and d2 obtained by sampling the target-frame disparity at p + f(p).
The output folder is what the official Spring subsampling tools expect.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from roco import models, paths
import flow_IO  # noqa: E402  (on sys.path through roco.paths)


def image_tensor(path, device):
    array = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float()[None].to(device)


def right_disparity(model, left, right, iters, scale_iters):
    pred = models.infer_stereo(model, torch.flip(right, [-1]), torch.flip(left, [-1]), iters, scale_iters)
    return torch.flip(pred, [-1])


def write_disp(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    flow_IO.writeDsp5File(value.detach().cpu().numpy().astype(np.float32), str(path))


def write_flow(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    flow_IO.writeFlowFile(value.detach().permute(1, 2, 0).cpu().numpy().astype(np.float32), str(path))


def output_paths(root, scene, side, direction, frame):
    return {"disp1": root / scene / f"disp1_{side}" / f"disp1_{side}_{frame:04d}.dsp5",
            "disp2": root / scene / f"disp2_{direction}_{side}" / f"disp2_{direction}_{side}_{frame:04d}.dsp5",
            "flow": root / scene / f"flow_{direction}_{side}" / f"flow_{direction}_{side}_{frame:04d}.flo5"}


def frames_for(scene):
    return [int(p.stem.rsplit("_", 1)[-1]) for p in sorted((scene / "frame_left").glob("*.png"))]


@torch.inference_mode()
def predict(a):
    device = "cuda:0"
    scenes = [p for p in sorted(a.images.iterdir()) if p.is_dir() and (p / "frame_left").is_dir()]
    stereo, flow = models.load_pair(a.checkpoint, a.stereo_init, a.flow_init, device, device)
    started = time.time()

    for scene in scenes:
        for frame in frames_for(scene):
            left_path = output_paths(a.out, scene.name, "left", "FW", frame)["disp1"]
            right_path = output_paths(a.out, scene.name, "right", "FW", frame)["disp1"]
            if left_path.exists() and right_path.exists():
                continue
            left = image_tensor(scene / "frame_left" / f"frame_left_{frame:04d}.png", device)
            right = image_tensor(scene / "frame_right" / f"frame_right_{frame:04d}.png", device)
            if not left_path.exists():
                write_disp(left_path, models.infer_stereo(stereo, left, right, a.iters, a.scale_iters))
            if not right_path.exists():
                write_disp(right_path, right_disparity(stereo, left, right, a.iters, a.scale_iters))
        print(f"{scene.name}: disparity done ({time.time() - started:.0f}s)", flush=True)

    for scene in scenes:
        frames = frames_for(scene)
        for frame0, frame1 in zip(frames[:-1], frames[1:]):
            images = {}
            for side in ("left", "right"):
                images[(side, 0)] = image_tensor(scene / f"frame_{side}" / f"frame_{side}_{frame0:04d}.png", device)
                images[(side, 1)] = image_tensor(scene / f"frame_{side}" / f"frame_{side}_{frame1:04d}.png", device)
                for direction, source, target, output_frame in (("FW", 0, 1, frame0), ("BW", 1, 0, frame1)):
                    out = output_paths(a.out, scene.name, side, direction, output_frame)
                    if out["flow"].exists():
                        f = torch.from_numpy(flow_IO.readFlowFile(str(out["flow"]))).permute(2, 0, 1).to(device).float()
                    else:
                        f = models.flow_forward(flow, images[(side, source)], images[(side, target)])["flows"][0, 0].float()
                        write_flow(out["flow"], f)
                    if not out["disp2"].exists():
                        target_frame = frame1 if direction == "FW" else frame0
                        native = flow_IO.readDispFile(str(output_paths(a.out, scene.name, side, "FW", target_frame)["disp1"]))
                        native = torch.from_numpy(native).to(device).float()
                        d2, _ = models.warp_field(native[None, None], f[None])
                        write_disp(out["disp2"], d2[0, 0])
        print(f"{scene.name}: flow and d2 done ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True, help="e.g. spring/test or a corrupted copy of it")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="best.pt written by train.py")
    p.add_argument("--stereo-init", type=Path, default=paths.DEFOM_INIT)
    p.add_argument("--flow-init", type=Path, default=paths.DPFLOW_INIT)
    p.add_argument("--iters", type=int, default=16)
    p.add_argument("--scale-iters", type=int, default=6)
    predict(p.parse_args())
