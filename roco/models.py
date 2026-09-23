from argparse import Namespace

import torch
import torch.nn.functional as F

from . import paths  # noqa: F401  (adds DEFOM-Stereo and flow_library to sys.path)
from core.defom_stereo import DEFOMStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402
import ptlflow  # noqa: E402


def defom_args():
    return Namespace(
        dinov2_encoder="vits", idepth_scale=0.5, hidden_dims=[128] * 3,
        corr_implementation="reg", shared_backbone=False, corr_levels=2,
        corr_radius=4, scale_list=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        scale_corr_radius=2, n_downsample=2, context_norm="batch",
        n_gru_layers=3, mixed_precision=False,
    )


def load_stereo(checkpoint, device="cuda"):
    model = DEFOMStereo(defom_args())
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=True)
    return model.to(device).eval()


def load_flow(checkpoint, device="cuda"):
    return ptlflow.get_model("dpflow", ckpt_path=str(checkpoint)).to(device).eval()


def load_pair(checkpoint, stereo_init, flow_init, stereo_device="cuda:0", flow_device="cuda:0"):
    """Build both experts from their public inits and load a trained paired checkpoint."""
    stereo = load_stereo(stereo_init, stereo_device)
    flow = load_flow(flow_init, flow_device)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        stereo.load_state_dict(state["stereo"], strict=True)
        flow.load_state_dict(state["flow"], strict=True)
    return stereo.eval(), flow.eval()


@torch.inference_mode()
def infer_stereo(model, left, right, iters, scale_iters):
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)
    pred = model(left, right, iters=iters, scale_iters=scale_iters, test_mode=True)
    return padder.unpad(pred)[0, 0].float()


def flow_forward(model, first, second, target=None):
    """DPFlow on RGB images in [0, 255]; the model expects BGR in [0, 1]."""
    images = torch.stack((first, second), 1).flip(2).div(255.).contiguous()
    inputs = {"images": images}
    if target is not None:
        inputs.update(flows=torch.nan_to_num(target)[:, None],
                      valids=torch.isfinite(target).all(1, keepdim=True)[:, None].float())
    out = model(inputs)
    # On CUDA OOM DPFlow silently switches to local correlation or returns None.
    if out is None or model.corr_mode != "allpairs":
        raise RuntimeError(f"DPFlow left all-pairs correlation (corr_mode={model.corr_mode})")
    return out


def warp_field(field, flow):
    """Sample a BCHW field at p + flow(p) (bilinear, border padding)."""
    b, _, h, w = flow.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                            torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij")
    x2, y2 = xx[None] + flow[:, 0], yy[None] + flow[:, 1]
    grid = torch.stack((2 * x2 / max(w - 1, 1) - 1, 2 * y2 / max(h - 1, 1) - 1), -1)
    warped = F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)
    valid = (x2 >= 0) & (x2 <= w - 1) & (y2 >= 0) & (y2 <= h - 1)
    return warped, valid[:, None]
