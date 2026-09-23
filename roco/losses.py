import torch


def valid_disp(d):
    """Spring disparity is valid where finite; zero disparity (sky) is valid."""
    return torch.isfinite(d) & (d >= 0)


def masked_mean(value, mask):
    if not torch.isfinite(value).all():
        raise FloatingPointError("nonfinite loss map")
    mask = mask.expand_as(value)
    return value[mask].mean() if mask.any() else value.sum() * 0.


def robust_error(a, b):
    return ((a - torch.nan_to_num(b)).square() + .01).sqrt() - .1


def stereo_loss(predictions, gt):
    """Sequence loss over DEFOM-Stereo iterations; later iterations weigh more."""
    n = len(predictions)
    weights = [.9 ** (15 * (n - 1 - i) / max(1, n - 1)) for i in range(n)]
    valid = valid_disp(gt)
    return sum(w * masked_mean(robust_error(p, gt), valid) for w, p in zip(weights, predictions)) / sum(weights)


def flow_loss(out, gt, nll_weight=.01):
    """Sequence end-point loss over DPFlow predictions plus its likelihood term."""
    valid = torch.isfinite(gt).all(1, keepdim=True)
    losses = []
    for pred in out["flow_preds"]:
        error = ((pred - torch.nan_to_num(gt)).square().sum(1, keepdim=True) + .01).sqrt() - .1
        losses.append(masked_mean(error, valid))
    weights = [.9 ** (len(losses) - 1 - i) for i in range(len(losses))]
    endpoint = sum(w * v for w, v in zip(weights, losses)) / sum(weights)
    nll = [masked_mean(n, valid) for n in out.get("nf_preds", []) if n is not None]
    likelihood = sum(nll) / len(nll) if nll else endpoint * 0
    return endpoint + nll_weight * likelihood


def d2_loss(d2, gt2, valid2):
    return masked_mean(robust_error(d2, gt2), valid2)
