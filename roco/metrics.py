import torch

from .losses import valid_disp


def metric_sums(output, gt):
    """Pixel counts and error sums for D1, D2 and flow; pooled over pairs by `metrics`."""
    d1, d2, flow = output
    if not all(torch.isfinite(p).all() for p in output):
        raise FloatingPointError("nonfinite prediction")
    v1, v2 = valid_disp(gt["d1"]), valid_disp(gt["d2"])
    vf = torch.isfinite(gt["flow"]).all(1, keepdim=True)
    va = v1 & v2 & vf
    e1, e2 = (d1 - gt["d1"]).abs(), (d2 - gt["d2"]).abs()
    ef = (flow - gt["flow"]).norm(dim=1, keepdim=True)
    b1, b2, bf = e1 > 1, e2 > 1, ef > 1
    o1 = (e1 > 3) & (e1 > .05 * gt["d1"].abs())
    o2 = (e2 > 3) & (e2 > .05 * gt["d2"].abs())
    of = (ef > 3) & (ef > .05 * gt["flow"].norm(dim=1, keepdim=True))
    return {k: float(v) for k, v in dict(
        n1=v1.sum(), n2=v2.sum(), nf=vf.sum(), na=va.sum(),
        e1=e1[v1].sum(), e2=e2[v2].sum(), ef=ef[vf].sum(),
        b1=(b1 & v1).sum(), b2=(b2 & v2).sum(), bf=(bf & vf).sum(),
        ba=((b1 | b2 | bf) & va).sum(), outlier=((o1 | o2 | of) & va).sum()).items()}


def metrics(s):
    def ratio(a, b, scale=1):
        return scale * s[a] / s[b] if s[b] else None
    return {"d1_abs": ratio("e1", "n1"), "d2_abs": ratio("e2", "n2"),
            "flow_epe": ratio("ef", "nf"), "d1_1px": ratio("b1", "n1", 100),
            "d2_1px": ratio("b2", "n2", 100), "flow_1px": ratio("bf", "nf", 100),
            "sf_1px": ratio("ba", "na", 100), "sf_outlier": ratio("outlier", "na", 100)}
