"""Q4Aug appearance corruption, GT-guided transport of spatial corruption, and CorrMask.

A recipe is a dict with keys
  appearance: apply Q4Aug with probability 0.7
  shared:     one photometric/corruption setting for all four views (False: per view)
  transport:  move spatial masks to the other views with ground-truth disparity and flow
  mask:       CorrMask placement: "none", "priority", "random", "world" or "single"
"""
import cv2
import numpy as np

from .data import IMAGES

KINDS = ("blur", "noise", "jpeg", "pixelate", "rain", "snow", "spatter")
WEATHER = ("rain", "snow", "spatter")

RECIPES = {
    "geometry": dict(appearance=False, shared=True, transport=False, mask="none"),
    "independent_view": dict(appearance=True, shared=False, transport=False, mask="none"),
    "q4aug": dict(appearance=True, shared=True, transport=False, mask="none"),
    "q4aug_transport": dict(appearance=True, shared=True, transport=True, mask="none"),
    "q4aug_transport_randmask": dict(appearance=True, shared=True, transport=True, mask="random"),
    "full": dict(appearance=True, shared=True, transport=True, mask="priority"),
}


def photo_params(rng):
    return (rng.uniform(.75, 1.25), rng.uniform(.75, 1.25),
            rng.uniform(.7, 1.3), rng.uniform(.8, 1.2), rng.random() < .04)


def photo(image, p):
    brightness, contrast, saturation, gamma, gray_flag = p
    x = image.astype(np.float32) / 255
    mean = x.mean(axis=(0, 1), keepdims=True)
    x = (x - mean) * contrast + mean
    x *= brightness
    gray = x.mean(axis=2, keepdims=True)
    x = gray + saturation * (x - gray)
    x = np.clip(x, 0, 1) ** gamma
    if gray_flag:
        x = np.repeat(x.mean(axis=2, keepdims=True), 3, axis=2)
    return np.clip(255 * x, 0, 255).astype(np.uint8)


def degrade(image, kind, severity, nr):
    if kind == "blur":
        return cv2.GaussianBlur(image, (0, 0), .25 + 1.6 * severity)
    if kind == "noise":
        noise = nr.normal(0, 2 + 11 * severity, image.shape)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if kind == "jpeg":
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(92 - 57 * severity)])
        if ok:
            return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if kind == "pixelate":
        h, w = image.shape[:2]
        factor = .9 - .5 * severity
        small = cv2.resize(image, (max(2, int(w * factor)), max(2, int(h * factor))),
                           interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return image


def weather_mask(shape, kind, severity, rng, nr):
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    if kind == "rain":
        for _ in range(int((70 + 180 * severity) * h * w / (256 * 512))):
            x, y, length = rng.randrange(w), rng.randrange(h), rng.randint(8, 28)
            cv2.line(mask, (x, y), (min(w - 1, x + length // 4), min(h - 1, y + length)),
                     rng.uniform(.5, 1), rng.randint(1, 2))
        mask = cv2.GaussianBlur(mask, (3, 3), .7)
    elif kind == "snow":
        count = int((150 + 500 * severity) * h * w / (256 * 512))
        ys, xs = nr.integers(0, h, count), nr.integers(0, w, count)
        mask[ys, xs] = nr.uniform(.6, 1, count)
        mask = np.clip(cv2.GaussianBlur(mask, (0, 0), .7 + severity) * 3, 0, 1)
    else:
        for _ in range(int(8 + 28 * severity)):
            cv2.circle(mask, (rng.randrange(w), rng.randrange(h)), rng.randint(5, 24),
                       rng.uniform(.35, .9), -1)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
    return np.clip(mask, 0, 1)


def forward_splat_mask(mask, dx, dy):
    """Nearest-pixel forward splat with max aggregation and 3x3 dilation to close holes."""
    h, w = mask.shape
    yy, xx = np.mgrid[:h, :w]
    tx = np.rint(xx + np.nan_to_num(dx)).astype(np.int64)
    ty = np.rint(yy + np.nan_to_num(dy)).astype(np.int64)
    valid = (tx >= 0) & (tx < w) & (ty >= 0) & (ty < h) & (mask > 0)
    result = np.zeros(h * w, np.float32)
    np.maximum.at(result, (ty[valid] * w + tx[valid]), mask[valid])
    return cv2.dilate(result.reshape(h, w), np.ones((3, 3), np.uint8))


def overlay(image, mask, nr, white=True):
    alpha = (.8 * np.clip(mask, 0, 1))[..., None]
    texture = np.full_like(image, 245, dtype=np.float32) if white else nr.uniform(30, 225, image.shape).astype(np.float32)
    return np.clip(image * (1 - alpha) + texture * alpha, 0, 255).astype(np.uint8)


def transport(mask, item):
    """The L_t mask placed in L_t, R_t, L_t+1 and R_t+1 through ground-truth correspondence."""
    return (mask,
            forward_splat_mask(mask, -item["d1"], np.zeros_like(mask)),
            forward_splat_mask(mask, item["flow"][..., 0], item["flow"][..., 1]),
            forward_splat_mask(mask, item["flow"][..., 0] - item["d2"], item["flow"][..., 1]))


def mask_box(h, w, y, x, rng):
    mh = rng.randint(max(16, h // 12), max(24, h // 4))
    mw = rng.randint(max(24, w // 12), max(32, w // 4))
    m = np.zeros((h, w), np.float32)
    m[max(0, y - mh // 2):min(h, y + mh // 2), max(0, x - mw // 2):min(w, x + mw // 2)] = 1
    return cv2.GaussianBlur(m, (0, 0), 1.2)


def priority_mask(item, rng):
    """CorrMask centred on the highest-priority pixel (disparity edges, motion, image gradients)."""
    safe = np.nan_to_num(item["d1"])
    edge = np.hypot(cv2.Sobel(safe, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(safe, cv2.CV_32F, 0, 1, ksize=3))
    gray = cv2.cvtColor(item["l0"], cv2.COLOR_RGB2GRAY).astype(np.float32)
    detail = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    motion = np.linalg.norm(np.nan_to_num(item["flow"]), axis=2)

    def norm(x):
        return np.clip(x / max(float(np.percentile(x[np.isfinite(x)], 95)), 1e-6), 0, 1)

    score = .45 * norm(edge) + .30 * norm(motion) + .25 * norm(detail)
    flat = score.ravel() + np.asarray([rng.random() * .05 for _ in range(score.size)], np.float32)
    y, x = np.unravel_index(int(flat.argmax()), score.shape)
    return mask_box(*score.shape, y, x, rng)


def random_mask(item, rng):
    """Same size and blending as CorrMask, uniformly random centre."""
    h, w = item["d1"].shape
    y, x = rng.randrange(h), rng.randrange(w)
    return mask_box(h, w, y, x, rng)


def augment(item, rng, nr, recipe):
    """Returns the augmented images, an augmentation label and a mask label."""
    out = {k: v.copy() for k, v in item.items()}
    label, mask_label = "clean", "none"
    if not recipe["appearance"]:
        return out, label, mask_label

    if not recipe["shared"]:
        if rng.random() < .7:
            kinds = []
            for k in IMAGES:
                out[k] = photo(out[k], photo_params(rng))
                kind, severity = rng.choice(KINDS), rng.uniform(.15, .75)
                if kind in WEATHER:
                    out[k] = overlay(out[k], weather_mask(item["d1"].shape, kind, severity, rng, nr), nr, white=True)
                else:
                    out[k] = degrade(out[k], kind, severity, nr)
                kinds.append(kind)
            label = "q4indep:" + "+".join(kinds)
        return out, label, mask_label

    def paint(masks, white):
        for k, m in zip(IMAGES, masks):
            out[k] = overlay(out[k], m, nr, white=white)

    if rng.random() < .7:
        p = photo_params(rng)
        for k in IMAGES:
            out[k] = photo(out[k], p)
        kind = rng.choice(KINDS)
        severity = rng.uniform(.15, .75)
        label = "q4:" + kind
        if kind in WEATHER:
            m = weather_mask(item["d1"].shape, kind, severity, rng, nr)
            paint(transport(m, item) if recipe["transport"] else (m, m, m, m), True)
        else:
            for k in IMAGES:
                out[k] = degrade(out[k], kind, severity, nr)
        mode = recipe["mask"]
        if mode != "none" and rng.random() < .25:
            m = random_mask(item, rng) if mode == "random" else priority_mask(item, rng)
            world = mode == "world" or (mode in ("priority", "random") and rng.random() < .55)
            if world:
                paint(transport(m, item), False)
                mask_label = "world"
            else:
                k = rng.choice(("r0", "r1", "l1"))
                out[k] = overlay(out[k], m, nr, white=False)
                mask_label = "sensor:" + k
    return out, label, mask_label
