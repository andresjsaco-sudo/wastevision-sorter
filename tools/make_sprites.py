"""Extract waste object cutouts (RGBA sprites) from TrashNet and Household Garbage.

Pipeline: candidate list -> perceptual-hash dedup across sources -> salient-object
matting (IS-Net via rembg, ONNX Runtime on CPU) -> quality filters -> crop to the mask,
resize (longest side 256 px) -> WebP with alpha -> split BY OBJECT into train / val / sim.
The `sim` sprites are used only by the web simulator and never for training, so the
model is always evaluated on objects it has not seen.
"""
import json, random, sys
from collections import defaultdict
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image
from scipy import ndimage

RAW = Path("/root/wastevision/data/raw")
OUT = Path(__file__).resolve().parents[1] / "sprites"
SEED = 7
PER_CLASS_CAND = int(sys.argv[1]) if len(sys.argv) > 1 else 380
SPLIT = {"train": 160, "val": 25, "sim": 40}
CLASSES = ["plastic", "glass", "cardboard", "metal", "paper", "organic", "other"]
MAPPING = {
    ("trashnet", "cardboard"): "cardboard", ("trashnet", "glass"): "glass", ("trashnet", "metal"): "metal",
    ("trashnet", "paper"): "paper", ("trashnet", "plastic"): "plastic", ("trashnet", "trash"): "other",
    ("household", "cardboard"): "cardboard", ("household", "metal"): "metal", ("household", "paper"): "paper",
    ("household", "plastic"): "plastic", ("household", "green-glass"): "glass", ("household", "brown-glass"): "glass",
    ("household", "white-glass"): "glass", ("household", "biological"): "organic",
    ("household", "trash"): "other", ("household", "shoes"): "other", ("household", "clothes"): "other",
}


def candidates(rng):
    by = defaultdict(list)
    for (src, cls), tgt in MAPPING.items():
        fs = sorted(p for p in (RAW / src / cls).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        rng.shuffle(fs)
        by[tgt].append([(p, src, cls) for p in fs])
    out = {}
    for c, pools in by.items():
        # round-robin across sub-sources so no one dominates
        merged, i = [], 0
        while len(merged) < PER_CLASS_CAND * 3 and any(pools):
            pool = pools[i % len(pools)]
            if pool:
                merged.append(pool.pop())
            i += 1
        out[c] = merged
    return out


def quality(alpha):
    """Return cleaned alpha (largest component) or None if the cutout is not usable."""
    h, w = alpha.shape
    m = alpha > 96
    lab, n = ndimage.label(m)
    if n == 0:
        return None
    sizes = ndimage.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    main = lab == k
    area = main.sum()
    if not (0.05 * h * w <= area <= 0.80 * h * w):
        return None
    if sizes.max() < 0.88 * sizes.sum():
        return None
    border = main[0].sum() + main[-1].sum() + main[:, 0].sum() + main[:, -1].sum()
    if border > 0.35 * np.sqrt(area):          # object cut by the frame
        return None
    soft = ((alpha > 40) & (alpha < 215) & ndimage.binary_dilation(main, iterations=2)).sum()
    if soft > 0.22 * area:                      # matting unsure about most of the object
        return None
    ys, xs = np.where(main)
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    if area < 0.33 * bw * bh and min(bw, bh) > 0.5 * max(bw, bh):  # thin/hollow junk
        return None
    return np.where(ndimage.binary_dilation(main, iterations=2), alpha, 0).astype(np.uint8)


def main():
    from rembg import new_session, remove
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = new_session("isnet-general-use", sess_options=so)
    rng = random.Random(SEED)
    cands = candidates(rng)
    seen = np.zeros(0, dtype=np.uint64)
    need = sum(SPLIT.values())
    manifest = []
    for c in CLASSES:
        kept, tried = [], 0
        for p, src, cls in cands[c]:
            if len(kept) >= need or tried >= PER_CLASS_CAND:
                break
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue
            hv = np.uint64(int(str(imagehash.phash(img)), 16))
            if len(seen) and np.unpackbits(np.bitwise_xor(seen, hv).view(np.uint8).reshape(-1, 8), axis=1).sum(1).min() <= 2:
                continue                            # duplicate (e.g. TrashNet copy inside Household)
            seen = np.append(seen, hv)
            tried += 1
            img.thumbnail((640, 640))
            alpha = np.asarray(remove(img, session=sess, only_mask=True))
            a = quality(alpha)
            if a is None:
                continue
            ys, xs = np.where(a > 96)
            box = (max(xs.min() - 3, 0), max(ys.min() - 3, 0), min(xs.max() + 4, img.width), min(ys.max() + 4, img.height))
            rgba = img.convert("RGBA"); rgba.putalpha(Image.fromarray(a))
            rgba = rgba.crop(box)
            rgba.thumbnail((256, 256), Image.LANCZOS)
            kept.append((rgba, src, cls, p.name))
        rng.shuffle(kept)
        i = 0
        for split in ("sim", "val", "train"):
            n = min(SPLIT[split], len(kept) - i)
            d = OUT / split / c; d.mkdir(parents=True, exist_ok=True)
            for j, (rgba, src, cls, name) in enumerate(kept[i:i + n]):
                f = d / f"{c}_{j:03d}.webp"
                rgba.save(f, "WEBP", quality=88, method=4)
                manifest.append(dict(file=str(f.relative_to(OUT)), cls=c, split=split, source=src,
                                     source_class=cls, original=name, w=rgba.width, h=rgba.height))
            i += n
        print(c, "tried", tried, "kept", len(kept), flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=0))


if __name__ == "__main__":
    main()
