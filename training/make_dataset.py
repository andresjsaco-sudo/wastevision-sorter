"""Synthetic instance-segmentation dataset for the sorting line.

Every image is one frame of the virtual camera (640 x 480): object cutouts composited on
the belt texture with the same geometry, sizes, shadows and occlusion rule as the web
simulator. Because we place every object ourselves, the masks are exact.

Label rule (identical to the simulator's ground truth): an object is annotated with its
VISIBLE region if that region is at least 25 % of the object and 300 px.

Splits never share an object: train/val use training/sprites, test uses the sprites that
of the simulator (training/sprites/sim), which the model never sees during training.

    python training/make_dataset.py --out dataset --train 4000 --val 400 --test 400
"""
import argparse, json, math, random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "assets"
SCENE = json.loads((WEB / "scene.json").read_text())
CLASSES = SCENE["classes"]
W, H = SCENE["cam_w"], SCENE["belt_h"]
SH = SCENE["shadow"]
PAD, DOWN = 12, 6


def load_sprites(split):
    d = ROOT / "sprites" / ("sim" if split == "test" else split)
    return {c: sorted((d / c).glob("*.webp")) for c in CLASSES}


_cache = {}


def sprite(path):
    if path not in _cache:
        im = Image.open(path).convert("RGBA")
        a = im.getchannel("A")
        sil = Image.new("L", (im.width + 2 * PAD, im.height + 2 * PAD), 0); sil.paste(a, (PAD, PAD))
        small = sil.resize((max(2, round(sil.width / DOWN)), max(2, round(sil.height / DOWN))), Image.BOX)
        _cache[path] = (im, small.resize(sil.size, Image.BILINEAR))
    return _cache[path]


def place(canvas, layer, cx, cy):
    """Alpha-composite `layer` (RGBA) centred at (cx, cy); returns its alpha in frame coordinates."""
    x0, y0 = int(round(cx - layer.width / 2)), int(round(cy - layer.height / 2))
    if x0 >= W or y0 >= H or x0 + layer.width <= 0 or y0 + layer.height <= 0:
        return np.zeros((H, W), np.uint8)
    canvas.alpha_composite(layer, (max(x0, 0), max(y0, 0)),
                           (max(-x0, 0), max(-y0, 0), min(layer.width, W - x0), min(layer.height, H - y0)))
    full = np.zeros((H, W), np.uint8)
    a = np.asarray(layer.getchannel("A"))
    xs0, ys0, xs1, ys1 = max(x0, 0), max(y0, 0), min(x0 + layer.width, W), min(y0 + layer.height, H)
    if xs1 > xs0 and ys1 > ys0:
        full[ys0:ys1, xs0:xs1] = a[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    return full


def scene(rng, sprites, belt):
    off = rng.randrange(belt.width)
    canvas = Image.new("RGBA", (W, H))
    canvas.paste(belt.crop((off, 0, off + W, H)) if off + W <= belt.width else
                 Image.fromarray(np.roll(np.asarray(belt), -off, axis=1)[:, :W]), (0, 0))
    n = int(math.exp(rng.uniform(math.log(3), math.log(45))))
    classes = [c for c in CLASSES if sprites[c]]
    weights = [SCENE["mix"][c] for c in classes]
    items = []
    for _ in range(n):
        c = rng.choices(classes, weights)[0]
        lo, hi = SCENE["size_px"][c]
        L = rng.uniform(lo, hi); m = L * SCENE["y_margin_frac"]
        items.append(dict(cls=c, path=rng.choice(sprites[c]), L=L, x=rng.uniform(-70, W + 70),
                          y=rng.uniform(m, H - m), rot=rng.uniform(0, 360), z=rng.random()))
    items.sort(key=lambda it: it["z"])
    alphas = []
    for it in items:
        im, shadow = sprite(it["path"])
        k = it["L"] / max(im.size)
        size = (max(2, round(im.width * k)), max(2, round(im.height * k)))
        ssize = (max(2, round(shadow.width * k)), max(2, round(shadow.height * k)))
        sh = Image.new("RGBA", ssize, (0, 0, 0, 0))
        sh.putalpha(shadow.resize(ssize, Image.BILINEAR).point(lambda v: int(v * SH["alpha"])))
        place(canvas, sh.rotate(-it["rot"], Image.BICUBIC, expand=True), it["x"] + SH["dx"], it["y"] + SH["dy"])
        layer = im.resize(size, Image.BILINEAR).rotate(-it["rot"], Image.BICUBIC, expand=True)
        it["full"] = int((np.asarray(layer.getchannel("A")) > 127).sum())
        alphas.append(place(canvas, layer, it["x"], it["y"]) > 127)
    labels = []
    covered = np.zeros((H, W), bool)
    for it, a in zip(reversed(items), reversed(alphas)):          # top-most first
        vis = a & ~covered
        covered |= a
        area = int(vis.sum())
        if area < SCENE["min_visible_px"] or area < SCENE["min_visible_frac"] * it["full"]:
            continue
        cs, _ = cv2.findContours(vis.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cs, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 0.6 * SCENE["min_visible_px"]:
            continue
        poly = cv2.approxPolyDP(cnt, 1.0, True).reshape(-1, 2)
        if len(poly) >= 3:
            labels.append((CLASSES.index(it["cls"]), poly))
    img = canvas.convert("RGB")
    # camera variability: exposure, slight defocus and compression
    g = rng.uniform(0.88, 1.12); b = rng.uniform(-8, 8)
    img = Image.fromarray(np.clip(np.asarray(img, np.float32) * g + b, 0, 255).astype(np.uint8))
    if rng.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--test", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    belt = Image.open(WEB / "belt.jpg").convert("RGB")
    for i, (split, n) in enumerate([("train", args.train), ("val", args.val), ("test", args.test)]):
        rng = random.Random(args.seed * 100 + i)
        sprites = load_sprites(split)
        (out / "images" / split).mkdir(parents=True, exist_ok=True); (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        inst = 0
        for j in range(n):
            img, labels = scene(rng, sprites, belt)
            img.save(out / "images" / split / f"{split}_{j:05d}.jpg", quality=rng.randint(82, 95))
            with open(out / "labels" / split / f"{split}_{j:05d}.txt", "w") as f:
                for c, poly in labels:
                    f.write(f"{c} " + " ".join(f"{x / W:.5f} {y / H:.5f}" for x, y in poly) + "\n")
            inst += len(labels)
        print(f"{split}: {n} images, {inst} instances, {sum(len(v) for v in sprites.values())} sprites", flush=True)
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")


if __name__ == "__main__":
    main()
