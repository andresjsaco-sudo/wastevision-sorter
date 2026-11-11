"""Prepare the simulator assets: belt texture (tileable along the direction of travel) and
sprites.json with the 48x48 occupancy mask of every `sim` sprite (used for physics,
occlusion and ground truth in the browser and in Node)."""
import base64, json, shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "assets"
SCENE = json.loads((WEB / "scene.json").read_text())
N = SCENE["mask_grid"]
CELL = 256


def belt_texture(w=1280, h=480, seed=3):
    rng = np.random.default_rng(seed)
    base = np.array([47, 49, 54], np.float32)
    fine = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), 0.7, mode="wrap") * 5.5
    blot = ndimage.gaussian_filter(rng.normal(0, 1, (h, w)), (28, 46), mode="wrap") * 95
    streak = ndimage.gaussian_filter1d(rng.normal(0, 1, h), 2.5)[:, None] * 3.2     # wear along travel
    img = fine + blot + streak
    for x0 in range(0, w, 160):                                                      # moulded ribs
        img[:, x0:x0 + 2] -= 9; img[:, x0 + 2:x0 + 4] += 5
    dust = ndimage.gaussian_filter((rng.random((h, w)) > 0.9985).astype(np.float32), 1.1, mode="wrap") * 260
    rgb = base[None, None] + img[..., None] * np.array([1.0, 1.0, 1.04]) + dust[..., None]
    edge = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
    rgb *= (0.86 + 0.14 * np.clip(edge / 26, 0, 1))[:, None, None]                    # darker near the rails
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def main():
    belt_texture().save(WEB / "belt.jpg", quality=90)
    shutil.rmtree(WEB / "sprites", ignore_errors=True)
    bl = ROOT / "tools" / "sim_blocklist.txt"
    drop = set(bl.read_text().split()) if bl.exists() else set()
    files = [f for f in sorted((ROOT / "sprites" / "sim").glob("*/*.webp")) if f.name not in drop]
    cols = int(np.ceil(np.sqrt(len(files)))); rows = int(np.ceil(len(files) / cols))
    atlas = Image.new("RGBA", (cols * CELL, rows * CELL), (0, 0, 0, 0))
    sprites = []
    for i, src in enumerate(files):
        im = Image.open(src).convert("RGBA")
        ax, ay = (i % cols) * CELL, (i // cols) * CELL
        atlas.paste(im, (ax, ay))
        a = np.asarray(im)[..., 3] > 96
        small = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((N, N), Image.BOX)) > 127
        sprites.append(dict(cls=src.parent.name, w=im.width, h=im.height, ax=ax, ay=ay,
                            mask=base64.b64encode(np.packbits(small.reshape(-1)).tobytes()).decode()))
    atlas.save(WEB / "atlas.webp", "WEBP", quality=90, method=5)
    (WEB / "sprites.json").write_text(json.dumps(sprites))
    from collections import Counter
    print(Counter(s["cls"] for s in sprites))


if __name__ == "__main__":
    main()
