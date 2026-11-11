// Physical world of the simulation: the belt, the objects lying on it and what a
// suction cup would touch at a given point. No DOM access, so it also runs in Node.
import { G } from './config.js';

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function decodeMask(b64, n) {
  const bin = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
  const bits = new Uint8Array(n * n);
  for (let i = 0; i < n * n; i++) bits[i] = (bin.charCodeAt(i >> 3) >> (7 - (i & 7))) & 1;
  return bits;
}

export class World {
  constructor(scene, sprites, seed) {
    this.scene = scene;
    this.sprites = sprites;              // [{cls, w, h, bits(Uint8Array n*n)}]
    this.n = scene.mask_grid;
    this.rng = mulberry32(seed);
    this.byClass = {};
    sprites.forEach((sp, i) => (this.byClass[sp.cls] ??= []).push(i));
    this.classes = scene.classes.filter((c) => this.byClass[c]?.length);
    this.t = 0; this.s = 0;              // time, belt travel (px)
    this.items = []; this.nextId = 1;
    this.toSpawn = 0.3;
    this.exited = [];                    // items that fell off the end this step
  }

  x(it) { return it.u + this.s; }

  spawn() {
    const r = this.rng(); let acc = 0, cls = this.classes[0];
    const tot = this.classes.reduce((a, c) => a + this.scene.mix[c], 0);
    for (const c of this.classes) { acc += this.scene.mix[c] / tot; if (r <= acc) { cls = c; break; } }
    const pool = this.byClass[cls];
    const sp = pool[Math.floor(this.rng() * pool.length)];
    const [lo, hi] = this.scene.size_px[cls];
    const L = lo + this.rng() * (hi - lo);
    const m = L * this.scene.y_margin_frac;
    const it = {
      id: this.nextId++, cls, sp, L,
      k: L / Math.max(this.sprites[sp].w, this.sprites[sp].h),
      u: -L * 0.75 - this.s, y: m + this.rng() * (G.BELT_H - 2 * m),
      rot: this.rng() * Math.PI * 2, z: this.rng(), state: 'belt', born: this.t,
    };
    it.cos = Math.cos(it.rot); it.sin = Math.sin(it.rot);
    it.hw = this.sprites[sp].w * it.k / 2; it.hh = this.sprites[sp].h * it.k / 2;
    it.r = Math.hypot(it.hw, it.hh);
    this.items.push(it);
    return it;
  }

  step(dt, vb, rate) {
    this.t += dt; this.s += vb * dt;
    this.toSpawn -= dt;
    while (this.toSpawn <= 0) {
      this.spawn();
      this.toSpawn += -Math.log(1 - this.rng()) / Math.max(rate, 1e-3);
    }
    this.exited.length = 0;
    this.items = this.items.filter((it) => {
      if (it.state === 'belt' && this.x(it) - it.r > G.W) { this.exited.push(it); return false; }
      if (it.state === 'gone') return false;
      return true;
    });
  }

  // Is the object opaque at world point (wx, wy)? wy is measured from the belt's top edge.
  alphaAt(it, wx, wy) {
    const dx = wx - this.x(it), dy = wy - it.y;
    if (dx * dx + dy * dy > it.r * it.r) return false;
    const lx = dx * it.cos + dy * it.sin, ly = -dx * it.sin + dy * it.cos;
    const a = lx / (2 * it.hw) + 0.5, b = ly / (2 * it.hh) + 0.5;
    if (a < 0 || a >= 1 || b < 0 || b >= 1) return false;
    return this.sprites[it.sp].bits[Math.floor(b * this.n) * this.n + Math.floor(a * this.n)] === 1;
  }

  // The object a suction cup would grab at this point: the top-most opaque one.
  topItemAt(wx, wy) {
    let best = null;
    for (const it of this.items) {
      if (it.state !== 'belt' || (best && it.z < best.z)) continue;
      if (this.alphaAt(it, wx, wy)) best = it;
    }
    return best;
  }

  // Ground truth of what an overhead camera can see inside [x0, x0+w] x [0, BELT_H]:
  // for every object, the area not covered by objects above it. Used as the "oracle"
  // perception, to label training scenes with the same rule, and to score the model.
  visible(x0, w) {
    const n = this.n, out = [];
    const cand = this.items.filter((it) => it.state === 'belt' && this.x(it) + it.r > x0 && this.x(it) - it.r < x0 + w);
    for (const it of cand) {
      const above = cand.filter((o) => o !== it && o.z > it.z &&
        Math.hypot(this.x(o) - this.x(it), o.y - it.y) < o.r + it.r);
      const bits = this.sprites[it.sp].bits, cx = this.x(it);
      const cell = (2 * it.hw / n) * (2 * it.hh / n) * 4;      // sampled every 2nd cell
      let full = 0, vis = 0, sx = 0, sy = 0, bx0 = 1e9, by0 = 1e9, bx1 = -1e9, by1 = -1e9;
      const pts = [];
      for (let j = 1; j < n; j += 2) for (let i = 1; i < n; i += 2) {
        if (!bits[j * n + i]) continue;
        full++;
        const lx = ((i + 0.5) / n - 0.5) * 2 * it.hw, ly = ((j + 0.5) / n - 0.5) * 2 * it.hh;
        const wx = cx + lx * it.cos - ly * it.sin, wy = it.y + lx * it.sin + ly * it.cos;
        if (wx < x0 || wx >= x0 + w || wy < 0 || wy >= G.BELT_H) continue;
        let covered = false;
        for (const o of above) if (this.alphaAt(o, wx, wy)) { covered = true; break; }
        if (covered) continue;
        vis++; sx += wx; sy += wy; pts.push(wx, wy);
        if (wx < bx0) bx0 = wx; if (wx > bx1) bx1 = wx; if (wy < by0) by0 = wy; if (wy > by1) by1 = wy;
      }
      if (!vis) continue;
      const frac = vis / full, area = vis * cell;
      if (frac < this.scene.min_visible_frac || area < this.scene.min_visible_px) continue;
      // grasp point: visible sample closest to the centroid (always on the object)
      const mx = sx / vis, my = sy / vis; let gx = mx, gy = my, bd = 1e18;
      for (let q = 0; q < pts.length; q += 2) {
        const d = (pts[q] - mx) ** 2 + (pts[q + 1] - my) ** 2;
        if (d < bd) { bd = d; gx = pts[q]; gy = pts[q + 1]; }
      }
      out.push({ item: it, cls: it.cls, x: gx - x0, y: gy, bx: bx0 - x0, by: by0, bw: bx1 - bx0, bh: by1 - by0, area, frac, conf: 1 });
    }
    return out;
  }
}
