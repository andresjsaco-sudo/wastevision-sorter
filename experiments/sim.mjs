// Headless run of the same world / tracker / robot code used by the web page.
import fs from 'node:fs';
import { G, CLASSES } from '../web/js/config.js';
import { World, decodeMask, mulberry32 } from '../web/js/world.js';
import { Tracker } from '../web/js/tracker.js';
import { RobotCell, Metrics } from '../web/js/control.js';

const A = new URL('../web/assets/', import.meta.url);
export const scene = JSON.parse(fs.readFileSync(new URL('scene.json', A)));
export const sprites = JSON.parse(fs.readFileSync(new URL('sprites.json', A)));
for (const s of sprites) s.bits = decodeMask(s.mask, scene.mask_grid);

// perception: null = oracle; otherwise { miss, confusion: {true: {pred: p}}, conf: [lo, hi], hz }
export function simulate(o) {
  const P = { beltSpeed: 0.3, rate: 2, armsPerStation: 1, policy: 'edf', confThr: 0.5, gripFail: 0.03, seed: 1, duration: 300, warmup: 20, perception: null, ...o };
  const world = new World(scene, sprites, P.seed), tracker = new Tracker(), cell = new RobotCell(P.armsPerStation, P.seed);
  let metrics = new Metrics(CLASSES);
  const rng = mulberry32(P.seed + 77), vb = P.beltSpeed * G.PXM, dt = 1 / 60, period = 1 / (P.perception?.hz ?? 5);
  let last = -1, started = false;
  for (let t = 0; t < P.warmup + P.duration; t += dt) {
    world.step(dt, vb, P.rate);
    if (!started && world.t >= P.warmup) { metrics = new Metrics(CLASSES); started = true; }
    for (const it of world.exited) metrics.onExit(it);
    if (world.t - last >= period) {
      last = world.t;
      let dets = world.visible(G.CAM_X, G.CAM_W).map(({ item, ...d }) => d);
      const m = P.perception;
      if (m) {
        dets = dets.filter(() => rng() >= m.miss).map((d) => {
          const row = m.confusion?.[d.cls]; let cls = d.cls;
          if (row) { let r = rng(), acc = 0; for (const [k, p] of Object.entries(row)) { acc += p; if (r <= acc) { cls = k; break; } } }
          return { ...d, cls, conf: m.conf ? m.conf[0] + rng() * (m.conf[1] - m.conf[0]) : 1 };
        });
      }
      tracker.update(dets, world.s, world.t);
    }
    tracker.prune(world.s, world.t);
    cell.step(dt, world, tracker, { vb, policy: P.policy, confThr: P.confThr, gripFail: P.gripFail }, metrics);
  }
  return { ...metrics.summary(world.t), busy: cell.arms.map((a) => a.busy / world.t) };
}
