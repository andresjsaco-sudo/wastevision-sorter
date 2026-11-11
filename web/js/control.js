// Robot cell: task allocation, intercept planning and motion control of the gantry arms.
//
//  1. Allocation: every idle arm chooses, among the confirmed tracks of its two fractions
//     that it can still reach, the one given by the policy (EDF by default) and claims it.
//  2. Planning: the pick point is where the object WILL be when the head gets there
//     (the belt speed is fed forward), following a trapezoidal velocity profile.
//  3. Control: each axis is a double integrator driven by PD feedback plus acceleration
//     feed-forward with actuator saturation, so the head really has to track the plan.
//  4. Grasp: the suction cup closes on whatever object is on top at the head's ACTUAL
//     position. A wrong position or class is not corrected by the simulator.
import { G, ARM, STATIONS } from './config.js';
import { mulberry32 } from './world.js';

function trapTime(d) {
  const a = ARM.AMAX, v = ARM.VMAX;
  return d <= (v * v) / a ? 2 * Math.sqrt(d / a) : d / v + v / a;
}
// distance, speed and acceleration along a trapezoidal move of length d at time t
function trap(d, t) {
  const a = ARM.AMAX, v = ARM.VMAX;
  if (t <= 0) return [0, 0, 0];
  if (d <= (v * v) / a) {
    const tp = Math.sqrt(d / a);
    if (t < tp) return [0.5 * a * t * t, a * t, a];
    if (t < 2 * tp) { const r = 2 * tp - t; return [d - 0.5 * a * r * r, a * r, -a]; }
    return [d, 0, 0];
  }
  const ta = v / a, da = 0.5 * v * ta, tc = (d - 2 * da) / v;
  if (t < ta) return [0.5 * a * t * t, a * t, a];
  if (t < ta + tc) return [da + v * (t - ta), v, 0];
  if (t < 2 * ta + tc) { const r = 2 * ta + tc - t; return [d - 0.5 * a * r * r, a * r, -a]; }
  return [d, 0, 0];
}

export class RobotCell {
  constructor(armsPerStation, seed) {
    this.rng = mulberry32(seed ^ 0x9e3779b9);
    this.arms = [];
    STATIONS.forEach((classes, k) => {
      const x0 = G.ST_X0 + k * G.ST_W, w = G.ST_W / armsPerStation;
      for (let j = 0; j < armsPerStation; j++) {
        const zx0 = x0 + j * w, zx1 = zx0 + w, home = { x: zx0 + w * 0.35, y: G.BELT_H / 2 };
        this.arms.push({
          id: this.arms.length, station: k, classes, zx0, zx1, home,
          p: { ...home }, v: { x: 0, y: 0 }, r: { ...home }, rv: { x: 0, y: 0 }, ra: { x: 0, y: 0 },
          state: 'idle', seg: null, held: null, track: null, picks: 0, busy: 0,
        });
      }
    });
  }

  segment(arm, to, t0) {
    const dx = to.x - arm.r.x, dy = to.y - arm.r.y, d = Math.hypot(dx, dy) || 1e-6;
    arm.seg = { x0: arm.r.x, y0: arm.r.y, ux: dx / d, uy: dy / d, d, t0, T: trapTime(d) };
  }

  // Where and when can this arm meet the object? null if it leaves the zone first.
  intercept(arm, x, y, vb) {
    const lo = arm.zx0 + ARM.MARGIN, hi = arm.zx1 - ARM.MARGIN - vb * ARM.T_PICK;
    if (x > hi) return null;
    const tw = (lo - x) / vb;                       // time until the object enters the zone
    if (tw > 0 && tw >= trapTime(Math.hypot(lo - arm.r.x, y - arm.r.y))) return { tau: tw, px: lo, py: y, wait: tw };
    let tau = Math.max(tw, 0);
    for (let i = 0; i < 8; i++) tau = Math.max(trapTime(Math.hypot(x + vb * tau - arm.r.x, y - arm.r.y)), tw);
    const px = x + vb * tau;
    return px > hi ? null : { tau, px, py: y, wait: 0, deadline: (hi - x) / vb };
  }

  allocate(arm, tracker, world, P) {
    let best = null, bestKey = Infinity;
    for (const tr of tracker.tracks) {
      if (tr.claimed || tr.done || !arm.classes.includes(tr.cls) || tr.conf < P.confThr) continue;
      if (!tracker.ready(tr, world.s)) continue;
      const x = tr.u + world.s;
      const plan = this.intercept(arm, x, tr.y, P.vb);
      if (!plan || plan.wait > 0.5) continue;       // do not commit to objects still far upstream
      const deadline = (arm.zx1 - x) / P.vb;
      const key = P.policy === 'nearest' ? plan.tau : P.policy === 'fifo' ? tr.id : deadline;
      if (key < bestKey) { bestKey = key; best = { tr, plan }; }
    }
    if (!best) return false;
    best.tr.claimed = true;
    arm.track = best.tr; arm.gripped = false;
    arm.pick = { x: best.plan.px, y: best.plan.py, t: world.t + best.plan.tau };
    this.segment(arm, arm.pick, world.t);
    arm.state = 'approach';
    return true;
  }

  step(dt, world, tracker, P, metrics) {
    const t = world.t;
    for (const arm of this.arms) {
      if (arm.state !== 'idle' && arm.state !== 'home') arm.busy += dt;
      // ---------------- supervisor (state machine)
      if (arm.state === 'idle' || arm.state === 'home') {
        if (!this.allocate(arm, tracker, world, P)) {
          if (arm.state === 'idle' && Math.hypot(arm.r.x - arm.home.x, arm.r.y - arm.home.y) > 4) {
            this.segment(arm, arm.home, t); arm.state = 'home';
          } else if (arm.state === 'home' && t >= arm.seg.t0 + arm.seg.T) arm.state = 'idle';
        }
      } else if (arm.state === 'approach') {
        if (t >= arm.pick.t) { arm.state = 'grip'; arm.seg = null; }
      } else if (arm.state === 'grip') {
        if (!arm.gripped && t >= arm.pick.t + ARM.T_PICK * ARM.T_GRIP) {
          arm.gripped = true;
          const err = Math.hypot(arm.p.x - arm.r.x, arm.p.y - arm.r.y);
          let it = world.topItemAt(arm.p.x, arm.p.y);
          const slipped = it && this.rng() < P.gripFail;
          if (it && !slipped) { it.state = 'held'; arm.held = it; }
          arm.track.done = true; arm.track.doneAt = t;
          metrics.onGrip(t, arm.track, slipped ? null : it, err, !!it);
        }
        if (t >= arm.pick.t + ARM.T_PICK) {
          if (arm.held) {
            const top = arm.track.cls === arm.classes[0];
            arm.bin = arm.track.cls;
            const bx = Math.min(Math.max(arm.r.x, arm.zx0 + 34), arm.zx1 - 34);
            this.segment(arm, { x: bx, y: top ? -G.BIN_H * 0.52 : G.BELT_H + G.BIN_H * 0.52 }, t);
            arm.state = 'carry';
          } else arm.state = 'idle';
          arm.track = null;
        }
      } else if (arm.state === 'carry') {
        if (t >= arm.seg.t0 + arm.seg.T) { arm.state = 'drop'; arm.tDrop = t + ARM.T_DROP; }
      } else if (arm.state === 'drop') {
        if (t >= arm.tDrop) {
          const it = arm.held; arm.held = null; arm.picks++;
          it.state = 'binned'; it.binAt = t; it.bx = arm.p.x; it.by = arm.p.y;
          metrics.onBin(t, arm.bin, it.cls);
          arm.state = 'idle';
        }
      }
      // ---------------- reference trajectory
      if (arm.state === 'grip') {
        arm.r.x = arm.pick.x + P.vb * (t - arm.pick.t); arm.r.y = arm.pick.y;
        arm.rv.x = P.vb; arm.rv.y = 0; arm.ra.x = 0; arm.ra.y = 0;
      } else if (arm.seg && (arm.state === 'approach' || arm.state === 'carry' || arm.state === 'home')) {
        const [sd, sv, sa] = trap(arm.seg.d, t - arm.seg.t0);
        arm.r.x = arm.seg.x0 + arm.seg.ux * sd; arm.r.y = arm.seg.y0 + arm.seg.uy * sd;
        arm.rv.x = arm.seg.ux * sv; arm.rv.y = arm.seg.uy * sv;
        arm.ra.x = arm.seg.ux * sa; arm.ra.y = arm.seg.uy * sa;
      } else { arm.rv.x = arm.rv.y = arm.ra.x = arm.ra.y = 0; }
      // ---------------- axis dynamics: PD + feed-forward, saturated, semi-implicit Euler
      const n = Math.max(1, Math.ceil(dt / (1 / 240))), h = dt / n;
      for (let i = 0; i < n; i++) {
        let ux = arm.ra.x + ARM.KP * (arm.r.x - arm.p.x) + ARM.KD * (arm.rv.x - arm.v.x);
        let uy = arm.ra.y + ARM.KP * (arm.r.y - arm.p.y) + ARM.KD * (arm.rv.y - arm.v.y);
        const m = Math.hypot(ux, uy);
        if (m > ARM.UMAX) { ux *= ARM.UMAX / m; uy *= ARM.UMAX / m; }
        arm.v.x += ux * h; arm.v.y += uy * h;
        arm.p.x += arm.v.x * h; arm.p.y += arm.v.y * h;
      }
      if (arm.held) { arm.held.hx = arm.p.x; arm.held.hy = arm.p.y; }
    }
    for (const it of world.items) if (it.state === 'binned' && t - it.binAt > 0.45) it.state = 'gone';
  }
}

export class Metrics {
  constructor(classes) {
    this.classes = classes;
    this.bins = {}; this.end = {};
    for (const b of classes) { this.bins[b] = {}; this.end[b] = 0; for (const c of classes) this.bins[b][c] = 0; }
    this.grips = 0; this.air = 0; this.slips = 0; this.errSum = 0; this.errMax = 0;
    this.pickTimes = [];
    this.perc = { tp: 0, fp: 0, fn: 0, ok: 0, frames: 0, msSum: 0 };
  }
  onGrip(t, tr, it, err, touched) {
    this.grips++; this.errSum += err; if (err > this.errMax) this.errMax = err;
    if (!touched) this.air++; else if (!it) this.slips++;
  }
  onBin(t, bin, trueCls) { this.bins[bin][trueCls]++; this.pickTimes.push(t); }
  onExit(it) { this.end[it.cls]++; }
  // compare one frame of detections with what was really visible at capture time
  onPerception(dets, truth, ms) {
    const used = new Set(); const p = this.perc; p.frames++; p.msSum += ms;
    for (const d of dets) {
      let best = null, bd = 32;
      for (const g of truth) {
        if (used.has(g)) continue;
        const dist = Math.hypot(g.x - d.x, g.y - d.y);
        if (dist < bd) { bd = dist; best = g; }
      }
      if (best) { used.add(best); p.tp++; if (best.cls === d.cls) p.ok++; } else p.fp++;
    }
    p.fn += truth.length - used.size;
  }
  summary(t) {
    const rec = {}, pur = {}; let good = 0, binned = 0, recyc = 0;
    for (const c of this.classes) {
      let inBin = 0, out = this.end[c];
      for (const t2 of this.classes) inBin += this.bins[c][t2];
      for (const b of this.classes) out += this.bins[b][c];
      pur[c] = inBin ? this.bins[c][c] / inBin : null;
      rec[c] = out ? this.bins[c][c] / out : null;
      if (c !== 'other') { good += this.bins[c][c]; binned += inBin; recyc += out; }
    }
    let otherOut = this.end.other; for (const b of this.classes) otherOut += this.bins[b].other;
    const win = 60, recent = this.pickTimes.filter((x) => x > t - win).length;
    return {
      rec, pur, purity: binned ? good / binned : null, recovery: recyc ? good / recyc : null,
      residue: otherOut ? this.end.other / otherOut : null,
      ppm: recent * 60 / Math.min(Math.max(t, 1), win), binned,
      air: this.air, slips: this.slips, grips: this.grips,
      errMm: this.grips ? (this.errSum / this.grips) * 1000 / G.PXM : null,
      missed: this.classes.filter((c) => c !== 'other').reduce((a, c) => a + this.end[c], 0),
    };
  }
}
