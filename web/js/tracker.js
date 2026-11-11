// Multi-object tracker in belt coordinates. The belt encoder gives the travel s(t), so an
// object fixed on the belt has a constant coordinate u = x - s. Detections are converted
// with the s of the instant the frame was CAPTURED, which cancels the inference latency.
import { G, CLASSES } from './config.js';

const GATE = 30;          // px: maximum distance to associate a detection with a track

export class Tracker {
  constructor() { this.tracks = []; this.nextId = 1; }

  update(dets, sCapture, tNow) {
    const D = dets.map((d) => ({ d, u: G.CAM_X + d.x - sCapture, y: d.y, used: false }));
    const pairs = [];
    for (const tr of this.tracks) {
      if (tr.done) continue;
      for (const q of D) {
        const dist = Math.hypot(q.u - tr.u, q.y - tr.y);
        if (dist < GATE) pairs.push([dist, tr, q]);
      }
    }
    pairs.sort((a, b) => a[0] - b[0]);
    const taken = new Set();
    for (const [, tr, q] of pairs) {
      if (q.used || taken.has(tr)) continue;
      q.used = true; taken.add(tr);
      const a = 0.5;                                   // exponential smoothing of the position
      tr.u += a * (q.u - tr.u); tr.y += a * (q.y - tr.y);
      this.vote(tr, q.d, tNow);
    }
    for (const q of D) {
      if (q.used) continue;
      const tr = { id: this.nextId++, u: q.u, y: q.y, votes: {}, sum: 0, hits: 0, born: tNow, claimed: false, done: false };
      this.vote(tr, q.d, tNow);
      this.tracks.push(tr);
    }
  }

  vote(tr, d, tNow) {
    tr.votes[d.cls] = (tr.votes[d.cls] || 0) + d.conf;
    tr.sum += d.conf; tr.hits++; tr.last = tNow; tr.area = d.area; tr.det = d;
    let best = null;
    for (const c of CLASSES) if (tr.votes[c] && (!best || tr.votes[c] > tr.votes[best])) best = c;
    tr.cls = best;
    // confidence: mean score of the winning class, penalised when frames disagree
    tr.conf = (tr.votes[best] / tr.hits) * (tr.votes[best] / tr.sum);
  }

  // Tracks the planner may use: seen at least twice, or already out of the camera view.
  ready(tr, s) {
    const x = tr.u + s;
    return tr.hits >= 2 || x > G.CAM_X + G.CAM_W;
  }

  prune(s, tNow) {
    this.tracks = this.tracks.filter((tr) => {
      const x = tr.u + s;
      if (x > G.W + 40) return false;
      if (tr.done && tNow - tr.doneAt > 0.5) return false;
      // one-frame ghosts that were never confirmed while still inside the camera view
      if (tr.hits < 2 && x < G.CAM_X + G.CAM_W - 40 && tNow - tr.last > 0.9) return false;
      return true;
    });
  }
}
