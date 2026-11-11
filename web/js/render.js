// Drawing. `scene` is the belt with its objects and nothing else: it is the image the
// virtual camera captures. `view` adds bins, robots and the perception overlays.
import { G, COLOR, LABEL, STATIONS } from './config.js';

const SH_PAD = 12, SH_DOWN = 6;

function loadImage(src) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = src; });
}

// Soft shadow: silhouette reduced 6x and enlarged again (same recipe as the training generator).
function makeShadow(img) {
  const w = img.width + 2 * SH_PAD, h = img.height + 2 * SH_PAD;
  const sil = new OffscreenCanvas(w, h), c = sil.getContext('2d');
  c.drawImage(img, SH_PAD, SH_PAD);
  c.globalCompositeOperation = 'source-in'; c.fillStyle = '#000'; c.fillRect(0, 0, w, h);
  const sw = Math.max(2, Math.round(w / SH_DOWN)), sh = Math.max(2, Math.round(h / SH_DOWN));
  const small = new OffscreenCanvas(sw, sh), sc = small.getContext('2d');
  sc.imageSmoothingEnabled = true; sc.imageSmoothingQuality = 'high'; sc.drawImage(sil, 0, 0, sw, sh);
  const out = new OffscreenCanvas(w, h), oc = out.getContext('2d');
  oc.imageSmoothingEnabled = true; oc.imageSmoothingQuality = 'high'; oc.drawImage(small, 0, 0, w, h);
  return out;
}

export async function loadAssets(base, sprites) {
  const belt = await loadImage(`${base}/belt.jpg`);
  const atlas = await loadImage(`${base}/atlas.webp`);
  const imgs = await Promise.all(sprites.map((s) => createImageBitmap(atlas, s.ax, s.ay, s.w, s.h)));
  return { belt, imgs, shadows: imgs.map(makeShadow) };
}

export class Renderer {
  constructor(canvas, assets, scene) {
    this.cv = canvas; this.ctx = canvas.getContext('2d');
    canvas.width = G.W; canvas.height = G.H;
    this.scene = new OffscreenCanvas(G.W, G.BELT_H);
    this.sctx = this.scene.getContext('2d', { willReadFrequently: true });
    this.a = assets; this.sh = scene.shadow;
  }

  drawItem(c, it, x, y, scale = 1, shadow = true) {
    const img = this.a.imgs[it.sp], hw = it.hw * scale, hh = it.hh * scale, k = it.k * scale;
    if (shadow) {
      c.save(); c.translate(x + this.sh.dx, y + this.sh.dy); c.rotate(it.rot); c.globalAlpha = this.sh.alpha;
      c.drawImage(this.a.shadows[it.sp], -hw - SH_PAD * k, -hh - SH_PAD * k, 2 * hw + 2 * SH_PAD * k, 2 * hh + 2 * SH_PAD * k);
      c.restore();
    }
    c.save(); c.translate(x, y); c.rotate(it.rot); c.drawImage(img, -hw, -hh, 2 * hw, 2 * hh); c.restore();
  }

  drawScene(world) {
    const c = this.sctx, tw = this.a.belt.width;
    for (let x = (world.s % tw) - tw; x < G.W; x += tw) c.drawImage(this.a.belt, Math.round(x), 0);
    const items = world.items.filter((it) => it.state === 'belt').sort((a, b) => a.z - b.z);
    for (const it of items) this.drawItem(c, it, world.x(it), it.y);
  }

  capture() { return this.sctx.getImageData(G.CAM_X, 0, G.CAM_W, G.BELT_H); }

  drawView(world, cell, tracker, metrics, ui) {
    const c = this.ctx, Y = G.BELT_Y, t = world.t;
    c.fillStyle = '#0b0e11'; c.fillRect(0, 0, G.W, G.H);
    // bins
    const sum = metrics.summary(t);
    STATIONS.forEach((cls2, k) => {
      const x0 = G.ST_X0 + k * G.ST_W + 10, w = G.ST_W - 20;
      cls2.forEach((cl, side) => {
        const y0 = side === 0 ? 8 : Y + G.BELT_H + 12, h = G.BIN_H - 20;
        c.fillStyle = '#12171c'; c.strokeStyle = COLOR[cl]; c.lineWidth = 2;
        c.beginPath(); c.roundRect(x0, y0, w, h, 8); c.fill(); c.stroke();
        c.fillStyle = COLOR[cl]; c.font = '600 15px system-ui, sans-serif'; c.textBaseline = 'top';
        c.fillText(LABEL[cl].toUpperCase(), x0 + 12, y0 + 10);
        const n = Object.values(metrics.bins[cl]).reduce((a, b) => a + b, 0);
        c.fillStyle = '#9aa4ae'; c.font = '13px ui-monospace, monospace';
        c.fillText(`${n} uds`, x0 + 12, y0 + 34);
        if (sum.pur[cl] != null) c.fillText(`pureza ${(sum.pur[cl] * 100).toFixed(0)}%`, x0 + 12, y0 + 52);
      });
    });
    // belt
    c.drawImage(this.scene, 0, Y);
    c.fillStyle = '#1d242b'; c.fillRect(0, Y - 6, G.W, 6); c.fillRect(0, Y + G.BELT_H, G.W, 6);
    // objects that were dropped, shrinking into the bin
    for (const it of world.items) {
      if (it.state !== 'binned') continue;
      const q = Math.min(1, (t - it.binAt) / 0.45);
      c.globalAlpha = 1 - q * q; this.drawItem(c, it, it.bx, it.by + Y, 0.8 - 0.45 * q, false); c.globalAlpha = 1;
    }
    // camera field of view
    c.strokeStyle = 'rgba(126,224,129,.85)'; c.lineWidth = 2; c.setLineDash([10, 8]);
    c.strokeRect(G.CAM_X, Y + 1, G.CAM_W, G.BELT_H - 2); c.setLineDash([]);
    c.fillStyle = 'rgba(126,224,129,.95)'; c.font = '600 12px ui-monospace, monospace'; c.textBaseline = 'bottom';
    c.fillText('CAMPO DE LA CÁMARA  640 x 480', G.CAM_X + 4, Y - 10);
    // perception overlay, carried along with the belt since the frame was captured
    const P = ui.lastPerception;
    if (ui.showDets && P) {
      const dx = world.s - P.s;
      c.save(); c.beginPath(); c.rect(0, Y, G.W, G.BELT_H); c.clip();
      if (P.overlay && dx < 260) {
        c.globalAlpha = 0.5; c.imageSmoothingEnabled = false;
        c.drawImage(P.overlay.bmp, ...P.overlay.src, G.CAM_X + dx, Y, G.CAM_W, G.BELT_H);
        c.globalAlpha = 1; c.imageSmoothingEnabled = true;
      }
      c.font = '600 11px ui-monospace, monospace'; c.textBaseline = 'bottom';
      for (const d of dx < 260 ? P.dets : []) {
        const x = G.CAM_X + d.bx + dx, y = Y + d.by;
        c.strokeStyle = COLOR[d.cls]; c.lineWidth = 1.5; c.strokeRect(x, y, d.bw, d.bh);
        const txt = `${LABEL[d.cls]} ${(d.conf * 100).toFixed(0)}`;
        const tw = c.measureText(txt).width + 6;
        c.fillStyle = COLOR[d.cls]; c.fillRect(x, y - 14, tw, 14);
        c.fillStyle = '#0b0e11'; c.fillText(txt, x + 3, y - 1);
      }
      c.restore();
    }
    // tracks handed to the robots
    if (ui.showTracks) {
      for (const tr of tracker.tracks) {
        const x = tr.u + world.s;
        if (tr.done || x < G.CAM_X + G.CAM_W || x > G.W) continue;
        const low = tr.conf < ui.confThr || tr.cls === 'other';
        c.strokeStyle = COLOR[tr.cls]; c.fillStyle = COLOR[tr.cls]; c.lineWidth = 2; c.globalAlpha = low ? 0.35 : 1;
        c.beginPath(); c.arc(x, Y + tr.y, 9, 0, 6.283); c.stroke();
        c.beginPath(); c.arc(x, Y + tr.y, 2.5, 0, 6.283); c.fill();
        if (tr.claimed) { c.beginPath(); c.arc(x, Y + tr.y, 14, 0, 6.283); c.stroke(); }
        c.globalAlpha = 1;
      }
    }
    // robots
    for (const arm of cell.arms) {
      const x = arm.p.x, y = Y + arm.p.y;
      c.fillStyle = 'rgba(255,255,255,.035)'; c.fillRect(arm.zx0 + 2, Y, arm.zx1 - arm.zx0 - 4, G.BELT_H);
      c.strokeStyle = 'rgba(255,255,255,.12)'; c.lineWidth = 1; c.strokeRect(arm.zx0 + 2, Y, arm.zx1 - arm.zx0 - 4, G.BELT_H);
      if (ui.showPlan && arm.state === 'approach') {
        c.strokeStyle = 'rgba(255,255,255,.55)'; c.setLineDash([4, 5]); c.lineWidth = 1.2;
        c.beginPath(); c.moveTo(x, y); c.lineTo(arm.pick.x, Y + arm.pick.y); c.stroke(); c.setLineDash([]);
        c.beginPath(); c.arc(arm.pick.x, Y + arm.pick.y, 5, 0, 6.283); c.stroke();
      }
      if (arm.held) this.drawItem(c, arm.held, x, y, 0.92, false);
      c.fillStyle = '#2a323b'; c.fillRect(x - 6, 2, 12, G.H - 4);                    // bridge
      c.fillStyle = '#3b4652'; c.fillRect(x - 2, 2, 4, G.H - 4);
      const active = arm.state === 'grip' || arm.state === 'carry' || arm.state === 'drop';
      c.fillStyle = '#d7dde3'; c.beginPath(); c.roundRect(x - 15, y - 15, 30, 30, 6); c.fill();   // carriage
      c.fillStyle = active ? '#7ee081' : '#56616d'; c.beginPath(); c.arc(x, y, 7, 0, 6.283); c.fill();
      c.fillStyle = '#2a323b'; c.beginPath(); c.roundRect(x - 14, 0, 28, 16, 4); c.fill();
      c.fillStyle = '#e6eaee'; c.font = '700 11px ui-monospace, monospace'; c.textBaseline = 'middle'; c.textAlign = 'center';
      c.fillText(`R${arm.id + 1}`, x, 8.5); c.textAlign = 'left';
    }
    // end of the line
    c.fillStyle = '#9aa4ae'; c.font = '600 12px ui-monospace, monospace'; c.textBaseline = 'bottom';
    c.fillText('RECHAZO >', G.W - G.TAIL + 2, Y - 10);
  }
}
