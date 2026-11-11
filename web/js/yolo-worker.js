/* Instance segmentation in a Web Worker with ONNX Runtime Web (WASM, optional WebGPU).
   Input: the RGBA frame captured by the virtual camera. Output: one detection per object
   with class, score, box, visible area, grasp point and a class-coloured mask overlay.
   Decodes the raw YOLO-seg head: output0 [1, 4+nc+32, N], output1 (prototypes) [1, 32, mh, mw]. */
let session = null, cfg = null, ortReady = null;

const PALETTE = { plastic: [242, 193, 78], glass: [63, 193, 176], cardboard: [194, 143, 92], metal: [157, 180, 208],
  paper: [236, 231, 214], organic: [124, 207, 95], other: [138, 143, 153] };

async function init(msg) {
  cfg = msg;
  const gpu = msg.ep === 'webgpu';
  if (!ortReady) {
    importScripts(gpu ? msg.ortGpuUrl : msg.ortUrl);
    ort.env.wasm.wasmPaths = gpu ? msg.ortGpuBase : msg.ortBase;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
    ortReady = true;
  }
  const buf = await (await fetch(msg.modelUrl)).arrayBuffer();
  session = await ort.InferenceSession.create(buf, { executionProviders: gpu ? ['webgpu', 'wasm'] : ['wasm'], graphOptimizationLevel: 'all' });
  // warm-up
  const [h, w] = cfg.imgsz;
  await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', new Float32Array(3 * h * w), [1, 3, h, w]) });
  return { threads: ort.env.wasm.numThreads, ep: gpu ? 'webgpu' : 'wasm', bytes: buf.byteLength };
}

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1), x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter + 1e-9);
}

async function infer(msg) {
  const t0 = performance.now();
  const W = msg.w, H = msg.h, [inH, inW] = cfg.imgsz;
  let rgba = new Uint8ClampedArray(msg.buf), r = 1, padX = 0, padY = 0;
  if (W !== inW || H !== inH) {                         // letterbox
    r = Math.min(inW / W, inH / H);
    const nw = Math.round(W * r), nh = Math.round(H * r); padX = (inW - nw) >> 1; padY = (inH - nh) >> 1;
    const oc = new OffscreenCanvas(inW, inH), c = oc.getContext('2d');
    c.fillStyle = 'rgb(114,114,114)'; c.fillRect(0, 0, inW, inH);
    c.drawImage(await createImageBitmap(new ImageData(rgba, W, H)), padX, padY, nw, nh);
    rgba = c.getImageData(0, 0, inW, inH).data;
  }
  const n = inW * inH, x = new Float32Array(3 * n);
  for (let i = 0, p = 0; i < n; i++, p += 4) { x[i] = rgba[p] / 255; x[n + i] = rgba[p + 1] / 255; x[2 * n + i] = rgba[p + 2] / 255; }
  const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, inH, inW]) });
  const o0 = out[session.outputNames[0]], o1 = out[session.outputNames[1]];
  const C = o0.dims[1], N = o0.dims[2], d = o0.data;
  const nm = o1.dims[1], mh = o1.dims[2], mw = o1.dims[3], pr = o1.data, nc = C - 4 - nm;
  const names = cfg.names;
  let cand = [];
  for (let j = 0; j < N; j++) {
    let best = 0, bc = 0;
    for (let k = 0; k < nc; k++) { const v = d[(4 + k) * N + j]; if (v > best) { best = v; bc = k; } }
    if (best < cfg.conf) continue;
    const cx = d[j], cy = d[N + j], w = d[2 * N + j], h = d[3 * N + j];
    cand.push({ j, score: best, k: bc, x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 });
  }
  cand.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const cnd of cand) {
    if (keep.length >= 80) break;
    if (keep.every((q) => iou(q, cnd) < cfg.iou)) keep.push(cnd);
  }
  const ov = new Uint8ClampedArray(mw * mh * 4), sx = inW / mw, sy = inH / mh, dets = [];
  for (const q of keep) {
    const bx0 = Math.max(0, Math.floor(q.x1 / sx)), by0 = Math.max(0, Math.floor(q.y1 / sy));
    const bx1 = Math.min(mw, Math.ceil(q.x2 / sx)), by1 = Math.min(mh, Math.ceil(q.y2 / sy));
    const coef = new Float32Array(nm);
    for (let k = 0; k < nm; k++) coef[k] = d[(4 + nc + k) * N + q.j];
    let cnt = 0, mx = 0, my = 0; const pts = [];
    for (let yy = by0; yy < by1; yy++) for (let xx = bx0; xx < bx1; xx++) {
      let v = 0; const off = yy * mw + xx;
      for (let k = 0; k < nm; k++) v += coef[k] * pr[k * mh * mw + off];
      if (v > 0) { cnt++; mx += xx; my += yy; pts.push(xx, yy); }
    }
    if (!cnt) continue;
    mx /= cnt; my /= cnt;
    let gx = mx, gy = my, bd = 1e9;                      // grasp point: mask pixel nearest the centroid
    for (let i = 0; i < pts.length; i += 2) { const e = (pts[i] - mx) ** 2 + (pts[i + 1] - my) ** 2; if (e < bd) { bd = e; gx = pts[i]; gy = pts[i + 1]; } }
    const area = cnt * sx * sy / (r * r);
    if (area < cfg.minArea) continue;
    const cls = names[q.k] ?? names[q.k % names.length], col = PALETTE[cls] || [255, 255, 255];
    for (let i = 0; i < pts.length; i += 2) { const o = (pts[i + 1] * mw + pts[i]) * 4; ov[o] = col[0]; ov[o + 1] = col[1]; ov[o + 2] = col[2]; ov[o + 3] = 255; }
    dets.push({
      cls, conf: q.score, area,
      x: ((gx + 0.5) * sx - padX) / r, y: ((gy + 0.5) * sy - padY) / r,
      bx: (q.x1 - padX) / r, by: (q.y1 - padY) / r, bw: (q.x2 - q.x1) / r, bh: (q.y2 - q.y1) / r,
    });
  }
  const bmp = await createImageBitmap(new ImageData(ov, mw, mh));
  const src = [padX / sx, padY / sy, (inW - 2 * padX) / sx, (inH - 2 * padY) / sy];
  return { dets, overlay: { bmp, src }, ms: performance.now() - t0 };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') self.postMessage({ type: 'ready', id: m.id, info: await init(m) });
    else if (m.type === 'infer') {
      const res = await infer(m);
      self.postMessage({ type: 'result', id: m.id, ...res }, [res.overlay.bmp]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
};
