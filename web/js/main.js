import { G, CLASSES, LABEL, COLOR, DEFAULTS } from './config.js';
import { World, decodeMask } from './world.js';
import { Tracker } from './tracker.js';
import { RobotCell, Metrics } from './control.js';
import { Renderer, loadAssets } from './render.js';

const $ = (s) => document.querySelector(s);
const qs = new URLSearchParams(location.search);
const ORT_VER = '1.31.1';
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;

const P = { ...DEFAULTS };                 // live parameters
const ui = { showDets: true, showTracks: true, showPlan: true, lastPerception: null, confThr: P.confThr };
let scene, sprites, world, tracker, cell, metrics, renderer;
let paused = false, lastFrame = 0, lastCapture = -1, busy = false, fps = 0;
let worker = null, model = null, reqId = 0, percHz = 0, lastPercWall = 0, latMs = 0;
const pending = new Map();

function reset() {
  world = new World(scene, sprites, P.seed);
  tracker = new Tracker();
  cell = new RobotCell(P.armsPerStation, P.seed);
  metrics = new Metrics(CLASSES);
  ui.lastPerception = null; lastCapture = -1; busy = false;
  // pre-fill the line so the page does not start with an empty belt
  const vb = P.beltSpeed * G.PXM;
  for (let i = 0; i < Math.round((G.CAM_X + G.CAM_W * 0.6) / vb / 0.02); i++) world.step(0.02, vb, P.rate);
}

// ---------------------------------------------------------------- perception
function call(msg, transfer = []) {
  return new Promise((res, rej) => { const id = ++reqId; pending.set(id, { res, rej }); worker.postMessage({ ...msg, id }, transfer); });
}

async function loadModel(m, ep) {
  setStatus(`Cargando ${m.name}...`, 'warn');
  if (worker) worker.terminate();
  pending.clear(); busy = false; model = null;
  worker = new Worker('js/yolo-worker.js');
  worker.onmessage = (e) => {
    const p = pending.get(e.data.id); if (!p) return; pending.delete(e.data.id);
    e.data.type === 'error' ? p.rej(new Error(e.data.message)) : p.res(e.data);
  };
  const base = new URL('vendor/ort/', location.href).href;
  const r = await call({
    type: 'init', modelUrl: new URL(m.file, location.href).href, imgsz: m.imgsz, names: m.names || CLASSES,
    conf: 0.25, iou: 0.6, minArea: scene.min_visible_px, ep,
    ortUrl: base + 'ort.wasm.min.js', ortBase: base, ortGpuUrl: ORT_CDN + 'ort.webgpu.min.js', ortGpuBase: ORT_CDN,
  });
  model = { ...m, info: r.info };
  setStatus(`${m.name} · ${r.info.ep.toUpperCase()}${r.info.ep === 'wasm' ? ` x${r.info.threads}` : ''}`, 'ok');
}

function setStatus(txt, kind) { const el = $('#percStatus'); el.textContent = txt; el.dataset.kind = kind; }

function perceive() {
  const t = world.t;
  if (busy || t - lastCapture < (P.perception === 'oracle' ? 0.2 : 0.12)) return;
  lastCapture = t;
  const s = world.s, truth = world.visible(G.CAM_X, G.CAM_W);
  if (P.perception === 'oracle' || !model) {
    const dets = truth.map(({ item, ...d }) => d);
    tracker.update(dets, s, t);
    ui.lastPerception = { s, dets, overlay: null };
    return;
  }
  busy = true;
  const img = renderer.capture(), wall = performance.now();
  call({ type: 'infer', buf: img.data.buffer, w: img.width, h: img.height }, [img.data.buffer]).then((r) => {
    busy = false;
    tracker.update(r.dets, s, world.t);
    metrics.onPerception(r.dets, truth, r.ms);
    ui.lastPerception?.overlay?.bmp.close?.();
    ui.lastPerception = { s, dets: r.dets, overlay: r.overlay };
    const now = performance.now();
    latMs = latMs ? latMs * 0.8 + (now - wall) * 0.2 : now - wall;
    if (lastPercWall) percHz = percHz * 0.8 + (1000 / (now - lastPercWall)) * 0.2;
    lastPercWall = now;
  }).catch((err) => { busy = false; setStatus('Error del modelo: ' + err.message, 'err'); P.perception = 'oracle'; $('#perception').value = 'oracle'; });
}

// ---------------------------------------------------------------- main loop
function frame(now) {
  requestAnimationFrame(frame);
  const dtReal = Math.min(0.05, (now - lastFrame) / 1000 || 0.016); lastFrame = now;
  fps = fps * 0.95 + (1 / dtReal) * 0.05;
  if (!paused) {
    const vb = P.beltSpeed * G.PXM;
    world.step(dtReal, vb, P.rate);
    for (const it of world.exited) metrics.onExit(it);
    renderer.drawScene(world);
    perceive();
    tracker.prune(world.s, world.t);
    cell.step(dtReal, world, tracker, { vb, policy: P.policy, confThr: P.confThr, gripFail: P.gripFail }, metrics);
  }
  ui.confThr = P.confThr;
  renderer.drawView(world, cell, tracker, metrics, ui);
}

// ---------------------------------------------------------------- panel
const pct = (v) => (v == null ? '--' : (v * 100).toFixed(1) + '%');
function refreshPanel() {
  const s = metrics.summary(world.t), pc = metrics.perc;
  const set = (id, v) => { $(id).textContent = v; };
  set('#kRecovery', pct(s.recovery)); set('#kPurity', pct(s.purity)); set('#kPpm', s.ppm.toFixed(0));
  set('#kMissed', s.missed); set('#kResidue', pct(s.residue));
  set('#kAir', s.grips ? `${s.air + s.slips} / ${s.grips}` : '--');
  set('#kErr', s.errMm == null ? '--' : s.errMm.toFixed(1) + ' mm');
  set('#kFps', fps.toFixed(0));
  const oracle = P.perception === 'oracle' || !model;
  set('#kRecall', oracle ? 'n/a' : pct(pc.tp + pc.fn ? pc.tp / (pc.tp + pc.fn) : null));
  set('#kPrec', oracle ? 'n/a' : pct(pc.tp + pc.fp ? pc.tp / (pc.tp + pc.fp) : null));
  set('#kCls', oracle ? 'n/a' : pct(pc.tp ? pc.ok / pc.tp : null));
  set('#kLat', oracle ? 'n/a' : `${latMs.toFixed(0)} ms · ${percHz.toFixed(1)} Hz`);
  $('#fracBody').innerHTML = CLASSES.map((c) => {
    const n = c === 'other' ? metrics.end.other : metrics.bins[c][c];
    return `<tr><td><i style="background:${COLOR[c]}"></i>${LABEL[c]}</td><td>${n}</td>` +
      `<td>${pct(c === 'other' ? s.residue : s.rec[c])}</td><td>${c === 'other' ? '--' : pct(s.pur[c])}</td></tr>`;
  }).join('');
}

function bind(id, key, fmt, parse = Number) {
  const el = $(id), out = $(id + 'Out');
  const apply = () => { P[key] = parse(el.value); if (out) out.textContent = fmt(P[key]); };
  el.value = P[key]; el.addEventListener('input', apply); apply();
}

async function boot() {
  [scene, sprites] = await Promise.all([fetch('assets/scene.json').then((r) => r.json()), fetch('assets/sprites.json').then((r) => r.json())]);
  for (const s of sprites) s.bits = decodeMask(s.mask, scene.mask_grid);
  const assets = await loadAssets('assets', sprites);
  renderer = new Renderer($('#view'), assets, scene);
  reset();

  bind('#beltSpeed', 'beltSpeed', (v) => `${v.toFixed(2)} m/s`);
  bind('#rate', 'rate', (v) => `${(v * 60).toFixed(0)} obj/min`);
  bind('#confThr', 'confThr', (v) => v.toFixed(2));
  bind('#gripFail', 'gripFail', (v) => `${(v * 100).toFixed(0)}%`);
  bind('#policy', 'policy', null, String);
  $('#arms').addEventListener('change', (e) => {
    P.armsPerStation = Number(e.target.value);
    for (const a of cell.arms) if (a.held) a.held.state = 'gone';
    for (const tr of tracker.tracks) tr.claimed = false;
    cell = new RobotCell(P.armsPerStation, P.seed);
  });
  for (const [id, k] of [['#showDets', 'showDets'], ['#showTracks', 'showTracks'], ['#showPlan', 'showPlan']]) {
    $(id).addEventListener('change', (e) => { ui[k] = e.target.checked; });
  }
  $('#pause').addEventListener('click', (e) => { paused = !paused; e.target.textContent = paused ? 'Reanudar' : 'Pausar'; });
  $('#reset').addEventListener('click', reset);

  // models
  let models = [];
  try { models = (await fetch('models/models.json').then((r) => r.json())).models; } catch { /* none yet */ }
  if (qs.get('model')) models.push({ id: 'custom', name: 'Modelo de prueba', file: qs.get('model'), imgsz: (qs.get('imgsz') || '480,640').split(',').map(Number), available: true });
  const sel = $('#perception');
  for (const m of models) {
    const o = document.createElement('option'); o.value = m.id; o.disabled = !m.available;
    o.textContent = m.available ? `Modelo: ${m.name}` : `Modelo: ${m.name} (pendiente de entrenar)`;
    sel.appendChild(o);
  }
  const choose = async () => {
    const v = sel.value;
    if (v === 'oracle') { P.perception = 'oracle'; setStatus('Oráculo (verdad del simulador)', 'ok'); return; }
    try {
      await loadModel(models.find((m) => m.id === v), $('#gpu').checked ? 'webgpu' : 'wasm');
      P.perception = v; metrics.perc = { tp: 0, fp: 0, fn: 0, ok: 0, frames: 0, msSum: 0 };
    } catch (err) { setStatus('No se pudo cargar el modelo: ' + err.message, 'err'); sel.value = 'oracle'; P.perception = 'oracle'; }
  };
  sel.addEventListener('change', choose); $('#gpu').addEventListener('change', () => sel.value !== 'oracle' && choose());
  const first = qs.get('model') ? 'custom' : models.find((m) => m.available && m.default)?.id;
  if (first) { sel.value = first; choose(); } else setStatus('Oráculo (verdad del simulador)', 'ok');

  setInterval(refreshPanel, 250);
  requestAnimationFrame(frame);
  window.__sim = { get world() { return world; }, get metrics() { return metrics; }, get cell() { return cell; }, get tracker() { return tracker; }, P, get model() { return model; } };
}

boot().catch((e) => { document.body.insertAdjacentHTML('afterbegin', `<pre class="fatal">${e.stack || e}</pre>`); });
