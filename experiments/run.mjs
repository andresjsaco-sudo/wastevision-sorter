// Sweeps of the sorting line with oracle perception (what the robots could do with perfect
// vision) and with a degraded perception model. Results go to experiments/results.json.
import fs from 'node:fs';
import { simulate } from './sim.mjs';

const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
function avg(o, seeds = [1, 2, 3]) {
  const r = seeds.map((seed) => simulate({ ...o, seed }));
  return { recovery: mean(r.map((x) => x.recovery)), purity: mean(r.map((x) => x.purity ?? 1)), ppm: mean(r.map((x) => x.ppm)),
    errMm: mean(r.map((x) => x.errMm)), busy: mean(r.map((x) => mean(x.busy))) };
}
const f = (v) => (v * 100).toFixed(1).padStart(6);
const out = {};

console.log('\nE1  load x robots per station (belt 0.30 m/s, EDF, oracle)\n rate/min  arms  recovery  purity   ppm  busy');
out.load = [];
for (const arms of [1, 2]) for (const rate of [1, 2, 3, 4, 6]) {
  const r = avg({ rate, armsPerStation: arms }); out.load.push({ rate, arms, ...r });
  console.log(`${String(rate * 60).padStart(9)} ${String(arms).padStart(5)} ${f(r.recovery)}% ${f(r.purity)}% ${r.ppm.toFixed(0).padStart(5)} ${f(r.busy)}%`);
}

console.log('\nE2  belt speed (120 obj/min, 1 robot per station, EDF, oracle)\n  m/s  recovery   ppm  err mm');
out.speed = [];
for (const beltSpeed of [0.15, 0.3, 0.45, 0.6, 0.8]) {
  const r = avg({ beltSpeed }); out.speed.push({ beltSpeed, ...r });
  console.log(`${beltSpeed.toFixed(2).padStart(5)} ${f(r.recovery)}% ${r.ppm.toFixed(0).padStart(5)} ${r.errMm.toFixed(2).padStart(7)}`);
}

console.log('\nE3  allocation policy under overload (240 obj/min, 1 robot per station, oracle)\n policy   recovery   ppm');
out.policy = [];
for (const policy of ['edf', 'nearest', 'fifo']) {
  const r = avg({ policy, rate: 4 }); out.policy.push({ policy, ...r });
  console.log(`${policy.padEnd(8)} ${f(r.recovery)}% ${r.ppm.toFixed(0).padStart(5)}`);
}

console.log('\nE4  sensitivity to perception errors (120 obj/min, 2 robots per station)\n miss  wrong-class  recovery  purity');
out.perception = [];
const classes = ['plastic', 'glass', 'cardboard', 'metal', 'paper', 'organic', 'other'];
for (const [miss, wrong] of [[0, 0], [0.05, 0.05], [0.1, 0.1], [0.2, 0.2], [0.1, 0.3]]) {
  const confusion = Object.fromEntries(classes.map((c) => [c, Object.fromEntries(classes.map((k) => [k, k === c ? 1 - wrong : wrong / 6]))]));
  const r = avg({ armsPerStation: 2, perception: { miss, confusion, hz: 5 } }); out.perception.push({ miss, wrong, ...r });
  console.log(`${f(miss)}% ${f(wrong)}%     ${f(r.recovery)}% ${f(r.purity)}%`);
}
fs.writeFileSync(new URL('results.json', import.meta.url), JSON.stringify(out, null, 1));
