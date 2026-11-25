// End-to-end check of full-page reading in node, through exactly the browser code path
// (web/ocr.js readPage: Rust letterbox -> int8 detector -> Rust post-processing -> crops ->
// int8 recognisers), on generated pages with ground truth (make_pages.py).
//
//   .venv\Scripts\python.exe web\test\make_pages.py --offset 900 --n 100
//   node web/test/page_e2e.mjs [--wasm rust/target/web-simd/.../hxh_infer_wasm.wasm]
//        [--det web/models/det.hxhm] [--rec web/models/model.hxhm] [--rec-v web/models/model_v.hxhm]
//        [--pad 0.3,0.2] [--unclip 1.5 | --shrink 0.4] [--sweep | --conf-sweep]
//
// Detection: a predicted line matches a ground-truth line at quad IoU >= 0.5 (greedy by IoU).
// CER counts every ground-truth line; missed lines count as full deletions. --sweep tries
// several crop paddings and unclip settings.
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { HxhDetector, HxhEngine, idsToText, readPage } from "../ocr.js";

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, "..", "..");
const argv = process.argv.slice(2);
const opt = (name, def) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : def;
};
const wasmPath = opt("--wasm", join(root, "web", "hxh_infer.wasm"));
const detPath = opt("--det", join(root, "web", "models", "det.hxhm"));
const recPath = opt("--rec", join(root, "web", "models", "model.hxhm"));
const recVPath = opt("--rec-v", join(root, "web", "models", "model_v.hxhm"));
const pagesDir = opt("--pages", join(here, "out", "pages"));
const [padAlong, padAcross] = opt("--pad", "0.3,0.2").split(",").map(Number);
const sweep = argv.includes("--sweep");
const confSweep = argv.includes("--conf-sweep");
const shortScore = Number(opt("--short-score", "0")); // one-symbol readings need at least this detector score // drop lines whose reading confidence is below t
const baseParams = {};
if (argv.includes("--unclip")) baseParams.unclip_ratio = Number(opt("--unclip"));
if (argv.includes("--shrink")) baseParams.shrink_ratio = Number(opt("--shrink"));

// the reader's vocab: --rec-json, else the .json next to --rec (model family dirs), else web/models
const recJson = opt("--rec-json", existsSync(recPath.replace(/\.hxhm$/, ".json")) ? recPath.replace(/\.hxhm$/, ".json") : join(root, "web", "models", "model.json"));
const vocab = JSON.parse(readFileSync(recJson, "utf8")).vocab;
const mod = await WebAssembly.compile(readFileSync(wasmPath));
const det = await HxhDetector.create(mod, readFileSync(detPath));
const rec = { h: await HxhEngine.create(mod, readFileSync(recPath)) };
try {
  rec.v = await HxhEngine.create(mod, readFileSync(recVPath));
} catch (e) {
  console.log(`no vertical recogniser (${e.message}); vertical lines use the horizontal one`);
}
console.log(`detector ${det.size}x${det.size} -> ${det.outH}x${det.outW}, ${(det.macs / 1e6).toFixed(0)} M MACs; ` +
  `recognisers h${rec.v ? " + v" : ""}`);

const pages = JSON.parse(readFileSync(join(pagesDir, "pages.json"), "utf8")).map((p) => {
  const buf = readFileSync(join(pagesDir, p.file));
  return { ...p, gray: new Uint8Array(buf.buffer, buf.byteOffset + buf.length - p.width * p.height, p.width * p.height) };
});

// ---------------------------------------------------------------- geometry + metrics
const poly = (q) => [0, 1, 2, 3].map((i) => [q[2 * i], q[2 * i + 1]]);
const area = (P) => Math.abs(P.reduce((s, [x, y], i) => { const [u, v] = P[(i + 1) % P.length]; return s + x * v - u * y; }, 0)) / 2;
const ccw = (P) => (P.reduce((s, [x, y], i) => { const [u, v] = P[(i + 1) % P.length]; return s + x * v - u * y; }, 0) < 0 ? P.slice().reverse() : P);
function clip(subject, clipper) { // Sutherland-Hodgman, both convex and counter-clockwise (y down: any consistent winding)
  let out = subject;
  for (let i = 0; i < clipper.length && out.length; i++) {
    const [ax, ay] = clipper[i], [bx, by] = clipper[(i + 1) % clipper.length];
    const side = ([x, y]) => (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    const inp = out;
    out = [];
    for (let j = 0; j < inp.length; j++) {
      const P = inp[j], Q = inp[(j + 1) % inp.length], sp = side(P), sq = side(Q);
      if (sp >= 0) out.push(P);
      if ((sp >= 0) !== (sq >= 0)) { const t = sp / (sp - sq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); }
    }
  }
  return out;
}
function iou(qa, qb) {
  const A = ccw(poly(qa)), B = ccw(poly(qb));
  const I = clip(A, B);
  const ai = I.length > 2 ? area(I) : 0;
  return ai / (area(A) + area(B) - ai || 1);
}
function edit(a, b) {
  a = [...a]; b = [...b];
  const d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = d[0];
    d[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const t = d[j];
      d[j] = Math.min(d[j] + 1, d[j - 1] + 1, prev + (a[i - 1] !== b[j - 1]));
      prev = t;
    }
  }
  return d[b.length];
}

function evaluate(params, pad, verbose = false, minConf = 0, minScore = 0) {
  const acc = { tp: 0, fp: 0, fn: 0, orientOk: 0, h: { e: 0, n: 0, lines: 0, exact: 0 }, v: { e: 0, n: 0, lines: 0, exact: 0 },
    matched: { e: 0, n: 0 }, tDet: 0, tRead: 0, tAll: 0, nLines: 0, fpText: 0, fpIcon: 0, fpOther: 0 };
  for (const p of pages) {
    const t0 = performance.now();
    const r = readPage(det, rec, p.gray, p.width, p.height, { params, pad, tiles: argv.includes("--tiles") });
    if (minConf > 0 || minScore > 0) r.lines = r.lines.filter((l) => l.conf >= minConf && l.score >= minScore
      && (shortScore <= 0 || idsToText(l.ids, vocab).trim().length >= 2 || l.score >= shortScore));
    acc.tAll += performance.now() - t0;
    acc.tDet += r.tLetterbox + r.tDetect + r.tPost;
    acc.tRead += r.tRead;
    acc.nLines += r.lines.length;
    const pairs = [];
    p.lines.forEach((g, gi) => r.lines.forEach((l, li) => { const v = iou(g.quad, l.quad); if (v >= 0.5) pairs.push([v, gi, li]); }));
    pairs.sort((a, b) => b[0] - a[0]);
    const gUsed = new Map(), lUsed = new Set();
    for (const [, gi, li] of pairs) if (!gUsed.has(gi) && !lUsed.has(li)) { gUsed.set(gi, li); lUsed.add(li); }
    acc.tp += gUsed.size; acc.fp += r.lines.length - lUsed.size; acc.fn += p.lines.length - gUsed.size;
    // unmatched lines the app would show (non-empty reading): on non-Hunter text, on icons, or elsewhere
    r.lines.forEach((l, li) => {
      if (lUsed.has(li) || !idsToText(l.ids, vocab).trim()) return;
      const L = ccw(poly(l.quad)), aL = area(L) || 1;
      let best = 0, script = "";
      for (const f of p.foreign || []) {
        const I = clip(L, ccw(poly(f.quad)));
        const v = I.length > 2 ? area(I) / aL : 0;
        if (v > best) { best = v; script = f.script; }
      }
      if (best >= 0.4) script === "icon" ? acc.fpIcon++ : acc.fpText++;
      else acc.fpOther++;
    });
    p.lines.forEach((g, gi) => {
      const k = g.orient === "v" ? acc.v : acc.h;
      const n = [...g.text].length;
      k.n += n; k.lines++;
      if (!gUsed.has(gi)) { k.e += n; return; }
      const l = r.lines[gUsed.get(gi)];
      if (l.vertical === (g.orient === "v")) acc.orientOk++;
      const text = idsToText(l.ids, vocab);
      const e = edit(text, g.text);
      k.e += e; acc.matched.e += e; acc.matched.n += n;
      if (e === 0) k.exact++;
      if (verbose && e > 0 && Math.random() < 0.02) console.log(`    ${g.orient} "${g.text}" -> "${text}"`);
    });
  }
  const P = acc.tp / (acc.tp + acc.fp || 1), R = acc.tp / (acc.tp + acc.fn || 1);
  return {
    P, R, F1: (2 * P * R) / (P + R || 1), orient: acc.orientOk / (acc.tp || 1),
    cerH: acc.h.e / (acc.h.n || 1), cerV: acc.v.e / (acc.v.n || 1), cerMatched: acc.matched.e / (acc.matched.n || 1),
    exactH: acc.h.exact / (acc.h.lines || 1), exactV: acc.v.exact / (acc.v.lines || 1),
    msPage: acc.tAll / pages.length, msDet: acc.tDet / pages.length, msLine: acc.tRead / (acc.nLines || 1),
    fpText: acc.fpText / pages.length, fpIcon: acc.fpIcon / pages.length, fpOther: acc.fpOther / pages.length,
  };
}

const pct = (x) => `${(100 * x).toFixed(2)}%`;
const show = (tag, m) => console.log(`${tag.padEnd(34)} P ${pct(m.P)} R ${pct(m.R)} F1 ${pct(m.F1)} orient ${pct(m.orient)} | ` +
  `CER h ${pct(m.cerH)} v ${pct(m.cerV)} matched ${pct(m.cerMatched)} | exact h ${pct(m.exactH)} v ${pct(m.exactV)} | ` +
  `${m.msPage.toFixed(1)} ms/page (detect ${m.msDet.toFixed(1)}, ${m.msLine.toFixed(1)} ms/line)` +
  ` | shown false lines per page: on text ${m.fpText.toFixed(2)}, on icons ${m.fpIcon.toFixed(2)}, other ${m.fpOther.toFixed(2)}`);

console.log(`${pages.length} pages, ${pages.reduce((s, p) => s + p.lines.length, 0)} lines`);
if (argv.includes("--score-sweep")) { // shown = reading confidence >= 0.8 and detector score >= t
  for (const t of (opt("--scores", "0,0.55,0.6,0.65,0.7")).split(",").map(Number)) {
    show(`conf 0.8, min detector score ${t}`, evaluate(baseParams, { along: padAlong, across: padAcross }, false, 0.8, t));
  }
} else if (confSweep) {
  for (const t of [0, 0.8, 0.85, 0.9, 0.93, 0.95]) {
    show(`min reading confidence ${t}`, evaluate(baseParams, { along: padAlong, across: padAcross }, false, t));
  }
} else if (!sweep) {
  show(`pad ${padAlong},${padAcross} ${JSON.stringify(baseParams)}`, evaluate(baseParams, { along: padAlong, across: padAcross }, argv.includes("--verbose")));
} else {
  for (const params of [{ unclip_ratio: 1.5 }, { unclip_ratio: 1.0 }, { unclip_ratio: 2.0 }, { shrink_ratio: 0.4 }]) {
    for (const [a, c] of [[0, 0], [0.3, 0.2], [0.3, 0.3], [0.5, 0.3]]) {
      show(`${JSON.stringify(params)} pad ${a},${c}`, evaluate(params, { along: a, across: c }));
    }
  }
}
