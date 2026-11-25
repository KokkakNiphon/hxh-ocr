// Blog assets: run one page through exactly the browser pipeline (web/ocr.js readPage on
// web/hxh_infer.wasm with web/models/{det,model,model_v}.hxhm and det.json's params/pad) and
// dump every intermediate the article shows. Called by scripts/blog/make_assets.py.
//
//   node scripts/blog/pipeline_assets.mjs --page page.pgm --out <dir> [--runs 7] [--line-reps 5]
//
// <dir> receives:
//   letterbox.u8   the S x S detector input, copied from wasm memory (hxh_image_ptr) right after
//                  hxh_det_letterbox
//   maps.i8        the detector's raw int8 output, NHWC [outH, outW, channels] (hxh_logits_ptr):
//                  channel 0 text-kernel logit, channel 1 vertical-orientation logit;
//                  value * 2^exp is the float logit
//   strip_N.u8     line N's reader input (prep.pixels, height x width, row-major), reading order
//   meta.json      sizes, timings (median over --runs readPage calls) and every line
//
// The instrumented pass calls the same exports in the same order as HxhDetector.detect, then
// readPage itself is run for the timings and the lines; the two detections are checked equal.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { cpus } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  ctcDetails, DET_PARAMS, expandQuad, HxhDetector, HxhEngine, idsToText, readCrop, readPage, toRomaji,
} from "../../web/ocr.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const argv = process.argv.slice(2);
const opt = (name, def) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : def;
};
const pagePath = opt("--page");
const outDir = opt("--out");
const runs = Number(opt("--runs", 7));
const lineReps = Number(opt("--line-reps", 5));
if (!pagePath || !outDir) throw new Error("usage: --page page.pgm --out dir");
mkdirSync(outDir, { recursive: true });

// ---------------------------------------------------------------- inputs
function readPgm(path) {
  const buf = readFileSync(path);
  const head = buf.subarray(0, 64).toString("latin1").split(/\s+/);
  if (head[0] !== "P5") throw new Error("not a binary PGM");
  const W = Number(head[1]), H = Number(head[2]);
  return { W, H, gray: new Uint8Array(buf.buffer, buf.byteOffset + buf.length - W * H, W * H).slice() };
}
const { W, H, gray } = readPgm(pagePath);
const models = join(root, "web", "models");
const detCfg = JSON.parse(readFileSync(join(models, "det.json"), "utf8"));
const vocab = JSON.parse(readFileSync(join(models, "model.json"), "utf8")).vocab;
const mod = await WebAssembly.compile(readFileSync(join(root, "web", "hxh_infer.wasm")));
const det = await HxhDetector.create(mod, readFileSync(join(models, "det.hxhm")));
const rec = {
  h: await HxhEngine.create(mod, readFileSync(join(models, "model.hxhm"))),
  v: await HxhEngine.create(mod, readFileSync(join(models, "model_v.hxhm"))),
};
const opts = { params: detCfg.params || {}, pad: detCfg.pad || {}, invert: false };

// ---------------------------------------------------------------- instrumented detection
const x = det.x;
const check = (rc) => { if (rc < 0) throw new Error(`hxh_det error ${rc}`); return rc; };
const pp = x.hxh_det_image(W, H);
new Uint8Array(x.memory.buffer, pp, W * H).set(gray);
check(x.hxh_det_letterbox());
const S = det.size;
const letterbox = new Uint8Array(x.memory.buffer, x.hxh_image_ptr(), S * S).slice();
const contentW = x.hxh_det_content_width(), contentH = x.hxh_det_content_height();
check(x.hxh_run());
const outH = x.hxh_out_height(), outW = x.hxh_out_width(), channels = x.hxh_channels(), exp = x.hxh_logits_exp();
const maps = new Int8Array(x.memory.buffer, x.hxh_logits_ptr(), outH * outW * channels).slice();
const p = new Float64Array(x.memory.buffer, x.hxh_det_params_ptr(), 8);
for (const [k, v] of Object.entries(opts.params)) if (k in DET_PARAMS) p[DET_PARAMS[k]] = v;
const nRaw = check(x.hxh_det_detect());
const rawQuads = Array.from(new Float32Array(x.memory.buffer, x.hxh_det_lines_ptr(), 16 * nRaw))
  .filter((_, i) => i % 16 < 8);
writeFileSync(join(outDir, "letterbox.u8"), letterbox);
writeFileSync(join(outDir, "maps.i8"), new Uint8Array(maps.buffer));

// ---------------------------------------------------------------- readPage: timings + lines
const median = (a) => { const s = [...a].sort((u, v) => u - v); return s[s.length >> 1]; };
const now = () => performance.now();
for (let i = 0; i < 2; i++) readPage(det, rec, gray, W, H, opts); // warm-up (JIT, memory growth)
const T = { letterbox: [], detector: [], post: [], read: [], total: [] };
let r;
for (let i = 0; i < runs; i++) {
  const t0 = now();
  r = readPage(det, rec, gray, W, H, opts);
  T.total.push(now() - t0);
  T.letterbox.push(r.tLetterbox); T.detector.push(r.tDetect); T.post.push(r.tPost); T.read.push(r.tRead);
}
const same = r.lines.length === nRaw && r.lines.every((l, i) => l.quad.every((v, j) => v === rawQuads[8 * i + j]));
if (!same) throw new Error("readPage lines differ from the instrumented detection");

// per-line read time: crop of the padded quad + prepareLine + reader, as recogniseLines does
const lineMs = r.lines.map((l) => {
  const ts = [];
  for (let k = 0; k < lineReps; k++) {
    const t0 = now();
    const crop = det.cropQuad(expandQuad(l.quad, opts.pad));
    readCrop(l.vertical ? rec.v : rec.h, crop, { invert: false });
    ts.push(now() - t0);
  }
  return median(ts);
});

const lines = r.lines.map((l, i) => {
  const text = idsToText(l.ids, vocab);
  const e = l.engine;
  writeFileSync(join(outDir, `strip_${i}.u8`), l.prep.pixels);
  const chars = ctcDetails(l.logits, e.timesteps, e.channels, e.numClasses, e.logitsExp).map((c) => ({
    ch: c.id >= 1 && c.id <= vocab.length ? vocab[c.id - 1] : "?", p: +c.prob.toFixed(4),
    alt: c.alt >= 1 ? vocab[c.alt - 1] : null, alt_p: +c.altProb.toFixed(4),
  }));
  return {
    index: i, quad: l.quad.map((v) => +v.toFixed(2)), vertical: l.vertical, block: l.block,
    length: +l.length.toFixed(2), thickness: +l.thickness.toFixed(2), det_score: +l.score.toFixed(4),
    text, romaji: toRomaji(text), confidence: +l.conf.toFixed(4), chars,
    crop: { w: l.crop.w, h: l.crop.h },
    strip: { file: `strip_${i}.u8`, width: l.prep.width, height: l.prep.height, content_width: l.prep.contentWidth,
      natural_width: l.prep.naturalWidth, squeezed: l.prep.squeezed, upscaled: l.prep.upscaled },
    read_ms: +lineMs[i].toFixed(2),
  };
});

const round = (v) => +v.toFixed(2);
const meta = {
  page: { width: W, height: H },
  letterbox: { size: S, content_width: contentW, content_height: contentH, pad_value: 255 },
  maps: { height: outH, width: outW, channels, exp, layout: "NHWC int8; channel 0 text-kernel logit, channel 1 vertical-orientation logit" },
  detector: { name: detCfg.name, macs: det.macs, params: opts.params, pad: opts.pad, min_conf: detCfg.min_conf },
  readers: { h: { width: rec.h.width, height: rec.h.height, macs: rec.h.macs }, v: { width: rec.v.width, height: rec.v.height, macs: rec.v.macs } },
  timings_ms: {
    letterbox: round(median(T.letterbox)), detector: round(median(T.detector)), post: round(median(T.post)),
    read_all_lines: round(median(T.read)), read_per_line_mean: round(median(T.read) / Math.max(1, lines.length)),
    page_total: round(median(T.total)),
    runs, warmup: 2, line_reps: lineReps,
    note: "median over `runs` readPage calls after 2 warm-ups (readPage reads every detected line, empty ones " +
      "included; read_all_lines covers crop + prepareLine + reader for all of them); read_ms per line is the median of " +
      "`line_reps` crop + prepareLine + reader calls; node, one thread, wasm SIMD, CPU shared with other jobs",
  },
  machine: { node: process.version, cpu: cpus()[0]?.model?.trim(), threads: 1, simd: true, cpu_shared: true },
  line_count: lines.length,
  lines,
};
writeFileSync(join(outDir, "meta.json"), JSON.stringify(meta, null, 1));
console.log(`${lines.length} lines; letterbox ${meta.timings_ms.letterbox} ms, detector ${meta.timings_ms.detector} ms, ` +
  `post ${meta.timings_ms.post} ms, read ${meta.timings_ms.read_all_lines} ms (${meta.timings_ms.read_per_line_mean} ms/line)`);
for (const l of lines) console.log(`  ${l.index} ${l.vertical ? "v" : "h"} ${l.text} (${l.romaji}) conf ${l.confidence}`);
