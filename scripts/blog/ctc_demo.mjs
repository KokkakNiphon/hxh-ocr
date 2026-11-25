// Data for the blog's CTC animation: one example line through the exact browser reader
// (web/ocr.js prepareLine + the int8 wasm engine), with the best class and its probability for
// every output column. Writes web/blog/data/ctc.json and web/blog/img/ctc_strip.pgm (the 48 x 768
// strip the model saw; scripts/blog/ctc_demo.py turns it into a PNG).
//
//   python scripts/blog/ctc_demo.py            (runs this script too)
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { HxhEngine, ctcDetails, idsToText, prepareLine, toRomaji } from "../../web/ocr.js";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const [pgmPath, label] = process.argv.slice(2);
const buf = readFileSync(pgmPath);
const [, w, h] = buf.toString("latin1", 0, 32).split(/\s+/).map(Number);
const gray = new Uint8Array(buf.buffer, buf.byteOffset + buf.length - w * h, w * h);

const mj = JSON.parse(readFileSync(join(root, "web/models/model.json"), "utf8"));
const vocab = mj.vocab;
const eng = await HxhEngine.create(readFileSync(join(root, "web/hxh_infer.wasm")), readFileSync(join(root, "web/models/model.hxhm")));
const prep = prepareLine(gray, w, h, { height: eng.height, width: eng.width });
const out = eng.run(prep.pixels);
const T = eng.timesteps, C = eng.channels, K = eng.numClasses, scale = 2 ** eng.logitsExp;

const columns = [];
for (let t = 0; t < T; t++) {
  const row = t * C;
  let best = 0;
  for (let c = 1; c < K; c++) if (out.logits[row + c] > out.logits[row + best]) best = c;
  let z = 0;
  const e = [];
  for (let c = 0; c < K; c++) z += e[c] = Math.exp((out.logits[row + c] - out.logits[row + best]) * scale);
  // the three most likely classes at this column ("" = blank), for the "top guesses" view
  const top = e.map((v, c) => [c, v / z]).sort((a, b) => b[1] - a[1]).slice(0, 3)
    .map(([c, p]) => [c === 0 ? "" : vocab[c - 1], +p.toFixed(4)]);
  columns.push({ id: best, ch: best === 0 ? "" : vocab[best - 1], p: +(1 / z).toFixed(4), top });
}
const text = idsToText(out.ids, vocab);
const chars = ctcDetails(out.logits, T, C, K, eng.logitsExp).map((c) => ({ ch: vocab[c.id - 1], t0: c.t0, t1: c.t1, p: +c.prob.toFixed(4) }));
writeFileSync(join(root, "web/blog/data/ctc.json"), JSON.stringify({
  source: "web/examples/" + pgmPath.split(/[\\/]/).pop().replace(".pgm", ".png"),
  label, text, romaji: toRomaji(text), match: text === label,
  height: eng.height, width: eng.width, contentWidth: prep.contentWidth, timesteps: T, stride: eng.width / T,
  columns, chars,
}, null, 0));
writeFileSync(join(root, "web/blog/img/ctc_strip.pgm"), Buffer.concat([Buffer.from(`P5\n${eng.width} ${eng.height}\n255\n`), Buffer.from(prep.pixels)]));
console.log(`${text} (label ${label}) · ${T} columns, ${columns.filter((c) => c.id).length} non-blank, ${chars.length} characters`);
