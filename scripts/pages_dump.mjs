// Dump every line web/ocr.js readPage finds on a page set (no confidence / score filter): quad,
// orientation, reading, reading confidence, detector score. For per-line-type detector checks
// (scripts/det_check_v6.py).
//
//   node scripts/pages_dump.mjs <pages dir> <det.hxhm> <model.hxhm> <model_v.hxhm> [--tiles] > lines.json
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { HxhDetector, HxhEngine, idsToText, readPage } from "../web/ocr.js";

const [dir, detPath, recPath, recVPath] = process.argv.slice(2);
const root = new URL("..", import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1");
const wasm = await WebAssembly.compile(readFileSync(join(root, "web", "hxh_infer.wasm")));
const det = await HxhDetector.create(wasm, readFileSync(detPath));
const rec = { h: await HxhEngine.create(wasm, readFileSync(recPath)), v: await HxhEngine.create(wasm, readFileSync(recVPath)) };
const detJson = detPath.replace(/\.hxhm$/, ".json");
const meta = JSON.parse(readFileSync(existsSync(detJson) ? detJson : join(root, "web", "models", "det.json"), "utf8"));
const recJson = recPath.replace(/\.hxhm$/, ".json");
const vocab = JSON.parse(readFileSync(existsSync(recJson) ? recJson : join(root, "web", "models", "model.json"), "utf8")).vocab;
const pages = JSON.parse(readFileSync(join(dir, "pages.json"), "utf8"));
const out = [];
for (const p of pages) {
  const buf = readFileSync(join(dir, p.file));
  const gray = new Uint8Array(buf.buffer, buf.byteOffset + buf.length - p.width * p.height, p.width * p.height);
  const r = readPage(det, rec, gray, p.width, p.height, { params: meta.params || {}, pad: meta.pad || { along: 0.3, across: 0.2 }, tiles: process.argv.includes("--tiles") });
  out.push({ file: p.file, source: p.source, lines: r.lines.map((l) => ({ quad: l.quad, vertical: l.vertical, text: idsToText(l.ids, vocab), conf: l.conf, score: l.score })) });
}
process.stdout.write(JSON.stringify(out));
