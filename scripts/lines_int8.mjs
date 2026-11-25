// Read line crops with an int8 reader exactly as the app reads a detected line (web/ocr.js
// readCrop: area resize to the model height, replicate-right padding, squeeze on overflow, WASM
// SIMD engine, greedy CTC in the engine). Driven by scripts/eval_lines_int8.py.
//
//   node scripts/lines_int8.mjs <crops.bin> <model.hxhm> <model.json> > readings.json
//
// crops.bin: u32 count, then per crop u32 width, u32 height, width*height u8 (upright; vertical
// columns already turned 90 degrees counter-clockwise). Output: [[text, confidence], ...].
import { readFileSync } from "node:fs";
import { HxhEngine, idsToText, readCrop } from "../web/ocr.js";

const [binPath, modelPath, jsonPath] = process.argv.slice(2);
const wasm = await WebAssembly.compile(readFileSync(new URL("../web/hxh_infer.wasm", import.meta.url)));
const engine = await HxhEngine.create(wasm, readFileSync(modelPath));
const vocab = JSON.parse(readFileSync(jsonPath, "utf8")).vocab;
const b = readFileSync(binPath);
const n = b.readUInt32LE(0);
let off = 4;
const out = [];
for (let i = 0; i < n; i++) {
  const w = b.readUInt32LE(off), h = b.readUInt32LE(off + 4);
  off += 8;
  const pixels = new Uint8Array(b.buffer, b.byteOffset + off, w * h);
  off += w * h;
  const r = readCrop(engine, { pixels, w, h });
  out.push([idsToText(r.ids, vocab), Math.round(r.conf * 1e4) / 1e4]);
}
process.stdout.write(JSON.stringify(out));
