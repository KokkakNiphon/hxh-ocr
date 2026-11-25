// Node end-to-end check of the text-line detector path in the wasm build: the toy detector
// (rust/hxh-infer/tests/data/toy_det, format v2: Resize + Concat, raw output) runs through
// hxh_det_image -> hxh_det_letterbox -> hxh_run -> hxh_det_detect -> hxh_det_crop, and
//   1. its int8 maps must equal ESP-PPQ's simulation (golden.bin),
//   2. its line records (16 f32) and the FNV-1a of every crop must equal the host's
//      (lines_default.txt / lines_shrink07.txt, written by
//      `cargo run --release -p hxh-detect --example detect -- golden ...`), for two parameter sets,
// plus checks on synthetic maps (hxh_det_detect_map with a letterbox frame: positions in
// original-image pixels, orientation, corner order, reading order), letterbox properties,
// hxh_det_crop_quad (exact copy / 90 degree turn) and timings at the real detector's size.
//
//   node hxh-infer-wasm/test_detect.mjs <engine.wasm> [--scalar]
// Exit code 0 only if everything matches.
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const wasmPath = args[0];
const scalar = args.includes("--scalar");
if (!wasmPath) {
  console.error("usage: node test_detect.mjs <engine.wasm> [--scalar]");
  process.exit(2);
}
const td = join(here, "..", "hxh-infer", "tests", "data", "toy_det");
const wasmModule = await WebAssembly.compile(readFileSync(wasmPath));
const imports = { env: { hxh_now: () => performance.now() } };
const instantiate = async () => (await WebAssembly.instantiate(wasmModule, imports)).exports;

let failures = 0;
const check = (ok, msg) => {
  if (!ok) {
    failures++;
    console.log(`FAIL: ${msg}`);
  }
  return ok;
};
const fnv1a = (bytes) => {
  let h = 0x811c9dc5;
  for (const b of bytes) h = Math.imul(h ^ b, 0x01000193) >>> 0;
  return h;
};
const PARAMS = { bin_thresh: 0, min_score: 1, min_side: 2, unclip_ratio: 3, shrink_ratio: 4, block_gap: 5 };
const setParams = (x, p) => {
  const f = new Float64Array(x.memory.buffer, x.hxh_det_params_ptr(), 8);
  f.set([0.3, 0.5, 3, 1.5, 0, 1, 0, 0]);
  for (const [k, v] of Object.entries(p)) f[PARAMS[k]] = v;
};
// views must be created after any call that may grow memory (growing detaches old buffers)
const u8 = (inst, p, n) => new Uint8Array(inst.memory.buffer, p, n);
// copies: a later crop may grow memory and detach any view
const records = (x, n) => Array.from({ length: n }, (_, i) => new Float32Array(x.memory.buffer, x.hxh_det_lines_ptr() + 64 * i, 16).slice());

// ------------------------------------------------------------------ 1 + 2: toy detector
const x = await instantiate();
const simd = x.hxh_set_simd(scalar ? 0 : 1) === 1;
console.log(`${resolve(wasmPath)}: ${simd ? "SIMD128" : "scalar"} kernels`);
const model = readFileSync(join(td, "model.hxhm"));
const mp = x.hxh_alloc(model.length);
new Uint8Array(x.memory.buffer, mp, model.length).set(model);
const rc = x.hxh_load(mp, model.length);
if (rc !== 0) throw new Error(`hxh_load: ${rc}`);
const [W, H, OH, OW, C] = [x.hxh_image_width(), x.hxh_image_height(), x.hxh_out_height(), x.hxh_out_width(), x.hxh_channels()];
check(x.hxh_raw_output() === 1 && OH === 64 && OW === 64 && C === 2, `toy detector: raw output ${x.hxh_raw_output()}, ${OH}x${OW}x${C}`);

const g = readFileSync(join(td, "golden.bin"));
const gv = new DataView(g.buffer, g.byteOffset, g.byteLength);
const nVec = gv.getUint32(8, true);
const T = gv.getUint16(16, true) || gv.getUint32(24, true);
const maxIds = gv.getUint16(20, true);
const rec = 2 * H * W + T * C + 2 + 2 * maxIds;

function parseLines(file) {
  const out = [];
  for (const line of readFileSync(join(td, file), "utf8").split("\n")) {
    if (!line || line.startsWith("#")) continue;
    const t = line.trim().split(/\s+/);
    if (t[0] === "vector") out[Number(t[1])] = [];
    else out[out.length - 1].push({ rec: Float32Array.from(t.slice(0, 16).map(Number)), fnv: parseInt(t[16], 16) });
  }
  return out;
}

let mapsOk = 0;
const sets = [
  ["lines_default.txt", {}],
  ["lines_shrink07.txt", { shrink_ratio: 0.7, block_gap: 0.5 }],
];
for (const [file, p] of sets) {
  const want = parseLines(file);
  setParams(x, p);
  let lineOk = 0, lineTotal = 0, cropOk = 0;
  for (let v = 0; v < nVec; v++) {
    const base = 32 + v * rec;
    const img = g.subarray(base, base + H * W);
    // the golden image is the page itself: letterboxing a 128 x 128 page into 128 x 128 copies it
    const ip = x.hxh_det_image(W, H);
    u8(x, ip, W * H).set(img);
    check(x.hxh_det_letterbox() === 0, "hxh_det_letterbox");
    check(x.hxh_det_content_width() === W && x.hxh_det_content_height() === H, "content size");
    const lb = new Uint8Array(x.memory.buffer, x.hxh_image_ptr(), W * H);
    check(lb.every((b, i) => b === img[i]), `vector ${v}: letterbox of a page of the input size is a copy`);
    check(x.hxh_run() === 0, "hxh_run on a raw-output model returns 0 ids");
    if (file === sets[0][0]) {
      const out = new Int8Array(x.memory.buffer, x.hxh_logits_ptr(), T * C);
      const ref = new Int8Array(g.buffer, g.byteOffset + base + 2 * H * W, T * C);
      if (check(out.every((b, i) => b === ref[i]), `vector ${v}: detector maps differ from ESP-PPQ`)) mapsOk++;
    }
    const n = x.hxh_det_detect();
    check(n === want[v].length && x.hxh_det_line_count() === n, `${file} vector ${v}: ${n} lines, host ${want[v].length}`);
    const got = records(x, n);
    for (let i = 0; i < Math.min(n, want[v].length); i++) {
      lineTotal++;
      if (check(got[i].every((f, k) => Object.is(f, want[v][i].rec[k])), `${file} vector ${v} line ${i}: ${got[i]} vs ${want[v][i].rec}`)) lineOk++;
      const cp = x.hxh_det_crop(i);
      const [cw, ch] = [x.hxh_det_crop_width(), x.hxh_det_crop_height()];
      check(cw === got[i][14] && ch === got[i][15], "crop size = record[14..16]");
      if (check(cp !== 0 && fnv1a(new Uint8Array(x.memory.buffer, cp, cw * ch)) === want[v][i].fnv, `${file} vector ${v} crop ${i}`)) cropOk++;
    }
  }
  console.log(`${file}: line records identical to the host ${lineOk}/${lineTotal}, crops identical ${cropOk}/${lineTotal}`);
}
console.log(`toy detector maps bit-exact vs ESP-PPQ: ${mapsOk}/${nVec}`);
check(mapsOk === nVec, "maps");

// ------------------------------------------------------------------ 3: synthetic maps
{
  const y = await instantiate();
  const [h, w] = [80, 80];
  const mp = y.hxh_alloc(h * w * 2);
  const map = new Int8Array(y.memory.buffer, mp, h * w * 2).fill(-100);
  // (centre, axis angle, length, thickness, vertical), grid pixels
  const lines = [[[20, 15], 0.15, 24, 4, false], [[60, 40], 1.5, 30, 5, true], [[40, 60], 0.0, 30, 4, false], [[22, 40], 1.57, 20, 4, true]];
  for (const [[cx, cy], a, len, th, vert] of lines) {
    const [ux, uy, nx, ny] = [Math.cos(a), Math.sin(a), -Math.sin(a), Math.cos(a)];
    for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) {
      const [dx, dy] = [xx - cx, yy - cy];
      if (Math.abs(dx * ux + dy * uy) <= len / 2 && Math.abs(dx * nx + dy * ny) <= th / 2) {
        map[(yy * w + xx) * 2] = 64;
        map[(yy * w + xx) * 2 + 1] = vert ? 40 : -40;
      }
    }
  }
  // letterbox 160 x 160 holding a 160 x 140 image of a 640 x 560 page: grid -> page scale 8
  check(y.hxh_det_detect_map(mp, h, w, 2, -4) === -32, "detect_map without a frame is refused");
  y.hxh_det_set_frame(160, 160, 160, 140, 640, 560);
  setParams(y, { unclip_ratio: 0 });
  const n = y.hxh_det_detect_map(mp, h, w, 2, -4);
  const r = records(y, n);
  check(n === 4, `synthetic map: ${n} lines, want 4`);
  // every line is its own block (far apart or different orientation): blocks go top to bottom,
  // so the expected reading order is A (y 13), B (y 25), D (y 30), C (y 58)
  const k = 8;
  let ok = 0;
  const seq = [];
  for (let i = 0; i < n; i++) {
    const q = r[i];
    const c = [(q[0] + q[2] + q[4] + q[6]) / 4, (q[1] + q[3] + q[5] + q[7]) / 4];
    const j = lines.findIndex((l) => Math.hypot(c[0] - ((l[0][0] + 0.5) * k - 0.5), c[1] - ((l[0][1] + 0.5) * k - 0.5)) < k);
    seq.push(j);
    if (j < 0) continue;
    const [, , len, th, vert] = lines[j];
    const good = (q[9] === 1) === vert && Math.abs(q[12] / k - len) < 2 && Math.abs(q[13] / k - th) < 2 && q[10] === i
      && (vert ? q[0] > q[6] && q[3] > q[1] : q[2] > q[0] && q[7] > q[1]);
    if (check(good, `synthetic line ${i}: ${Array.from(q).map((v) => v.toFixed(1))}`)) ok++;
  }
  check(seq.join() === "0,1,3,2", `synthetic reading order ${seq}, want 0,1,3,2`);
  console.log(`synthetic map (letterbox frame, page scale 8): ${ok}/4 lines at the drawn place with the right ` +
    `orientation and corner order; reading order ${seq}`);

  // letterbox: a 1000 x 700 page into the toy's 128 x 128 input
  const z = await instantiate();
  const mm = z.hxh_alloc(model.length);
  new Uint8Array(z.memory.buffer, mm, model.length).set(model);
  z.hxh_load(mm, model.length);
  const pw = 1000, ph = 700;
  const pp = z.hxh_det_image(pw, ph);
  const page = () => u8(z, pp, pw * ph);
  page().fill(90);
  for (let yy = 300; yy < 400; yy++) page().fill(10, yy * pw + 200, yy * pw + 800);
  check(z.hxh_det_letterbox() === 0, "letterbox");
  const lb = new Uint8Array(z.memory.buffer, z.hxh_image_ptr(), 128 * 128);
  const [cw, ch] = [z.hxh_det_content_width(), z.hxh_det_content_height()];
  const pad = lb.every((v, i) => (i % 128 < cw && Math.floor(i / 128) < ch) || v === 255);
  // fully covered output pixels average to exactly 90 or 10
  const inside = lb[Math.floor(350 * 128 / pw) * 128 + 64] === 10 && lb[5 * 128 + 5] === 90;
  check(cw === 128 && ch === 90 && pad && inside, `letterbox 1000x700 -> content ${cw}x${ch}, padding 255 ${pad}, area values ${inside}`);
  // crop_quad: an axis-aligned box is an exact copy, a vertical column comes out turned CCW
  const setQuad = (q) => new Float32Array(z.memory.buffer, z.hxh_det_quad_ptr(), 8).set(q);
  const pg = page();
  for (let yy = 0; yy < ph; yy++) for (let xx = 0; xx < pw; xx++) pg[yy * pw + xx] = (xx * 7 + yy * 3) & 255;
  setQuad([99.5, 49.5, 139.5, 49.5, 139.5, 59.5, 99.5, 59.5]);
  let cp = z.hxh_det_crop_quad(0, 0);
  let crop = u8(z, cp, 40 * 10), P = page();
  check(z.hxh_det_crop_width() === 40 && z.hxh_det_crop_height() === 10 && crop.every((v, i) => v === P[(50 + Math.floor(i / 40)) * pw + 100 + (i % 40)]),
    "crop_quad of an axis-aligned box is a copy");
  setQuad([109.5, 49.5, 109.5, 89.5, 99.5, 89.5, 99.5, 49.5]); // column x 100..109, y 50..89, q0 = top-right
  cp = z.hxh_det_crop_quad(0, 0);
  crop = u8(z, cp, 40 * 10);
  P = page();
  check(z.hxh_det_crop_width() === 40 && z.hxh_det_crop_height() === 10 && crop.every((v, i) => v === P[(50 + (i % 40)) * pw + 109 - Math.floor(i / 40)]),
    "crop_quad of a vertical column is turned 90 degrees counter-clockwise");
  console.log("letterbox and crop_quad checks done");

  // timing at the real detector's size: a 320 x 320 x 2 map with 60 lines, and letterboxing a 2480 x 3508 page to 640
  const t = await instantiate();
  const G = 320;
  const bp = t.hxh_alloc(G * G * 2);
  const big = new Int8Array(t.memory.buffer, bp, G * G * 2).fill(-90);
  for (let l = 0; l < 60; l++) {
    const [x0, y0] = [10 + (l % 3) * 100, 10 + Math.floor(l / 3) * 15];
    for (let yy = y0; yy < y0 + 4; yy++) for (let xx = x0; xx < x0 + 80; xx++) big[(yy * G + xx) * 2] = 50;
  }
  t.hxh_det_set_frame(640, 640, 640, 640, 2480, 2480);
  t.hxh_det_detect_map(bp, G, G, 2, -3);
  let ts = [];
  for (let i = 0; i < 20; i++) {
    const t0 = performance.now();
    const m = t.hxh_det_detect_map(bp, G, G, 2, -3);
    ts.push(performance.now() - t0);
    if (i === 0) check(m === 60, `320x320 map: ${m} lines, want 60`);
  }
  ts.sort((a, b) => a - b);
  const tDetect = ts[10];
  const t2 = await instantiate();
  const m2 = t2.hxh_alloc(model.length);
  new Uint8Array(t2.memory.buffer, m2, model.length).set(model);
  t2.hxh_load(m2, model.length);
  // the toy model's input is 128: time the 640 letterbox through hxh-detect's own export path
  // by measuring the area resize of an A4 scan into the 128 input, scaled by output pixels
  const [sw, sh] = [2480, 3508];
  const sp = t2.hxh_det_image(sw, sh);
  u8(t2, sp, sw * sh).fill(200);
  ts = [];
  for (let i = 0; i < 5; i++) {
    const t0 = performance.now();
    t2.hxh_det_letterbox();
    ts.push(performance.now() - t0);
  }
  ts.sort((a, b) => a - b);
  console.log(`timing: post-processing of a 320x320x2 map with 60 lines ${tDetect.toFixed(2)} ms (median of 20); ` +
    `letterbox of a 2480x3508 page into the 128 px toy input ${ts[2].toFixed(1)} ms`);
}

console.log(failures ? `FAILED (${failures})` : "all detector checks passed");
process.exit(failures ? 1 : 0);
