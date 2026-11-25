// Node check of the wasm build: bit-exactness against golden.bin files and timing.
//
//   node hxh-infer-wasm/test_wasm.mjs <engine.wasm> <model.hxhm> <golden.bin> [<model> <golden>]...
//   node hxh-infer-wasm/test_wasm.mjs <engine.wasm> --all        # the 3 toy sets + the real models
//
// Options:
//   --all           every golden set: rust/hxh-infer/tests/data/{toy_tqt,toy_mean,toy_random,toy_det} and
//                   experiments/rust_engine/{mnv4s_h32_ptq_holdout, v3_mnv4s_h32_w512_tqt,
//                   qat_v3_mnv4s_h32_ptq, qat_v3_mnv4s3_h32_ptq, qat_v3_mnv4s_h48_ptq}
//   --scalar        run the scalar kernels in a SIMD build (hxh_set_simd(0))
//   --bench-ms N    steady-state timing budget per model in ms (default 3000; 0 = skip timing)
//   --selftest N    first run N random scalar-vs-SIMD kernel cases (needs a build with
//                   `--features selftest`)
//   --profile       per-op time table and a summary by op class (needs `--features profile`)
// Each model gets a fresh instance (the wasm bump allocator loads one model per instance).
// Exit code 0 only if every logit and every decode matches.
import { readFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const wasmPath = args.shift();
let scalar = false, benchMs = 3000, selftest = 0, profile = false;
const pairs = [];
while (args.length) {
  const a = args.shift();
  if (a === "--scalar") scalar = true;
  else if (a === "--bench-ms") benchMs = Number(args.shift());
  else if (a === "--selftest") selftest = Number(args.shift());
  else if (a === "--profile") profile = true;
  else if (a === "--all") {
    const td = join(here, "..", "hxh-infer", "tests", "data");
    for (const m of ["toy_tqt", "toy_mean", "toy_random", "toy_det"]) pairs.push([join(td, m, "model.hxhm"), join(td, m, "golden.bin")]);
    const ex = join(here, "..", "..", "experiments", "rust_engine");
    for (const m of ["mnv4s_h32_ptq_holdout", "v3_mnv4s_h32_w512_tqt", "qat_v3_mnv4s_h32_ptq", "qat_v3_mnv4s3_h32_ptq",
      "qat_v3_mnv4s_h48_ptq"]) pairs.push([join(ex, m, "model.hxhm"), join(ex, m, "golden.bin")]);
  } else pairs.push([a, args.shift()]);
}
if (!wasmPath || (!pairs.length && !selftest) || pairs.some(([m, g]) => !m || !g)) {
  console.error("usage: node test_wasm.mjs <engine.wasm> (--all | <model.hxhm> <golden.bin> ...) [--scalar] [--bench-ms N] [--selftest N]");
  process.exit(2);
}
const wasmBytes = readFileSync(wasmPath);
const wasmModule = await WebAssembly.compile(wasmBytes);
// the `profile` feature imports a millisecond clock; release builds import nothing
const imports = { env: { hxh_now: () => performance.now() } };
const instantiate = async () => (await WebAssembly.instantiate(wasmModule, imports)).exports;
const probe = await instantiate();
const simdBuild = probe.hxh_set_simd ? probe.hxh_set_simd(1) === 1 : false;
const kernels = simdBuild && !scalar ? "SIMD128" : "scalar";
console.log(`${resolve(wasmPath)}: ${wasmBytes.length} bytes, ${simdBuild ? "SIMD build" : "no SIMD"}, running ${kernels} kernels`);

let failed = false;
if (selftest) {
  if (!probe.hxh_selftest) throw new Error("this build has no hxh_selftest (cargo --features selftest)");
  let bad = 0;
  const t0 = performance.now();
  for (let seed = 1; seed <= 4; seed++) bad += probe.hxh_selftest(seed, Math.ceil(selftest / 4));
  console.log(`selftest: ${probe.hxh_selftest_cases()} random cases compared (conv / depthwise, Add + Requant, AvgPool, ` +
    `Resize + Concat), ` +
    `${bad} mismatching (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  if (bad !== 0) failed = true;
}

const rows = [];
for (const [modelPath, goldenPath] of pairs) {
  if (!existsSync(modelPath) || !existsSync(goldenPath)) {
    console.log(`skip ${modelPath}: missing model or golden file`);
    failed = true;
    continue;
  }
  const x = await instantiate();
  const mem = () => x.memory.buffer;
  if (x.hxh_set_simd) x.hxh_set_simd(scalar ? 0 : 1);
  const model = readFileSync(modelPath);
  const mp = x.hxh_alloc(model.length);
  new Uint8Array(mem(), mp, model.length).set(model);
  const rc = x.hxh_load(mp, model.length);
  if (rc !== 0) throw new Error(`${modelPath}: hxh_load failed: ${rc}`);
  const W = x.hxh_image_width(), H = x.hxh_image_height(), T = x.hxh_timesteps(), C = x.hxh_channels();

  const g = readFileSync(goldenPath);
  const dv = new DataView(g.buffer, g.byteOffset, g.byteLength);
  const n = dv.getUint32(8, true);
  const [gh, gw, gt16, gc, maxIds] = [12, 14, 16, 18, 20].map((o) => dv.getUint16(o, true));
  // T = 0 in the u16 field: the u32 at offset 24 holds it; max_ids = 0 for raw-output (detector) goldens
  const gt = gt16 || dv.getUint32(24, true);
  if (gh !== H || gw !== W || gt !== T || gc !== C) throw new Error(`${goldenPath}: golden shape mismatch`);
  const rec = 2 * H * W + T * C + 2 + 2 * maxIds;

  let okLogits = 0, okIds = 0, diffValues = 0;
  const times = [];
  for (let i = 0; i < n; i++) {
    const base = 32 + i * rec;
    new Uint8Array(mem(), x.hxh_image_ptr(), H * W).set(g.subarray(base, base + H * W));
    const t0 = performance.now();
    const k = x.hxh_run();
    times.push(performance.now() - t0);
    if (k < 0) throw new Error(`hxh_run failed: ${k}`);
    const logits = new Int8Array(mem(), x.hxh_logits_ptr(), T * C);
    const want = new Int8Array(g.buffer, g.byteOffset + base + 2 * H * W, T * C);
    let d = 0;
    for (let j = 0; j < T * C; j++) d += logits[j] !== want[j];
    diffValues += d;
    if (d === 0) okLogits++;
    const ids = new Uint16Array(mem().slice(x.hxh_ids_ptr(), x.hxh_ids_ptr() + 2 * k));
    const o = base + 2 * H * W + T * C;
    const nw = dv.getUint16(o, true);
    if (k === nw && ids.every((v, j) => v === dv.getUint16(o + 2 + 2 * j, true))) okIds++;
  }
  times.sort((a, b) => a - b);
  const med = times[n >> 1];
  // steady-state timing on the first image: median and minimum of single runs
  let ms = NaN, best = NaN;
  if (benchMs > 0) {
    new Uint8Array(mem(), x.hxh_image_ptr(), H * W).set(g.subarray(32, 32 + H * W));
    for (let r = 0; r < 3; r++) x.hxh_run();
    const reps = Math.max(10, Math.ceil(benchMs / Math.max(med, 0.01)));
    const ts = [];
    for (let r = 0; r < reps; r++) {
      const t0 = performance.now();
      x.hxh_run();
      ts.push(performance.now() - t0);
    }
    ts.sort((a, b) => a - b);
    ms = ts[ts.length >> 1];
    best = ts[0];
  }
  const macs = x.hxh_macs();
  const name = dirname(resolve(modelPath)).split(/[\\/]/).pop();
  if (profile) printProfile(x, name, g.subarray(32, 32 + H * W), H * W);
  const ok = okLogits === n && okIds === n;
  if (!ok) failed = true;
  rows.push({ name, input: `${H}x${W}`, MMACs: (macs / 1e6).toFixed(1), vectors: n, "logits bit-exact": `${okLogits}/${n}`,
    "differing values": diffValues, "CTC ids identical": `${okIds}/${n}`, "first-pass median ms": med.toFixed(3),
    "median ms": ms.toFixed(3), "min ms": best.toFixed(3), "GMAC/s (median)": (macs / ms / 1e6).toFixed(2) });
  console.log(`${name}: logits bit-exact ${okLogits}/${n} (${diffValues} differing values), CTC ids identical ${okIds}/${n}, ` +
    (benchMs > 0 ? `${ms.toFixed(3)} ms median, ${best.toFixed(3)} ms min, ` : "") +
    `first pass over the golden set: median ${med.toFixed(3)} ms -> ${(macs / (benchMs > 0 ? ms : med) / 1e6).toFixed(2)} GMAC/s`);
}
if (rows.length > 1) console.table(rows);

function printProfile(x, name, image, size) {
  if (!x.hxh_profile) throw new Error("this build has no hxh_profile (cargo --features profile)");
  new Uint8Array(x.memory.buffer, x.hxh_image_ptr(), size).set(image);
  const reps = 30;
  const p = x.hxh_profile(reps);
  const nOps = x.hxh_op_count ? x.hxh_op_count() : 0;
  const kinds = ["?", "Conv", "DwConv", "Add", "Lut", "Requant", "AvgPool", "Resize", "Concat"];
  const f = new Float64Array(x.memory.buffer, p);
  const classes = {};
  let total = 0;
  console.log(`${name}: per-op profile (fastest of ${reps} runs per op, ${kernels})`);
  for (let i = 0; 9 * i + 8 < f.length && (nOps === 0 || i < nOps); i++) {
    const [kind, kh, kw, sh, sw, cin, pix, cout, ms] = f.subarray(9 * i, 9 * i + 9);
    if (kind < 1 || kind > 8 || !(ms >= 0)) break;
    const macs = kind === 1 ? pix * cout * kh * kw * cin : kind === 2 ? pix * cout * kh * kw : 0;
    const cls = kind === 1 ? (kh === 1 && kw === 1 ? "conv 1x1" : cin === 1 ? "conv stem (cin 1)" : `conv ${kh}x${kw}`)
      : kind === 2 ? "depthwise" : "other";
    const c = (classes[cls] ??= { ms: 0, macs: 0 });
    c.ms += ms; c.macs += macs; total += ms;
    console.log(`  op${String(i).padEnd(3)} ${kinds[kind].padEnd(7)} k${kh}x${kw} s${sh}x${sw} cin ${String(cin).padStart(3)} ` +
      `-> ${String(pix).padStart(5)} px x ${String(cout).padStart(3)}  ${(ms * 1000).toFixed(1).padStart(7)} us` +
      (macs ? `  ${(macs / ms / 1e6).toFixed(2).padStart(6)} GMAC/s` : ""));
  }
  for (const [cls, c] of Object.entries(classes)) {
    console.log(`  ${cls.padEnd(18)} ${c.ms.toFixed(3).padStart(8)} ms ${(100 * c.ms / total).toFixed(1).padStart(5)}%` +
      (c.macs ? `  ${(c.macs / 1e6).toFixed(1).padStart(6)} M MACs  ${(c.macs / c.ms / 1e6).toFixed(2)} GMAC/s` : ""));
  }
}
console.log(failed ? "FAILED" : "all bit-exact");
process.exit(failed ? 1 : 0);
