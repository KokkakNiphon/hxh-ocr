// Node check of the browser pipeline (web/ocr.js + the wasm engine) against the Python
// reference written by make_reference.py. No browser needed.
//
//   .venv\Scripts\python.exe web\test\make_reference.py          (once)
//   node web/test/verify.mjs [--wasm web/hxh_infer.wasm] [--model web/models/model.hxhm] [--robust]
//
// Checks: rgbaToGray and resizeArea bit for bit against OpenCV; for every val line, the
// JS-preprocessed strip (the model's input size, e.g. 48x768) against resize_pad_u8 (pixels) and both strips' greedy
// decodes against each other and the label; toRomaji against hxh_ocr.glyphs.to_romaji;
// ctcDetails against the wasm decoder. --robust adds input perturbations (inverted,
// loose/tight boxes, margins, low resolution) and compares reading the box as drawn with
// the app's default (readLine with auto-fit: best of drawn and fitted box).
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import {
  HxhEngine, ctcDetails, idsToText, prepareLine, readLine, resizeArea, rgbaToGray, toRomaji,
} from "../ocr.js";

const here = dirname(fileURLToPath(import.meta.url));
const argv = process.argv.slice(2);
const opt = (name, def) => {
  const i = argv.indexOf(name);
  return i >= 0 ? argv[i + 1] : def;
};
const wasmPath = opt("--wasm", join(here, "..", "hxh_infer.wasm"));
const modelPath = opt("--model", join(here, "..", "models", "model.hxhm"));
const refDir = opt("--ref", join(here, "out"));
const robust = argv.includes("--robust");

const meta = JSON.parse(readFileSync(join(refDir, "reference.json"), "utf8"));
const bin = readFileSync(join(refDir, "reference.bin"));
const blob = ([off, len]) => new Uint8Array(bin.buffer, bin.byteOffset + off, len);
const modelJson = JSON.parse(readFileSync(join(dirname(modelPath), "model.json"), "utf8"));
const vocab = modelJson.vocab;
const engine = await HxhEngine.create(readFileSync(wasmPath), readFileSync(modelPath));
const { width: W, height: H } = engine;
let failures = 0;
const check = (ok, msg) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`);
  if (!ok) failures++;
};

// ------------------------------------------------------------ grayscale + resize vs OpenCV
{
  let ok = 0;
  for (const c of meta.gray) {
    const rgb = blob(c.rgb), n = c.w * c.h, rgba = new Uint8Array(4 * n);
    for (let i = 0; i < n; i++) rgba.set([rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], 255], 4 * i);
    const g = rgbaToGray(rgba), want = blob(c.gray);
    if (g.every((v, i) => v === want[i])) ok++;
  }
  check(ok === meta.gray.length, `rgbaToGray == cv2 RGB2GRAY: ${ok}/${meta.gray.length} images bit-exact`);
}
{
  let ok = 0, maxd = 0, bad = [];
  for (const c of meta.resize) {
    const got = resizeArea(blob(c.src), c.sw, c.sh, c.dw, c.dh), want = blob(c.dst);
    let d = 0;
    for (let i = 0; i < want.length; i++) d = Math.max(d, Math.abs(got[i] - want[i]));
    if (d === 0 && got.length === want.length) ok++;
    else bad.push(`${c.sw}x${c.sh}->${c.dw}x${c.dh} (max diff ${d})`);
    maxd = Math.max(maxd, d);
  }
  check(ok === meta.resize.length,
    `resizeArea == cv2 INTER_AREA (OpenCV ${meta.cv2}): ${ok}/${meta.resize.length} cases bit-exact, max diff ${maxd}` +
    (bad.length ? `\n      e.g. ${bad.slice(0, 5).join(", ")}` : ""));
}

// ------------------------------------------------------------ val lines
const grayToRgba = (g) => {
  const out = new Uint8Array(4 * g.length);
  for (let i = 0; i < g.length; i++) { out[4 * i] = out[4 * i + 1] = out[4 * i + 2] = g[i]; out[4 * i + 3] = 255; }
  return out;
};
const edit = (a, b) => {
  a = [...a]; b = [...b];
  let d = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const nd = [i];
    for (let j = 1; j <= b.length; j++) nd[j] = Math.min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (a[i - 1] !== b[j - 1]));
    d = nd;
  }
  return d[b.length];
};
const read = (strip) => {
  const { ids, logits } = engine.run(strip);
  return { text: idsToText(ids, vocab), ids, logits };
};

const rows = [];
let tPre = 0, tInf = 0, ctcSame = 0;
for (const L of meta.lines) {
  const gray = blob(L.gray);
  const t0 = performance.now();
  const prep = prepareLine(rgbaToGray(grayToRgba(gray)), L.w, L.h, { height: H, width: W });
  const t1 = performance.now();
  const js = read(prep.pixels);
  tInf += performance.now() - t1;
  tPre += t1 - t0;
  const py = read(blob(L.ref));
  const ref = blob(L.ref);
  let ndiff = 0, maxd = 0;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(prep.pixels[i] - ref[i]);
    if (d) { ndiff++; maxd = Math.max(maxd, d); }
  }
  const det = ctcDetails(js.logits, engine.timesteps, engine.channels, engine.numClasses, engine.logitsExp);
  if (det.length === js.ids.length && det.every((c, i) => c.id === js.ids[i])) ctcSame++;
  rows.push({ L, prep, js: js.text, py: py.text, ndiff, maxd, romajiOk: toRomaji(L.label) === L.romaji });
}
const n = rows.length;
const fit = rows.filter((r) => !r.prep.squeezed);
const sq = rows.filter((r) => r.prep.squeezed);
const count = (rs, f) => rs.filter(f).length;
const cer = (rs, key) => {
  let e = 0, t = 0;
  for (const r of rs) { e += edit(r[key], r.L.label); t += [...r.L.label].length; }
  return ((100 * e) / Math.max(t, 1)).toFixed(2);
};

console.log(`\nfirst 20 lines (${meta.data}):`);
console.log("  #  style  src WxH   ->w   label                       JS decode                    =label =python  px diff");
rows.slice(0, 20).forEach((r, i) => {
  const pad = (s, k) => s + " ".repeat(Math.max(0, k - [...s].length * 2));
  console.log(`  ${String(i).padStart(2)} ${r.L.style.padEnd(6)} ${`${r.L.w}x${r.L.h}`.padEnd(8)} ${String(r.prep.naturalWidth).padStart(4)}${r.prep.squeezed ? "*" : " "} ` +
    `${pad(r.L.label, 28)}${pad(r.js, 28)} ${r.js === r.L.label ? "yes" : "no "}    ${r.js === r.py ? "yes" : "NO "}     ${r.ndiff}`);
});
console.log(`  (* = wider than ${W} px at height ${H}, squeezed)\n`);

check(count(rows, (r) => r.ndiff === 0) === n, `JS strip == Python resize_pad_u8 strip: ${count(rows, (r) => r.ndiff === 0)}/${n} bit-exact` +
  ` (max pixel diff ${Math.max(...rows.map((r) => r.maxd))})`);
check(count(rows, (r) => r.js === r.py) === n, `decode(JS strip) == decode(Python strip): ${count(rows, (r) => r.js === r.py)}/${n}`);
check(ctcSame === n, `ctcDetails ids == wasm decoder ids: ${ctcSame}/${n}`);
check(count(rows, (r) => r.romajiOk) === n && Object.entries(meta.romaji).every(([k, v]) => toRomaji(k) === v),
  `toRomaji == hxh_ocr.glyphs.to_romaji on ${n} labels + ${Object.keys(meta.romaji).length} vocab/edge strings`);
console.log(`\ndecode == label: ${count(rows, (r) => r.js === r.L.label)}/${n} lines, CER ${cer(rows, "js")}%`);
console.log(`  fits in ${W} px:  ${count(fit, (r) => r.js === r.L.label)}/${fit.length} exact, CER ${cer(fit, "js")}%`);
console.log(`  squeezed (>${W}):${count(sq, (r) => r.js === r.L.label)}/${sq.length} exact, CER ${cer(sq, "js")}%`);
console.log(`timing (node, this wasm): preprocess ${(tPre / n).toFixed(2)} ms, inference ${(tInf / n).toFixed(2)} ms per line` +
  ` (${(engine.macs / (tInf / n) / 1e6).toFixed(2)} GMAC/s)`);

// ------------------------------------------------------------ robustness (real-looking crops)
if (robust) {
  const padRows = (g, w, h, top, bottom) => {
    const out = new Uint8Array(w * (h + top + bottom));
    for (let y = 0; y < h + top + bottom; y++) {
      const sy = Math.min(h - 1, Math.max(0, y - top));
      out.set(g.subarray(sy * w, sy * w + w), y * w);
    }
    return { g: out, w, h: h + top + bottom };
  };
  const padCols = (g, w, h, left, right) => {
    const nw = w + left + right, out = new Uint8Array(nw * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < nw; x++) out[y * nw + x] = g[y * w + Math.min(w - 1, Math.max(0, x - left))];
    return { g: out, w: nw, h };
  };
  const downTo = (g, w, h, nh) => {
    const nw = Math.max(1, Math.round((w * nh) / h));
    return { g: resizeArea(g, w, h, nw, nh), w: nw, h: nh };
  };
  // each variant: an image plus the box a user might draw on it (default: the whole image)
  const inner = (g, w, h, top, bottom) => ({ g, w, h, sel: { x: 0, y: top, w, h: h - top - bottom } });
  const variants = {
    "as is": (g, w, h) => ({ g, w, h }),
    "inverted (Invert off)": (g, w, h) => ({ g: g.map((v) => 255 - v), w, h }),
    "inverted + Invert on": (g, w, h) => ({ g: g.map((v) => 255 - v), w, h, invert: true }),
    "loose: +25% height top & bottom": (g, w, h) => padRows(g, w, h, h >> 2, h >> 2),
    "loose: +50% height top & bottom": (g, w, h) => padRows(g, w, h, h >> 1, h >> 1),
    "tight: box cuts 10% top & bottom": (g, w, h) => inner(g, w, h, Math.round(h * 0.1), Math.round(h * 0.1)),
    "off-centre: +40% top only": (g, w, h) => padRows(g, w, h, Math.round(h * 0.4), 0),
    "+1 line height margin left & right": (g, w, h) => padCols(g, w, h, h, h),
    "low-res: source 24 px high": (g, w, h) => downTo(g, w, h, 24),
    "low-res: source 16 px high": (g, w, h) => downTo(g, w, h, 16),
    "hi-res: source x4 (256 px high)": (g, w, h) => downTo(g, w, h, 4 * h),
  };
  console.log(`\nrobustness on the ${fit.length} lines that fit (exact lines / CER; box as drawn vs auto-fit on):`);
  for (const [name, f] of Object.entries(variants)) {
    const res = [false, true].map((fitBox) => {
      let ok = 0, e = 0, t = 0, squeezed = 0, fitted = 0;
      for (const r of fit) {
        const v = f(blob(r.L.gray), r.L.w, r.L.h);
        const sel = v.sel || { x: 0, y: 0, w: v.w, h: v.h };
        const out = readLine(engine, v.g, v.w, v.h, sel, { fit: fitBox, invert: !!v.invert });
        const text = idsToText(out.ids, vocab);
        if (out.prep.squeezed) squeezed++;
        if (out.box.fitted) fitted++;
        if (text === r.L.label) ok++;
        e += edit(text, r.L.label); t += [...r.L.label].length;
      }
      return `${String(ok).padStart(4)}/${fit.length} ${((100 * e) / t).toFixed(2).padStart(6)}%` +
        (squeezed ? ` ${squeezed} sq` : "") + (fitBox ? ` (fitted box won ${fitted}x)` : "");
    });
    console.log(`  ${name.padEnd(36)} ${res[0].padEnd(22)} ${res[1]}`);
  }
  console.log(`  (sq = lines squeezed because they got wider than ${W} px at height ${H})`);

  // photos of a sign (make_reference.py render_page): boxes a person might draw by hand
  if (meta.pages?.length) {
    const kinds = {
      "tight (glyph bbox + 4 px)": (b, lh) => [b[0] - 4, b[1] - 4, b[2] + 4, b[3] + 4],
      "loose (+45% height, +40% sides)": (b, lh) => [b[0] - 0.4 * lh, b[1] - 0.45 * lh, b[2] + 0.4 * lh, b[3] + 0.45 * lh],
      "sloppy (+70% top, +15% bottom)": (b, lh) => [b[0] - 0.8 * lh, b[1] - 0.7 * lh, b[2] + 0.2 * lh, b[3] + 0.15 * lh],
    };
    const n = meta.pages.reduce((s, p) => s + p.texts.length, 0);
    console.log(`
sign photos (${meta.pages.length} pages, ${n} lines; exact lines, box as drawn vs auto-fit on):`);
    for (const [kind, f] of Object.entries(kinds)) {
      const ok = [0, 0];
      for (const p of meta.pages) {
        const g = blob(p.gray);
        p.texts.forEach((label, i) => {
          const b = f(p.boxes[i], p.boxes[i][3] - p.boxes[i][1]);
          const x0 = Math.max(0, Math.floor(b[0])), y0 = Math.max(0, Math.floor(b[1]));
          const sel = { x: x0, y: y0, w: Math.min(p.w, Math.ceil(b[2])) - x0, h: Math.min(p.h, Math.ceil(b[3])) - y0 };
          [false, true].forEach((fitBox, k) => {
            if (idsToText(readLine(engine, g, p.w, p.h, sel, { fit: fitBox }).ids, vocab) === label) ok[k]++;
          });
        });
      }
      console.log(`  ${kind.padEnd(36)} ${String(ok[0]).padStart(4)}/${n}               ${String(ok[1]).padStart(4)}/${n}`);
    }
  }
}

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
