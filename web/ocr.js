// HxH OCR core: preprocessing, the wasm engine wrapper, CTC details and romaji.
// Pure functions only (no DOM), so the same file runs in the browser and in node
// (web/test/verify.mjs). Keep exports as `export function|const|class` declarations:
// scripts/build_web.ps1 inlines this file into the single-file build by stripping
// the `export` keywords.

// ---------------------------------------------------------------- rounding helpers

/** Round half to even (Python's round(), OpenCV's cvRound / saturate_cast). */
export function roundHalfEven(x) {
  const r = Math.round(x); // half up
  return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

const f32 = Math.fround;
const clampU8 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

// ---------------------------------------------------------------- grayscale

/**
 * RGBA (canvas ImageData layout) -> u8 grayscale, OpenCV 4.x COLOR_RGB2GRAY bit for bit:
 * 0.299 R + 0.587 G + 0.114 B in 15-bit fixed point (RY15/GY15/BY15) with rounding.
 * Alpha is ignored; composite onto a background before calling (the app draws on white).
 */
export function rgbaToGray(rgba, n = rgba.length >> 2) {
  const out = new Uint8Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[i] = (rgba[j] * 9798 + rgba[j + 1] * 19235 + rgba[j + 2] * 3735 + 16384) >> 15;
  }
  return out;
}

// ---------------------------------------------------------------- INTER_AREA resize

// OpenCV computeResizeAreaTab (imgproc/src/resize.cpp): per destination pixel, the
// source pixels it covers and their (float32) weights.
function areaTab(ssize, dsize, scale) {
  const di = [], si = [], alpha = [];
  for (let dx = 0; dx < dsize; dx++) {
    const fsx1 = dx * scale;
    const fsx2 = fsx1 + scale;
    const cellWidth = Math.min(scale, ssize - fsx1);
    let sx1 = Math.ceil(fsx1);
    let sx2 = Math.floor(fsx2);
    sx2 = Math.min(sx2, ssize - 1);
    sx1 = Math.min(sx1, sx2);
    if (sx1 - fsx1 > 1e-3) {
      di.push(dx); si.push(sx1 - 1); alpha.push((sx1 - fsx1) / cellWidth);
    }
    for (let sx = sx1; sx < sx2; sx++) {
      di.push(dx); si.push(sx); alpha.push(1.0 / cellWidth);
    }
    if (fsx2 - sx2 > 1e-3) {
      di.push(dx); si.push(sx2); alpha.push(Math.min(Math.min(fsx2 - sx2, 1), cellWidth) / cellWidth);
    }
  }
  return { di: Int32Array.from(di), si: Int32Array.from(si), alpha: Float32Array.from(alpha) };
}

// True area averaging with fractional weights (ResizeArea_Invoker), float32 math in
// OpenCV's operation order.
function resizeAreaGeneral(src, sw, sh, dw, dh, scaleX, scaleY) {
  const xt = areaTab(sw, dw, scaleX);
  const yt = areaTab(sh, dh, scaleY);
  const dst = new Uint8Array(dw * dh);
  const buf = new Float32Array(dw);
  const sum = new Float32Array(dw);
  const flush = (dy) => {
    for (let dx = 0; dx < dw; dx++) dst[dy * dw + dx] = clampU8(roundHalfEven(sum[dx]));
  };
  let prevDy = yt.di[0];
  for (let j = 0; j < yt.di.length; j++) {
    const beta = yt.alpha[j];
    const dy = yt.di[j];
    const row = yt.si[j] * sw;
    buf.fill(0);
    for (let k = 0; k < xt.di.length; k++) {
      const d = xt.di[k];
      buf[d] = buf[d] + f32(src[row + xt.si[k]] * xt.alpha[k]);
    }
    if (dy !== prevDy) {
      flush(prevDy);
      for (let dx = 0; dx < dw; dx++) sum[dx] = f32(beta * buf[dx]);
      prevDy = dy;
    } else {
      for (let dx = 0; dx < dw; dx++) sum[dx] = sum[dx] + f32(beta * buf[dx]);
    }
  }
  flush(prevDy);
  return dst;
}

// Integer scale factors (resizeAreaFast_): plain box sums. 2x2 rounds half up
// ((sum + 2) >> 2, the SIMD path); other sizes use float(sum * (1/area)) and cvRound.
function resizeAreaFast(src, sw, dw, dh, ix, iy) {
  const dst = new Uint8Array(dw * dh);
  const area = ix * iy;
  const inv = f32(1 / area);
  for (let dy = 0; dy < dh; dy++) {
    const r0 = dy * iy * sw;
    for (let dx = 0; dx < dw; dx++) {
      const c0 = r0 + dx * ix;
      let s = 0;
      for (let y = 0; y < iy; y++) for (let x = 0; x < ix; x++) s += src[c0 + y * sw + x];
      dst[dy * dw + dx] = ix === 2 && iy === 2 ? (s + 2) >> 2 : clampU8(roundHalfEven(f32(s * inv)));
    }
  }
  return dst;
}

// INTER_AREA when either axis is upscaled: OpenCV's bilinear variant with "area"
// coefficients, 11-bit fixed point for 8-bit images (HResizeLinear + VResizeLinear).
function resizeAreaUpscale(src, sw, sh, dw, dh, invX, invY, scaleX, scaleY) {
  const COEF = 2048;
  const coeffs = (n, sn, scale, inv) => {
    const ofs = new Int32Array(n), a0 = new Int32Array(n), a1 = new Int32Array(n);
    let max = n;
    for (let d = 0; d < n; d++) {
      let s = Math.floor(d * scale);
      let f = f32((d + 1) - (s + 1) * inv);
      f = f <= 0 ? 0 : f32(f - Math.floor(f));
      if (s < 0) { f = 0; s = 0; }
      if (s + 1 >= sn) {
        max = Math.min(max, d);
        if (s >= sn - 1) { f = 0; s = sn - 1; }
      }
      ofs[d] = s;
      a0[d] = roundHalfEven(f32(f32(1 - f) * COEF));
      a1[d] = roundHalfEven(f32(f * COEF));
    }
    return { ofs, a0, a1, max };
  };
  const cx = coeffs(dw, sw, scaleX, invX);
  const cy = coeffs(dh, sh, scaleY, invY);
  const hrow = (sy) => {
    const out = new Int32Array(dw);
    const r = sy * sw;
    for (let dx = 0; dx < dw; dx++) {
      const sx = cx.ofs[dx];
      out[dx] = dx < cx.max ? src[r + sx] * cx.a0[dx] + src[r + sx + 1] * cx.a1[dx] : src[r + sx] * COEF;
    }
    return out;
  };
  const dst = new Uint8Array(dw * dh);
  for (let dy = 0; dy < dh; dy++) {
    const sy = cy.ofs[dy];
    const S0 = hrow(sy);
    const S1 = hrow(Math.min(sy + 1, sh - 1));
    const b0 = cy.a0[dy], b1 = cy.a1[dy];
    for (let dx = 0; dx < dw; dx++) {
      dst[dy * dw + dx] = clampU8((((b0 * (S0[dx] >> 4)) >> 16) + ((b1 * (S1[dx] >> 4)) >> 16) + 2) >> 2);
    }
  }
  return dst;
}

/**
 * u8 single-channel resize matching cv2.resize(src, (dw, dh), interpolation=cv2.INTER_AREA)
 * (OpenCV 4.x, no IPP for INTER_AREA). Downscaling averages every covered source pixel,
 * which is what training used; canvas drawImage would alias thin strokes away.
 */
export function resizeArea(src, sw, sh, dw, dh) {
  if (sw === dw && sh === dh) return src.slice();
  const invX = dw / sw, invY = dh / sh;
  const scaleX = 1 / invX, scaleY = 1 / invY;
  if (scaleX >= 1 && scaleY >= 1) {
    const ix = roundHalfEven(scaleX), iy = roundHalfEven(scaleY);
    if (Math.abs(scaleX - ix) < Number.EPSILON && Math.abs(scaleY - iy) < Number.EPSILON) {
      return resizeAreaFast(src, sw, dw, dh, ix, iy);
    }
    return resizeAreaGeneral(src, sw, sh, dw, dh, scaleX, scaleY);
  }
  return resizeAreaUpscale(src, sw, sh, dw, dh, invX, invY, scaleX, scaleY);
}

// ---------------------------------------------------------------- line preprocessing

/**
 * One cropped text line (u8 grayscale, w x h) -> the model's u8 input strip, exactly like
 * the calibration pipeline (experiments/esp_ppq_spike/spike_common.preprocess):
 * resize to `height` keeping aspect (INTER_AREA, Python round() for the width); if that is
 * wider than `width`, squeeze it to `width`; else pad on the right with the last column.
 * `invert` flips the strip (255 - v), for light text on a dark background.
 */
export function prepareLine(gray, w, h, { height = 32, width = 256, invert = false } = {}) {
  const natural = Math.max(1, roundHalfEven((w * height) / h));
  const cw = Math.min(natural, width);
  const small = resizeArea(gray, w, h, cw, height);
  const pixels = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const s = y * cw, d = y * width;
    pixels.set(small.subarray(s, s + cw), d);
    pixels.fill(small[s + cw - 1], d + cw, d + width);
  }
  if (invert) for (let i = 0; i < pixels.length; i++) pixels[i] = 255 - pixels[i];
  return { pixels, width, height, contentWidth: cw, naturalWidth: natural, squeezed: natural > width, upscaled: h < height };
}

// ---------------------------------------------------------------- auto-fit the box

const smooth = (a, k) =>
  a.map((_, i) => {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - k); j <= Math.min(a.length - 1, i + k); j++) { s += a[j]; n++; }
    return s / n;
  });

// (10th percentile, max) of a profile: the background floor and the text peak
function floorPeak(a) {
  const s = Float64Array.from(a).sort();
  return [s[Math.floor(0.1 * (s.length - 1))], s[s.length - 1]];
}

/**
 * Where is the text inside `sel` ({x, y, w, h}) of a W x H u8 image? Uses the energy of
 * horizontal gradients |g(x+1) - g(x)|: strokes make lots of it, while smooth backgrounds,
 * uneven lighting and horizontal clutter lines make little. Returns the main band of rows
 * (the run with most energy) and the columns with energy inside it, or null when nothing
 * stands out.
 */
export function findText(gray, W, H, sel) {
  const { x, y, w, h } = sel;
  if (w < 4 || h < 4) return null;
  const E = new Float32Array(w * h);
  const rows = new Float64Array(h);
  for (let r = 0; r < h; r++) {
    const o = (y + r) * W + x;
    let s = 0;
    for (let c = 0; c + 1 < w; c++) s += E[r * w + c] = Math.abs(gray[o + c + 1] - gray[o + c]);
    rows[r] = s / w;
  }
  const sm = smooth(rows, 1);
  const [floor, peak] = floorPeak(sm);
  if (peak - floor < 3) return null;
  const level = (f) => floor + f * (peak - floor);
  // runs of rows above 30% of the way to the peak, bridging small gaps; keep the strongest
  const gap = Math.max(2, Math.round(0.12 * h));
  let best = null;
  for (let r = 0; r < h;) {
    if (sm[r] < level(0.3)) { r++; continue; }
    let end = r;
    for (let q = r, g = 0; q < h && g <= gap; q++) {
      if (sm[q] >= level(0.3)) { end = q; g = 0; } else g++;
    }
    let mass = 0;
    for (let k = r; k <= end; k++) mass += sm[k] - floor;
    if (!best || mass > best.mass) best = { top: r, bottom: end, mass };
    r = end + 1;
  }
  let { top, bottom } = best;
  while (top > 0 && sm[top - 1] >= level(0.1)) top--; // faint edges: thin strokes, marks
  while (bottom < h - 1 && sm[bottom + 1] >= level(0.1)) bottom++;

  const cols = new Float64Array(w);
  for (let c = 0; c < w; c++) {
    let s = 0;
    for (let r = top; r <= bottom; r++) s += E[r * w + c];
    cols[c] = s / (bottom - top + 1);
  }
  const cs = smooth(cols, 2);
  const [cf, cp] = floorPeak(cs);
  let left = -1, right = -1;
  for (let c = 0; c < w; c++) {
    if (cs[c] >= cf + 0.15 * (cp - cf)) { if (left < 0) left = c; right = c; }
  }
  if (left < 0) return null;
  return { top: y + top, bottom: y + bottom, left: x + left, right: x + right };
}

/**
 * Turn a rough user box into a crop framed like the training lines: the text band takes
 * `ratio` of the crop height (centred; the crop may grow past the box, within the image)
 * and the sides keep `margin` x crop height of space (never wider than the box).
 * Returns the box unchanged (fitted: false) when no clear text is found.
 */
export function autoFit(gray, W, H, sel, { ratio = 0.62, margin = 0.3 } = {}) {
  let t;
  const f = sel.h / 160;
  if (f > 1.5) {
    // big box (e.g. a whole photo): analyse an area-downscaled copy, map the result back
    const sw = Math.max(4, Math.round(sel.w / f)), sh = Math.max(4, Math.round(sel.h / f));
    const s = findText(resizeArea(cropGray(gray, W, sel), sel.w, sel.h, sw, sh), sw, sh, { x: 0, y: 0, w: sw, h: sh });
    const fx = sel.w / sw, fy = sel.h / sh;
    t = s && {
      top: sel.y + Math.floor(s.top * fy), bottom: sel.y + Math.ceil((s.bottom + 1) * fy) - 1,
      left: sel.x + Math.floor(s.left * fx), right: sel.x + Math.ceil((s.right + 1) * fx) - 1,
    };
  } else {
    t = findText(gray, W, H, sel);
  }
  if (!t) return { ...sel, fitted: false };
  const th = t.bottom - t.top + 1;
  const ch = Math.min(H, Math.max(th + 2, Math.round(th / ratio)));
  let y0 = Math.round((t.top + t.bottom + 1) / 2 - ch / 2);
  y0 = Math.max(0, Math.min(H - ch, y0));
  const m = Math.round(margin * ch);
  const x0 = Math.max(sel.x, t.left - m), x1 = Math.min(sel.x + sel.w, t.right + 1 + m);
  return { x: x0, y: y0, w: x1 - x0, h: ch, fitted: true };
}

/** Copy a sub-rectangle out of a W-wide u8 image. */
export function cropGray(gray, W, { x, y, w, h }) {
  const out = new Uint8Array(w * h);
  for (let r = 0; r < h; r++) out.set(gray.subarray((y + r) * W + x, (y + r) * W + x + w), r * w);
  return out;
}

// ---------------------------------------------------------------- wasm engine

const LOAD_ERRORS = {
  [-1]: "out of memory",
  [-10]: "model file truncated",
  [-11]: "not a .hxhm file (bad magic)",
  [-12]: "unsupported .hxhm version",
  [-13]: "bad .hxhm header",
  [-14]: "bad tensor record",
  [-15]: "bad op record",
  [-16]: "arena too small",
  [-17]: "input size mismatch",
  [-18]: "model has no input lookup table",
  [-20]: "model needs a 1-channel input with an input LUT",
  [-30]: "no page given to the detector",
  [-32]: "detector frame not set",
  [-33]: "out of memory",
  [-34]: "bad size argument",
  [-35]: "the loaded model is not a detector",
};

/** A tiny module using wasm SIMD (i8x16.splat + popcnt), as in wasm-feature-detect. */
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0,
  253, 15, 253, 98, 11,
]);

export function wasmSimdSupported() {
  try {
    return WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
}

/**
 * A fresh instance of the wasm module with one model loaded (the module's bump allocator
 * holds one model per instance). `wasm` is the module bytes or a compiled WebAssembly.Module;
 * compile once and pass the Module when loading several models.
 */
async function instantiateWithModel(wasm, modelBytes) {
  const mod = wasm instanceof WebAssembly.Module ? wasm : await WebAssembly.compile(wasm);
  const x = (await WebAssembly.instantiate(mod, {})).exports;
  const p = x.hxh_alloc(modelBytes.length);
  if (!p) throw new Error("wasm: cannot allocate memory for the model");
  new Uint8Array(x.memory.buffer, p, modelBytes.length).set(modelBytes);
  const rc = x.hxh_load(p, modelBytes.length);
  if (rc !== 0) throw new Error(`model load failed: ${LOAD_ERRORS[rc] || "error " + rc}`);
  return x;
}

/** Thin wrapper over the raw exports of rust/hxh-infer-wasm for a recogniser (CTC) model. */
export class HxhEngine {
  static async create(wasm, modelBytes) {
    const x = await instantiateWithModel(wasm, modelBytes);
    if (x.hxh_raw_output && x.hxh_raw_output() === 1) throw new Error("this .hxhm is a detector, not a line recogniser");
    return new HxhEngine(x);
  }

  constructor(x) {
    this.x = x;
    this.width = x.hxh_image_width();
    this.height = x.hxh_image_height();
    this.timesteps = x.hxh_timesteps();
    this.channels = x.hxh_channels();
    this.numClasses = x.hxh_num_classes();
    this.logitsExp = x.hxh_logits_exp();
    this.macs = x.hxh_macs();
  }

  /** u8 strip (height x width) -> { ids (blank removed, from the wasm decoder), logits [T][C] int8 } */
  run(pixels) {
    const { x } = this;
    if (pixels.length !== this.width * this.height) throw new Error("input size mismatch");
    new Uint8Array(x.memory.buffer, x.hxh_image_ptr(), pixels.length).set(pixels);
    const n = x.hxh_run();
    if (n < 0) throw new Error(`hxh_run failed: ${n}`);
    const ids = new Uint16Array(x.memory.buffer, x.hxh_ids_ptr(), n).slice();
    const logits = new Int8Array(x.memory.buffer, x.hxh_logits_ptr(), this.timesteps * this.channels).slice();
    return { ids, logits };
  }
}

// ---------------------------------------------------------------- detector + page reading

/** Post-processing parameters of rust/hxh-detect (rust/DETECT.md), by index. */
export const DET_PARAMS = { bin_thresh: 0, min_score: 1, min_side: 2, unclip_ratio: 3, shrink_ratio: 4, block_gap: 5 };

const detCheck = (rc) => {
  if (rc < 0) throw new Error(`detector: ${LOAD_ERRORS[rc] || "error " + rc}`);
  return rc;
};

/**
 * The line detector (a raw-output .hxhm) plus the Rust post-processing (rust/hxh-detect):
 * page -> letterbox -> int8 maps -> line quads in reading order -> rectified crops.
 * Views of wasm memory are made after every call that may allocate (they can detach).
 */
export class HxhDetector {
  static async create(wasm, modelBytes) {
    const x = await instantiateWithModel(wasm, modelBytes);
    if (!x.hxh_raw_output || x.hxh_raw_output() !== 1) throw new Error("this .hxhm is not a detector (needs format v2, raw output)");
    return new HxhDetector(x);
  }

  constructor(x) {
    this.x = x;
    this.size = x.hxh_image_width(); // square letterbox input, e.g. 640
    this.outH = x.hxh_out_height();
    this.outW = x.hxh_out_width();
    this.macs = x.hxh_macs();
  }

  /**
   * Detect the text lines of a W x H u8 page. Returns { lines, tLetterbox, tRun, tPost }, lines
   * in reading order: { quad: [x0,y0,..,x3,y3] (page px, clockwise from the top-left in reading
   * orientation), score, vertical, block, length, thickness }. The page stays loaded for cropQuad.
   */
  detect(gray, W, H, params = {}) {
    const x = this.x;
    const now = () => performance.now();
    const t0 = now();
    const pp = x.hxh_det_image(W, H);
    if (!pp) throw new Error("detector: cannot allocate the page buffer");
    new Uint8Array(x.memory.buffer, pp, W * H).set(gray);
    detCheck(x.hxh_det_letterbox());
    const t1 = now();
    detCheck(x.hxh_run());
    const t2 = now();
    const p = new Float64Array(x.memory.buffer, x.hxh_det_params_ptr(), 8);
    for (const [k, v] of Object.entries(params)) if (k in DET_PARAMS) p[DET_PARAMS[k]] = v;
    const n = detCheck(x.hxh_det_detect());
    const recs = new Float32Array(x.memory.buffer, x.hxh_det_lines_ptr(), 16 * n).slice();
    const lines = [];
    for (let i = 0; i < n; i++) {
      const r = recs.subarray(16 * i, 16 * i + 16);
      lines.push({
        quad: Array.from(r.subarray(0, 8)), score: r[8], vertical: r[9] === 1, order: r[10], block: r[11],
        length: r[12], thickness: r[13],
      });
    }
    return { lines, tLetterbox: t1 - t0, tRun: t2 - t1, tPost: now() - t2 };
  }

  /** Upright u8 crop of a quad of the page last given to detect() (vertical: turned 90 deg CCW). */
  cropQuad(quad) {
    const x = this.x;
    new Float32Array(x.memory.buffer, x.hxh_det_quad_ptr(), 8).set(quad);
    const ptr = x.hxh_det_crop_quad(0, 0);
    if (!ptr) throw new Error("detector: crop failed");
    const w = x.hxh_det_crop_width(), h = x.hxh_det_crop_height();
    return { pixels: new Uint8Array(x.memory.buffer, ptr, w * h).slice(), w, h };
  }
}

/**
 * Grow a reading-oriented quad (q0->q1 along the text, q0->q3 across it) by `across` x its
 * thickness on both sides across the text and `along` x its thickness at both ends, so the
 * crop frames the glyphs with margins like the training lines.
 */
export function expandQuad(q, { across = 0, along = 0 } = {}) {
  const [x0, y0, x1, y1, x2, y2, x3, y3] = q;
  const len = Math.hypot(x1 - x0, y1 - y0) || 1, th = Math.hypot(x3 - x0, y3 - y0) || 1;
  const ux = (x1 - x0) / len, uy = (y1 - y0) / len, vx = (x3 - x0) / th, vy = (y3 - y0) / th;
  const a = along * th, c = across * th;
  return [
    x0 - ux * a - vx * c, y0 - uy * a - vy * c,
    x1 + ux * a - vx * c, y1 + uy * a - vy * c,
    x2 + ux * a + vx * c, y2 + uy * a + vy * c,
    x3 - ux * a + vx * c, y3 - uy * a + vy * c,
  ];
}

/** Is (px, py) inside the convex quad q (any winding)? */
export function pointInQuad(q, px, py) {
  let sign = 0;
  for (let i = 0; i < 4; i++) {
    const ax = q[2 * i], ay = q[2 * i + 1], bx = q[(2 * i + 2) % 8], by = q[(2 * i + 3) % 8];
    const c = (bx - ax) * (py - ay) - (by - ay) * (px - ax);
    if (c !== 0) {
      if (sign === 0) sign = Math.sign(c);
      else if (Math.sign(c) !== sign) return false;
    }
  }
  return true;
}

/** Distance from (px, py) to the quad's centre, in units of its thickness. */
function quadCentreDistance(l, px, py) {
  const q = l.quad;
  const cx = (q[0] + q[2] + q[4] + q[6]) / 4, cy = (q[1] + q[3] + q[5] + q[7]) / 4;
  return Math.hypot(px - cx, py - cy) / Math.max(1, l.thickness);
}

/** Recognise one upright crop ({ pixels, w, h }); vertical crops come already turned CCW. */
export function readCrop(engine, crop, { invert = false } = {}) {
  const prep = prepareLine(crop.pixels, crop.w, crop.h, { height: engine.height, width: engine.width, invert });
  const out = engine.run(prep.pixels);
  const logScore = pathScore(out.logits, engine.timesteps, engine.channels, engine.numClasses, engine.logitsExp);
  // (score stays the detector's mean kernel probability)
  return { ...out, prep, logScore, conf: readingConfidence(engine, out.logits) };
}

/**
 * Mean softmax probability of the characters a reading emits (0 when it emits none). Hunter
 * script reads at 0.95-1.0; other writing the detector picks up (Latin, Japanese, digits)
 * comes out as low-confidence kana, which is how the app tells the two apart.
 */
export function readingConfidence(engine, logits) {
  const d = ctcDetails(logits, engine.timesteps, engine.channels, engine.numClasses, engine.logitsExp);
  return d.length ? d.reduce((s, c) => s + c.prob, 0) / d.length : 0;
}

function recogniseLines(det, rec, lines, opts, offset = [0, 0]) {
  const { pad = {}, invert = false } = opts;
  const [ox, oy] = offset;
  const shift = (q) => q.map((v, i) => v + (i % 2 ? oy : ox));
  return lines.map((l) => {
    const crop = det.cropQuad(expandQuad(l.quad, pad));
    const engine = l.vertical && rec.v ? rec.v : rec.h;
    const r = readCrop(engine, crop, { invert });
    return { ...l, quad: shift(l.quad), engine, crop, ...r };
  });
}

/**
 * Read a whole page: detect, then recognise every line (vertical lines with rec.v when given).
 * rec = { h: HxhEngine, v?: HxhEngine }; opts = { params (DET_PARAMS names), pad (expandQuad),
 * invert, tiles: also detect on overlapping zoomed-in tiles of a big photo (detectTiled) }.
 * Returns { lines (reading order), tLetterbox, tDetect, tPost, tRead } (ms).
 */
export function readPage(det, rec, gray, W, H, opts = {}) {
  const d = opts.tiles ? detectTiled(det, gray, W, H, opts.params) : det.detect(gray, W, H, opts.params);
  const t0 = performance.now();
  let lines = recogniseLines(det, rec, d.lines, opts);
  if (opts.tiles && d.tiles) lines = refineRegion(det, rec, gray, W, H, lines, opts, det.size / Math.round(Math.min(2 * det.size, Math.max(det.size, 0.58 * Math.max(W, H)))));
  return { lines, tLetterbox: d.tLetterbox, tDetect: d.tRun, tPost: d.tPost, tRead: performance.now() - t0, tiles: d.tiles || 0 };
}

/**
 * "Tap a line": detect inside square windows around (px, py), the detector's own input size
 * first, so small text in a big photo is seen at a higher resolution than in the full-page
 * pass; then twice and half that. Returns the recognised line under the tap (or the nearest
 * within 1.5 line thicknesses) with its quad in page pixels and the window used, or null.
 */
export function readAt(det, rec, gray, W, H, px, py, opts = {}) {
  const L = Math.max(W, H);
  const sides = [...new Set([det.size, 2 * det.size, Math.round(det.size / 2)].map((s) => Math.min(s, L)))];
  for (const s of sides) {
    const w = Math.min(s, W), h = Math.min(s, H);
    const x0 = Math.max(0, Math.min(W - w, Math.round(px - w / 2)));
    const y0 = Math.max(0, Math.min(H - h, Math.round(py - h / 2)));
    const d = det.detect(cropGray(gray, W, { x: x0, y: y0, w, h }), w, h, opts.params);
    const lx = px - x0, ly = py - y0;
    let best = d.lines.find((l) => pointInQuad(l.quad, lx, ly));
    if (!best) {
      const near = d.lines.map((l) => [quadCentreDistance(l, lx, ly), l]).sort((a, b) => a[0] - b[0])[0];
      if (near && near[0] < 1.5) best = near[1];
    }
    if (best) return { ...recogniseLines(det, rec, [best], opts, [x0, y0])[0], window: { x: x0, y: y0, w, h } };
  }
  return null;
}

// ---------------------------------------------------------------- live camera helpers

function polyOf(q) {
  const P = [0, 1, 2, 3].map((i) => [q[2 * i], q[2 * i + 1]]);
  let s = 0;
  for (let i = 0; i < 4; i++) { const [x, y] = P[i], [u, v] = P[(i + 1) % 4]; s += x * v - u * y; }
  return s < 0 ? P.reverse() : P; // one winding for the clipper
}
function polyArea(P) {
  let s = 0;
  for (let i = 0; i < P.length; i++) { const [x, y] = P[i], [u, v] = P[(i + 1) % P.length]; s += x * v - u * y; }
  return Math.abs(s) / 2;
}

/** Intersection over union of two convex quads [x0,y0,..,x3,y3] (Sutherland-Hodgman). */
export function quadIoU(qa, qb) {
  const A = polyOf(qa), B = polyOf(qb);
  let out = A;
  for (let i = 0; i < 4 && out.length; i++) {
    const [ax, ay] = B[i], [bx, by] = B[(i + 1) % 4];
    const side = ([x, y]) => (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    const inp = out;
    out = [];
    for (let j = 0; j < inp.length; j++) {
      const P = inp[j], Q = inp[(j + 1) % inp.length], sp = side(P), sq = side(Q);
      if (sp >= 0) out.push(P);
      if ((sp >= 0) !== (sq >= 0)) { const t = sp / (sp - sq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); }
    }
  }
  const inter = out.length > 2 ? polyArea(out) : 0;
  return inter / (polyArea(A) + polyArea(B) - inter || 1);
}

/** Recognise lines the detector just found in the page it holds (see readPage). */
export function readLines(det, rec, lines, opts = {}) {
  return recogniseLines(det, rec, lines, opts);
}

/**
 * Keeps live-camera readings steady: each detected line is matched to a track from the
 * previous frames by quad IoU, and a track shows the most frequent of its last `keep`
 * readings, so one misread frame doesn't make the label flicker. Tracks unseen for more than
 * `maxMissed` frames are dropped. `update(lines, textOf)` returns the tracks:
 * { id, quad, vertical, text, hits, missed, line }.
 */
export class LiveTracker {
  constructor({ keep = 5, maxMissed = 2, minIoU = 0.3 } = {}) {
    Object.assign(this, { keep, maxMissed, minIoU, tracks: [], nextId: 1 });
  }

  update(lines, textOf) {
    for (const t of this.tracks) t.matched = false;
    const pairs = [];
    lines.forEach((l, li) => this.tracks.forEach((t, ti) => {
      const v = quadIoU(l.quad, t.quad);
      if (v >= this.minIoU) pairs.push([v, li, ti]);
    }));
    pairs.sort((a, b) => b[0] - a[0]);
    const taken = new Set();
    for (const [, li, ti] of pairs) {
      const t = this.tracks[ti];
      if (t.matched || taken.has(li)) continue;
      taken.add(li);
      this.feed(t, lines[li], textOf(lines[li]));
    }
    lines.forEach((l, li) => {
      if (taken.has(li)) return;
      const t = { id: this.nextId++, texts: [], hits: 0 };
      this.feed(t, l, textOf(l));
      this.tracks.push(t);
    });
    for (const t of this.tracks) if (!t.matched) t.missed++;
    this.tracks = this.tracks.filter((t) => t.missed <= this.maxMissed);
    return this.tracks;
  }

  feed(t, l, text) {
    Object.assign(t, { quad: l.quad, vertical: l.vertical, line: l, matched: true, missed: 0 });
    t.hits++;
    t.texts.push(text);
    if (t.texts.length > this.keep) t.texts.shift();
    const n = new Map();
    for (const s of t.texts) n.set(s, (n.get(s) || 0) + 1);
    let best = text, bestN = 0;
    for (const s of t.texts) if (n.get(s) >= bestN) { best = s; bestN = n.get(s); } // ties: the latest
    t.text = best;
  }

  reset() {
    this.tracks = [];
  }
}

// ---------------------------------------------------------------- CTC details

/**
 * Greedy CTC over int8 logits [T][C] (first numClasses channels, lowest id wins ties, like
 * the engine), with a softmax confidence per emitted character: the peak probability of its
 * class over its frames, and the best other non-blank class at that frame.
 */
export function ctcDetails(logits, T, C, numClasses, exp, blank = 0) {
  const scale = 2 ** exp;
  const chars = [];
  let prev = -1;
  const probs = new Float64Array(numClasses);
  for (let t = 0; t < T; t++) {
    const row = t * C;
    let best = 0;
    for (let c = 1; c < numClasses; c++) if (logits[row + c] > logits[row + best]) best = c;
    let z = 0;
    for (let c = 0; c < numClasses; c++) z += probs[c] = Math.exp((logits[row + c] - logits[row + best]) * scale);
    if (best !== blank) {
      const p = probs[best] / z;
      let alt = -1;
      for (let c = 0; c < numClasses; c++) {
        if (c !== best && c !== blank && (alt < 0 || probs[c] > probs[alt])) alt = c;
      }
      const cur = { id: best, prob: p, alt, altProb: alt >= 0 ? probs[alt] / z : 0, t0: t, t1: t };
      if (best !== prev) chars.push(cur);
      else {
        const last = chars[chars.length - 1];
        last.t1 = t;
        if (p > last.prob) Object.assign(last, { prob: p, alt: cur.alt, altProb: cur.altProb });
      }
    }
    prev = best;
  }
  return chars;
}

/**
 * Log-likelihood of the greedy path (sum over timesteps of log max softmax). Used to pick
 * between the box as drawn and the auto-fitted box: same T, so scores are comparable.
 */
export function pathScore(logits, T, C, numClasses, exp) {
  const scale = 2 ** exp;
  let s = 0;
  for (let t = 0; t < T; t++) {
    const row = t * C;
    let m = -Infinity;
    for (let c = 0; c < numClasses; c++) m = Math.max(m, logits[row + c]);
    let z = 0;
    for (let c = 0; c < numClasses; c++) z += Math.exp((logits[row + c] - m) * scale);
    s -= Math.log(z);
  }
  return s;
}

/**
 * Read one line: crop `sel` from a W x H u8 image, preprocess, run. With `fit`, the
 * auto-fitted box (autoFit) is read too and the candidate with the more likely greedy path
 * wins; this recovers most of the accuracy lost to loose or tight hand-drawn boxes while
 * leaving well-cropped lines alone (web/test/verify.mjs --robust).
 */
export function readLine(engine, gray, W, H, sel, { fit = true, invert = false } = {}) {
  const now = () => performance.now();
  let tPre = 0, tInf = 0, t0 = now();
  const boxes = [{ ...sel, fitted: false }];
  if (fit) {
    const f = autoFit(gray, W, H, sel);
    if (f.fitted && (f.x !== sel.x || f.y !== sel.y || f.w !== sel.w || f.h !== sel.h)) boxes.push(f);
  }
  tPre += now() - t0;
  let best = null;
  for (const box of boxes) {
    t0 = now();
    const prep = prepareLine(cropGray(gray, W, box), box.w, box.h, { height: engine.height, width: engine.width, invert });
    const t1 = now();
    const out = engine.run(prep.pixels);
    const score = pathScore(out.logits, engine.timesteps, engine.channels, engine.numClasses, engine.logitsExp);
    tInf += now() - t1;
    tPre += t1 - t0;
    if (!best || score > best.score) best = { box, prep, ...out, score };
  }
  return { ...best, candidates: boxes.length, tPre, tInf };
}

/** Class ids (blank removed) -> text, class i = vocab token i - 1. */
export function idsToText(ids, vocab) {
  let s = "";
  for (const i of ids) s += i >= 1 && i <= vocab.length ? vocab[i - 1] : "?";
  return s;
}

// ---------------------------------------------------------------- romaji

// Port of hxh_ocr/glyphs.py to_romaji: rough Hepburn.
const ROMAJI = (() => {
  const t = {};
  const rows = {
    "": "あいうえお", k: "かきくけこ", s: "さしすせそ", t: "たちつてと", n: "なにぬねの",
    h: "はひふへほ", m: "まみむめも", r: "らりるれろ", g: "がぎぐげご", z: "ざじずぜぞ",
    d: "だぢづでど", b: "ばびぶべぼ", p: "ぱぴぷぺぽ",
  };
  for (const [cons, kana] of Object.entries(rows)) {
    [...kana].forEach((ch, i) => (t[ch] = cons + "aiueo"[i]));
  }
  Object.assign(t, {
    し: "shi", ち: "chi", つ: "tsu", ふ: "fu", じ: "ji", ぢ: "ji", づ: "zu",
    や: "ya", ゆ: "yu", よ: "yo", わ: "wa", を: "wo", ん: "n", ゔ: "vu",
  });
  return t;
})();

const MACRON = { a: "ā", i: "ī", u: "ū", e: "ē", o: "ō" };

/** Rough Hepburn; ー puts a macron on the vowel before it (はんたー -> hantā). */
export function toRomaji(text) {
  const out = [];
  for (const ch of text) {
    if (ch === "ー") {
      const last = out.length ? out[out.length - 1] : "";
      const v = last ? last[last.length - 1] : "";
      if (MACRON[v]) out[out.length - 1] = last.slice(0, -1) + MACRON[v];
      else if (!"āīūēō".includes(v) || !v) out.push("-"); // a second ー adds nothing
    } else out.push(ROMAJI[ch] ?? ch);
  }
  return out.join("");
}

// ---------------------------------------------------------------- misc

/** base64 -> Uint8Array (single-file build). */
export function base64ToBytes(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

// ---------------------------------------------------------------- big photos: tiled detection

/**
 * Overlapping square tiles for finding small text in a big photo. The full-page pass shrinks
 * the long side to the detector's input (640), so in a 2000 px phone photo a line of text that
 * fills a fifth of the frame ends up a few pixels tall. Tiles of about 0.58 of the long side
 * (between 1 and 2 detector inputs) are seen at roughly twice that resolution; neighbours
 * overlap by a quarter so a line cut by one tile is whole in the next. None for small pages.
 */
export function pageTiles(W, H, size = 640) {
  const L = Math.max(W, H);
  if (L <= 1.5 * size) return [];
  const T = Math.round(Math.min(2 * size, Math.max(size, 0.58 * L)));
  const starts = (n) => {
    if (n <= T) return [0];
    const k = Math.ceil((n - T) / (0.75 * T)) + 1;
    return Array.from({ length: k }, (_, i) => Math.round((i * (n - T)) / (k - 1)));
  };
  const tiles = [];
  for (const y of starts(H)) for (const x of starts(W)) tiles.push({ x, y, w: Math.min(T, W), h: Math.min(T, H) });
  return tiles;
}

function quadInter(qa, qb) {
  const A = polyOf(qa), B = polyOf(qb);
  let out = A;
  for (let i = 0; i < 4 && out.length; i++) {
    const [ax, ay] = B[i], [bx, by] = B[(i + 1) % 4];
    const side = ([x, y]) => (bx - ax) * (y - ay) - (by - ay) * (x - ax);
    const inp = out;
    out = [];
    for (let j = 0; j < inp.length; j++) {
      const P = inp[j], Q = inp[(j + 1) % inp.length], sp = side(P), sq = side(Q);
      if (sp >= 0) out.push(P);
      if ((sp >= 0) !== (sq >= 0)) { const t = sp / (sp - sq); out.push([P[0] + t * (Q[0] - P[0]), P[1] + t * (Q[1] - P[1])]); }
    }
  }
  return { inter: out.length > 2 ? polyArea(out) : 0, a: polyArea(A), b: polyArea(B) };
}

/**
 * Merge line detections from the full-page pass and the tiles (all in page pixels).
 * - A tile line that touches an inner tile edge may be cut off there; it is only kept if nothing
 *   better covers it.
 * - Tile lines are seen at a higher resolution, so they win over full-pass lines they overlap,
 *   except when they cover too little of a full-pass line of the same thickness: then that line
 *   is the more complete one and replaces them (a line the tiles only found in pieces).
 * - Overlap is intersection over the smaller area, so a half line inside a whole line counts.
 */
function mergeDetections(full, tiled) {
  const ov = (p, q) => { const o = quadInter(p.quad, q.quad); return o.inter / (Math.min(o.a, o.b) || 1); };
  const byLen = (a, b) => b.length - a.length;
  const whole = tiled.filter((l) => !l.cut).sort(byLen), cut = tiled.filter((l) => l.cut).sort(byLen);
  const kept = [];
  for (const l of whole) if (!kept.some((k) => ov(l, k) > 0.5)) kept.push(l);
  for (const F of [...full].sort(byLen)) {
    const over = kept.filter((k) => ov(F, k) > 0.5);
    if (!over.length) { kept.push(F); continue; }
    const covered = over.reduce((s, k) => s + quadInter(F.quad, k.quad).inter, 0) / (quadInter(F.quad, F.quad).a || 1);
    const sameSize = over.every((k) => Math.max(k.thickness, F.thickness) / Math.max(1e-6, Math.min(k.thickness, F.thickness)) < 1.6);
    if (covered < 0.6 && sameSize) {
      for (const k of over) kept.splice(kept.indexOf(k), 1);
      kept.push(F);
    }
  }
  for (const l of cut) if (!kept.some((k) => ov(l, k) > 0.3)) kept.push(l);
  return kept;
}

/** Reading order for merged lines: rows top to bottom; across a row, horizontal lines left to
 *  right and vertical columns right to left (manga order). */
function orderLines(lines) {
  const box = (l) => {
    const xs = [0, 2, 4, 6].map((i) => l.quad[i]), ys = [1, 3, 5, 7].map((i) => l.quad[i]);
    return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
  };
  const items = lines.map((l) => ({ l, b: box(l) })).sort((p, q) => p.b.y0 - q.b.y0);
  const rows = [];
  for (const it of items) {
    const row = rows.find((r) => it.b.y0 < r.y1 - 0.5 * Math.min(r.h, it.b.y1 - it.b.y0));
    if (row && !it.l.vertical && row.horizontal) { row.items.push(it); row.y1 = Math.max(row.y1, it.b.y1); }
    else rows.push({ items: [it], y1: it.b.y1, h: it.b.y1 - it.b.y0, horizontal: !it.l.vertical });
  }
  const out = [];
  for (const r of rows) out.push(...r.items.sort((p, q) => (r.horizontal ? p.b.x0 - q.b.x0 : q.b.x1 - p.b.x1)).map((it) => it.l));
  return out.map((l, i) => ({ ...l, order: i }));
}

/**
 * Find the lines of a big photo: the full-page pass plus pageTiles(), merged. The full page is
 * detected last, so it is the page the detector holds for cropping afterwards. Returns the same
 * shape as HxhDetector.detect(), plus the number of tiles.
 */
export function detectTiled(det, gray, W, H, params = {}) {
  const tiles = pageTiles(W, H, det.size);
  if (!tiles.length) return { ...det.detect(gray, W, H, params), tiles: 0 };
  const t0 = performance.now();
  const tiled = [];
  for (const t of tiles) {
    const d = det.detect(cropGray(gray, W, t), t.w, t.h, params);
    for (const l of d.lines) {
      const quad = l.quad.map((v, i) => v + (i % 2 ? t.y : t.x));
      const m = Math.max(2, 0.6 * l.thickness);
      const xs = [0, 2, 4, 6].map((i) => quad[i]), ys = [1, 3, 5, 7].map((i) => quad[i]);
      const cut = (t.x > 0 && Math.min(...xs) < t.x + m) || (t.x + t.w < W && Math.max(...xs) > t.x + t.w - m)
        || (t.y > 0 && Math.min(...ys) < t.y + m) || (t.y + t.h < H && Math.max(...ys) > t.y + t.h - m);
      tiled.push({ ...l, quad, cut });
    }
  }
  const tTiles = performance.now() - t0;
  const d = det.detect(gray, W, H, params);
  const t1 = performance.now();
  const lines = orderLines(mergeDetections(d.lines, tiled));
  return { lines, tLetterbox: d.tLetterbox, tRun: d.tRun + tTiles, tPost: d.tPost + performance.now() - t1, tiles: tiles.length };
}

/**
 * Second look at a big photo: if the lines that read confidently as Hunter script sit in a
 * small part of the frame (a screen, a card, a sign), detect again inside that part at a higher
 * resolution and merge, preferring the closer look (same rules as the tiles). Lines outside
 * the region are kept as they are. Returns the merged lines, read, in reading order.
 */
function refineRegion(det, rec, gray, W, H, lines, opts, tileScale) {
  const minConf = opts.minConf ?? 0.8;
  const good = lines.filter((l) => l.conf >= minConf && l.ids.length >= 2 && l.score >= 0.55);
  if (!good.length) return lines;
  const xs = good.flatMap((l) => [0, 2, 4, 6].map((i) => l.quad[i])), ys = good.flatMap((l) => [1, 3, 5, 7].map((i) => l.quad[i]));
  const th = good.map((l) => l.thickness).sort((a, b) => a - b)[good.length >> 1];
  const m = 2.5 * th;
  const x0 = Math.max(0, Math.floor(Math.min(...xs) - m)), y0 = Math.max(0, Math.floor(Math.min(...ys) - m));
  const x1 = Math.min(W, Math.ceil(Math.max(...xs) + m)), y1 = Math.min(H, Math.ceil(Math.max(...ys) + m));
  const w = x1 - x0, h = y1 - y0;
  // only when the region is clearly smaller than the page (a new framing, at about the tiles' scale or sharper)
  if (w * h > 0.5 * W * H || det.size / Math.max(w, h) < 0.8 * tileScale) return lines;
  const d = det.detect(cropGray(gray, W, { x: x0, y: y0, w, h }), w, h, opts.params);
  const closer = recogniseLines(det, rec, d.lines, opts, [x0, y0]).map((l) => {
    const qx = [0, 2, 4, 6].map((i) => l.quad[i]), qy = [1, 3, 5, 7].map((i) => l.quad[i]);
    const mm = Math.max(2, 0.6 * l.thickness);
    const cut = (x0 > 0 && Math.min(...qx) < x0 + mm) || (x1 < W && Math.max(...qx) > x1 - mm)
      || (y0 > 0 && Math.min(...qy) < y0 + mm) || (y1 < H && Math.max(...qy) > y1 - mm);
    return { ...l, cut };
  });
  return orderLines(mergeDetections(lines, closer));
}
