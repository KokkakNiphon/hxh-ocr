# Text-line detection in the browser: `hxh_det_*` JS API

Full-page reading runs two models through the same wasm module (`hxh-infer-wasm`, built as
in `rust/FORMAT.md`): the **detector** finds the text lines, then every line crop goes
through the existing recogniser path (`prepareLine` + `hxh_run`). The detector's
post-processing (thresholding, connected components, rectangles, reading order, crops) is
Rust (`rust/hxh-detect`, no_std, no allocation) exported as raw `extern "C"` functions.
This file is everything needed to wire it into the demo.

```
page (u8 gray, any size)
  -> hxh_det_image + hxh_det_letterbox   detector input S x S (S = 640): long side -> S, top-left, pad 255
  -> hxh_run                             detector .hxhm (format v2, raw output): int8 [S/2, S/2, 2]
  -> hxh_det_detect                      lines: quads in page pixels, reading order, orientation
  -> hxh_det_crop(i)                     upright u8 strip per line, native resolution
  -> prepareLine(...) + recogniser instance hxh_run   (unchanged JS)
```

## Two instances

The module's bump allocator holds one model per instance, so instantiate the module twice
from the same compiled `WebAssembly.Module`: one instance loads the detector, the other the
recogniser. The `hxh_det_*` exports are used on the detector instance; crops are copied out
of it (they are plain `Uint8Array`s) and fed to the recogniser instance.

```js
const mod = await WebAssembly.compile(wasmBytes);           // the SIMD or no-SIMD build
const det = (await WebAssembly.instantiate(mod, {})).exports;
const rec = (await WebAssembly.instantiate(mod, {})).exports;
load(det, detectorHxhm);   // hxh_alloc + copy + hxh_load, as HxhEngine.create does
load(rec, recogniserHxhm);
```

**Detached views.** Any call that allocates (`hxh_load`, `hxh_det_image`,
`hxh_det_letterbox`, `hxh_det_detect*` the first time, `hxh_det_crop*` when a crop is bigger
than any before) may grow the memory and detach every existing view of `memory.buffer`.
Create views *after* such calls and copy what you keep (`.slice()`).

## Coordinates

Pixel centres at integers: pixel `(x, y)` covers `[x - 0.5, x + 0.5] x [y - 0.5, y + 0.5]`,
so a box around pixels 0..W-1 runs from -0.5 to W - 0.5 (the `rust/hxh-synth`
convention). Quads are in **original page pixels** and **clockwise from the top-left in
reading orientation**:

- horizontal line: TL, TR, BR, BL (`q0 -> q1` runs left to right along the text);
- vertical line (column): the column's TR, BR, BL, TL, i.e. `q0 -> q1` runs down the
  column. That is the TL, TR, BR, BL of the strip after turning the column 90 degrees
  counter-clockwise, which is exactly how `hxh_det_crop` returns it.

In both cases `|q0 q1|` is the line's length and `|q1 q2|` its thickness.

## Exports

All return values are `i32`/`u32`/pointers (numbers in JS). Negative = error (table below).

| export | arguments | returns |
|---|---|---|
| `hxh_det_image(w, h)` | page size | pointer to a `w*h` byte buffer for the grayscale page (row-major). Write the page there. Forgets the previous frame and lines. 0 on failure. |
| `hxh_det_letterbox()` | | 0 or error. Letterboxes the page into the loaded model's input buffer (`hxh_image_ptr`, `hxh_image_width x hxh_image_height`): scale `s = min(W_in / w, H_in / h)`, size `round_half_up(w s) x round_half_up(h s)`, area average when shrinking (as OpenCV `INTER_AREA`, within 1 grey level), bilinear when growing (as `INTER_LINEAR`), top-left, rest 255. Records the frame for `hxh_det_detect`. |
| `hxh_det_content_width()`, `hxh_det_content_height()` | | size of the page inside the letterbox (0 before a frame is set) |
| `hxh_run()` | | as for the recogniser; returns 0 (a raw-output model has no CTC ids). `hxh_raw_output()` = 1 for the detector; `hxh_out_height()`, `hxh_out_width()`, `hxh_channels()`, `hxh_logits_ptr()`, `hxh_logits_exp()` describe its int8 output. |
| `hxh_det_params_ptr()` | | pointer to 8 **f64** parameters (a `Float64Array`), editable before detecting; see below |
| `hxh_det_detect()` | | number of lines (>= 0) or error. Post-processes the output of the last `hxh_run` with the frame of the last `hxh_det_letterbox` (or `hxh_det_set_frame`; without either, the model input is taken as the page). At most 1024 lines. |
| `hxh_det_lines_ptr()` | | pointer to `count x 16` f32 line records, **sorted in reading order** (see below) |
| `hxh_det_line_count()` | | count of the last detection |
| `hxh_det_crop(i)` | line index | pointer to the rectified crop of line `i` from the page given to `hxh_det_image` (`record[14] x record[15]` bytes, row-major), 0 if `i` is out of range or there is no page. The buffer is reused by the next crop: copy it. |
| `hxh_det_crop_width()`, `hxh_det_crop_height()` | | size of the last crop |
| `hxh_det_quad_ptr()` | | pointer to 8 f32 (`x0 y0 x1 y1 x2 y2 x3 y3`, page pixels) for `hxh_det_crop_quad` |
| `hxh_det_crop_quad(w, h)` | output size, 0 = native | crop of the quad at `hxh_det_quad_ptr` (e.g. a "tap a line" box, or a ground-truth quad), same rules as `hxh_det_crop`; 0 on failure |
| `hxh_det_set_frame(in_w, in_h, content_w, content_h, orig_w, orig_h)` | | 0 or -34. Declares how a map relates to the page when the letterbox was not made by `hxh_det_letterbox`. |
| `hxh_det_detect_map(ptr, h, w, c, exp)` | an int8 NHWC map in wasm memory (`hxh_alloc`) | number of lines or error. Same as `hxh_det_detect` on any map (tests, a detector run elsewhere); needs a frame. |

**Error codes:** -2 no model loaded; -30 no page (`hxh_det_image` first); -32 no frame
(`hxh_det_detect_map` before `hxh_det_letterbox` / `hxh_det_set_frame`); -33 out of memory;
-34 bad size argument; -35 the loaded model is not a raw-output (detector) model. Model
loading itself keeps its codes (-10..-20); an older build rejects a version-2 `.hxhm` with
-12 (bad version).

### Parameters (`Float64Array(memory.buffer, hxh_det_params_ptr(), 8)`)

| index | name | default | meaning |
|---|---|---|---|
| 0 | `bin_thresh` | 0.3 | a map pixel is text when `sigmoid(kernel logit) > bin_thresh`; evaluated exactly as an integer comparison on the int8 logit (0.3 <=> logit > -0.8473) |
| 1 | `min_score` | 0.5 | drop a line whose mean kernel probability over its component is lower |
| 2 | `min_side` | 3 | drop a line whose final box has a short side below this many **letterbox** pixels |
| 3 | `unclip_ratio` | 1.5 | DBNet unclip: grow the kernel rectangle by `D = area * ratio / perimeter` on every side |
| 4 | `shrink_ratio` | 0 (off) | if > 0, replaces 3: the exact inverse of the training shrink `D = A (1 - r^2) / L` with `r = shrink_ratio` (`rust/hxh-synth` targets use r = 0.4) |
| 5 | `block_gap` | 1.0 | reading order: lines of the same orientation join a block when their axis-aligned boxes are at most `block_gap x` the thicker line's thickness apart in x and y |
| 6, 7 | | 0 | reserved |

### Line record (16 f32 per line, `Float32Array(memory.buffer, hxh_det_lines_ptr(), 16 * n)`)

| index | field |
|---|---|
| 0..7 | `x0 y0 x1 y1 x2 y2 x3 y3`: the quad in page pixels (see Coordinates) |
| 8 | score: mean kernel probability over the component |
| 9 | vertical: 1 if the mean vertical-orientation logit over the component is > 0, else 0 |
| 10 | order: rank in reading order (equals the record's index) |
| 11 | block: block id, numbered in reading order |
| 12 | length `|q0 q1|` (page px) |
| 13 | thickness `|q1 q2|` (page px) |
| 14, 15 | crop width and height = `round(length)`, `round(thickness)`, at least 1 |

## Example: read a page

```js
const check = (rc) => { if (rc < 0) throw new Error(`hxh_det error ${rc}`); return rc; };

function detectLines(det, gray, W, H, params = {}) {
  const pp = det.hxh_det_image(W, H);
  if (!pp) throw new Error("page buffer allocation failed");
  new Uint8Array(det.memory.buffer, pp, W * H).set(gray);   // view made after the allocating call
  check(det.hxh_det_letterbox());
  check(det.hxh_run());
  const p = new Float64Array(det.memory.buffer, det.hxh_det_params_ptr(), 8);
  Object.entries(params).forEach(([k, v]) => (p[{ bin_thresh: 0, min_score: 1, min_side: 2, unclip_ratio: 3, shrink_ratio: 4, block_gap: 5 }[k]] = v));
  const n = check(det.hxh_det_detect());
  const recs = new Float32Array(det.memory.buffer, det.hxh_det_lines_ptr(), 16 * n).slice();
  const lines = [];
  for (let i = 0; i < n; i++) {
    const r = recs.subarray(16 * i, 16 * i + 16);
    const ptr = det.hxh_det_crop(i);
    const [cw, ch] = [r[14], r[15]];
    const crop = new Uint8Array(det.memory.buffer, ptr, cw * ch).slice();
    lines.push({ quad: Array.from(r.subarray(0, 8)), score: r[8], vertical: r[9] === 1, block: r[11], crop, cw, ch });
  }
  return lines; // in reading order
}

// recognise: every crop is an upright strip, so the existing line path applies as is
for (const l of detectLines(det, gray, W, H)) {
  const { pixels } = prepareLine(l.crop, l.cw, l.ch, { height: recEngine.height, width: recEngine.width });
  const { ids } = recEngine.run(pixels);  // HxhEngine on the recogniser instance
  text.push(idsToText(ids, vocab));
}
```

(A fresh view is created after every call that may allocate.) A vertical recogniser, if used, wants the column upright instead: rotate the
crop back 90 degrees clockwise, or crop the quad `q1 q2 q3 q0` with `hxh_det_crop_quad`.

## What the post-processing does (`rust/hxh-detect`)

1. **Threshold** channel 0 in the integer domain: the largest int8 `t` with
   `sigmoid(t 2^exp) <= bin_thresh` is found once per map; a pixel is text iff `q > t`.
2. **Components**: 8-connectivity, breadth-first, in raster order of their first pixel.
   Per component: pixel count, mean probability (score), sum of the channel-1 logits
   (integer; vertical iff > 0).
3. **Shape**: the convex hull of the component's pixel *squares* (on every horizontal grid
   line, the outermost square corners of the two adjacent rows). A kernel `k` pixels thick
   therefore measures `k`; OpenCV contours (pixel centres, what DBNet/PaddleOCR use) measure
   `k - 1`, about 1 px less per side (measured: +1.00 px median on both sides).
4. **Minimum-area rectangle** of the hull by rotating calipers (O(m); one side lies on a hull
   edge). Identical to `cv2.minAreaRect` on the same points (max corner deviation 5e-5 px,
   OpenCV is float32) except for exact equal-area ties, where either rectangle is minimal
   (e.g. a diagonal pixel pair: 2 x 2 or 2.83 x 1.41; about 3 % of tiny components).
5. **Unclip** by `D` (parameter 3 or 4), then the filters: score, short side in letterbox
   px, and the centre must lie on the page content (not the letterbox padding).
6. **Quad**: the reading axis is the rectangle axis closest to +x (horizontal) or +y
   (vertical), so a single square glyph keeps its orientation. Grid -> letterbox:
   `x_lb = (u + 0.5) * in_w / map_w - 0.5`; letterbox -> page:
   `x = (x_lb + 0.5) * orig_w / content_w - 0.5` (same for y).
7. **Reading order**: blocks by union-find (same orientation, gap rule above); blocks top to
   bottom by the top of their box (ties: further right first); inside a block horizontal lines
   top to bottom (then left to right), vertical columns right to left (then top to bottom).
   Manga panel order (right to left across panels) is not modelled.
8. **Crop**: the projective map of the unit square onto the quad (exactly affine for the
   rectangles above), bilinear sampling with edge replication, at native resolution: width
   `round(|q0 q1|)`, height `round(|q1 q2|)`. An axis-aligned quad on pixel boundaries is an
   exact copy; a column comes out turned 90 degrees counter-clockwise.

Everything uses f64 with software `sqrt`/`exp`/`floor` (core has no float math), so the host
(`cargo run -p hxh-detect --example detect`) and wasm produce bit-identical records and crops.

### Choosing the unclip

The detector predicts shrunk kernels; how far to grow them back depends on how sharp its
kernel map is, so tune parameters 3/4 on the real model:

- With the exact rectangle shrink `r` used for the targets, a long line of height `h` has a
  kernel `(1 - (1 - r^2)) h = r^2 h` thick: 0.16 h at r = 0.4. DBNet's `unclip_ratio = 1.5`
  only grows that back to `2.5 x 0.16 h = 0.4 h`; it relies on the network's blurry map
  being much fatter than the kernel at `bin_thresh = 0.3`. `shrink_ratio = r` inverts the
  shrink exactly, but amplifies kernel errors by `1 / r^2` (6x at r = 0.4): one map pixel of
  error is 12 letterbox pixels of line height at stride 2.
- On the toy detector (75 s of CPU training, r = 0.7 targets) the predicted kernels at 0.3
  were already about as fat as the lines: box IoU with ground truth was best without any
  unclip (median 0.70; ratio 0.5: 0.54; 1.5: 0.28; exact inverse r = 0.7: 0.45). Measure the
  same on generated pages for the real model (golden_det.py `meta.json` holds the ground-truth
  quads of synthetic inputs).

## Cost (node 24, Ryzen 7 3700X, one thread, CPU shared)

- post-processing of a 320 x 320 x 2 map with 60 lines: about 1-2 ms;
- letterbox of a 2480 x 3508 page (A4 at 300 dpi): about 6-8 ms;
- the detector itself: the toy (56 M MACs at 128 x 128) runs at 22 GMAC/s with SIMD; the
  real model's MACs at 640 x 640 decide the rest.
- wasm size: the browser builds grow from 53.6 to 79.8 KB (SIMD; 20.1 -> 31.3 KB gzip) and
  from 17.2 to 37.9 KB (no SIMD; 7.4 -> 16.4 KB gzip).

## Tests

```powershell
# from rust
cargo test --release -p hxh-detect        # rectangles vs brute force, unclip, ordering, crops, letterbox
cargo build --release -p hxh-detect --example detect
node hxh-infer-wasm\test_detect.mjs target\web-simd\wasm32-unknown-unknown\wasm\hxh_infer_wasm.wasm
# from the repo root: OpenCV comparison on random maps (components, hulls, rectangles, lines, letterbox)
.venv-ppq\Scripts\python.exe experiments\det_engine\check_postprocess.py --n 300
# a whole page on the host: letterbox -> model -> lines -> crops as PGM files
cargo run --release -p hxh-detect --example detect -- page det.hxhm page.pgm --crops out_dir [--shrink 0.4]
```
