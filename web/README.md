# HxH alphabet reader (browser demo)

A mobile-first page that reads one line of Hunter x Hunter script. You pick a photo (on
phones the file picker also offers the camera) or a built-in example, drag a box around one
line, and the int8 model runs in the browser through the Rust engine compiled to
WebAssembly. Nothing is uploaded. Plain HTML, CSS and ES modules: no frameworks, no CDN, no
build step beyond cargo.

## Files

| file | what |
|---|---|
| `index.html`, `style.css` | page and styles (light/dark via `prefers-color-scheme`) |
| `app.js` | UI: image loading, box drawing (pointer events, mouse + touch), zoom, results |
| `ocr.js` | everything the UI and the node tests share: grayscale, INTER_AREA resize, line preprocessing, auto-fit, wasm wrapper, CTC details, romaji |
| `hxh_infer.wasm` | `rust/hxh-infer-wasm`, opt-level 3 + `simd128` (33 KB) |
| `hxh_infer_nosimd.wasm` | same without SIMD (17 KB), used when `WebAssembly.validate` rejects SIMD |
| `models/model.hxhm` | the model (v0: `experiments/rust_engine/mnv4s_h32_ptq`, 338 KB) |
| `models/model.json` | model name, input size, CTC blank, vocab (class i = `vocab[i-1]`) |
| `examples/` | 6 lines from the `synth_v3` val split as lossless PNG, plus `examples.json` (labels) |
| `dist/hxh-ocr.html` | single self-contained page (built, about 640 KB) |
| `test/` | dev only, not part of the site: reference generator, node verification, example maker |

## Build and run

```powershell
# from the repo root
.\scripts\build_web.ps1            # wasm (both builds), model copy, dist\hxh-ocr.html
.\scripts\build_web.ps1 -Verify    # ... then run web\test\verify.mjs
cd web; python -m http.server 8000 # open http://localhost:8000/ (phone: same Wi-Fi, use the PC's IP)
```

`index.html` needs a web server (it fetches the wasm and model). `dist/hxh-ocr.html` makes
no requests at all (wasm, model, examples, CSS and JS are inlined as base64/text), so it can
be opened from disk or published as a standalone page.

## Swapping the model

1. Convert it to `.hxhm` (`scripts/espdl_to_hxhm.py convert ...`, see `rust/FORMAT.md`).
2. `.\scripts\build_web.ps1 -Model path\to\model.hxhm` copies it to `models/model.hxhm`.
3. Edit `models/model.json`: `name`, `input` and `vocab`. The page takes the input size and
   class count from the `.hxhm` itself and warns in the console if the JSON disagrees.

The page only assumes a 1-channel u8 input with an input LUT (normalisation folded in), a
`[1, 1, T, C]` logits tensor and CTC blank 0, which is what the converter writes.

## Preprocessing (matches training)

`ocr.js` reproduces the calibration pipeline (`experiments/esp_ppq_spike/spike_common.preprocess`,
`scripts/espdl_to_hxhm.resize_pad_u8`) bit for bit:

1. Crop the box from the image (drawn on white, so transparency reads as paper).
2. Grayscale with OpenCV's `COLOR_RGB2GRAY` fixed point (0.299, 0.587, 0.114 as 15-bit integers).
3. Resize to height 32 keeping aspect: width = Python `round(w * 32 / h)` (half to even), with a
   port of OpenCV's `INTER_AREA` (fractional box filter in float32 in OpenCV's operation order,
   the integer-factor fast path, and the bilinear variant OpenCV uses when upscaling).
4. If narrower than 256, pad right with the last column. If wider, squeeze to 256 and say so.
5. Optional invert (255 - v). The model input LUT turns u8 into int8, the engine runs, and the
   wasm export does the greedy CTC decode. JS recomputes per-character softmax confidence from
   the logits; characters under 60% are underlined, and tapping one shows the runner-up.

**Auto-fit (on by default).** Hand-drawn boxes are rarely framed like the training crops,
and the model is sensitive to that. Auto-fit finds the text band from horizontal-gradient
energy (robust to uneven light, gradients and horizontal clutter lines), frames it so the band
is 62% of the crop height with 0.3 x height side margins, reads both the drawn box and the
fitted one, and keeps the more likely greedy path (sum of log max-softmax). The dashed box on
the image shows when the fitted box won.

## Verification (node, no browser)

```powershell
.venv\Scripts\python.exe web\test\make_reference.py   # OpenCV references -> web\test\out (about 30 MB)
node web\test\verify.mjs --robust                      # add --wasm web\hxh_infer_nosimd.wasm for the fallback
```

Results (2026-11-24, OpenCV 4.12.0, 300 horizontal lines of `data/synth_v3/val`):

- `rgbaToGray` = `cv2.cvtColor(RGB2GRAY)` on 8/8 random images; `resizeArea` = `cv2.resize(INTER_AREA)`
  on 294/294 random cases (down, integer factors, fractional, up, mixed): bit-exact.
- JS strip = Python `resize_pad_u8` strip on **300/300 lines, bit-exact**, so decodes agree
  300/300. The wasm decoder and the JS CTC pass agree 300/300; `toRomaji` matches
  `hxh_ocr.glyphs.to_romaji` on all labels and the whole vocab. Both wasm builds pass.
- Accuracy of the v0 model, box = whole line: 245/300 exact (CER 8.1%); lines that fit in
  256 px: 197/223 (CER 3.0%); lines squeezed from >256 px: 48/77 (CER 13.8%).
- Robustness (223 lines that fit, CER, box as drawn -> auto-fit on): as is 3.0% -> 2.3%;
  inverted 3.2% -> 2.5%; loose box +25% height 15.9% -> 6.0%; +50% 44.8% -> 15.8%; box cutting
  10% top and bottom 10.0% -> 5.2%; off-centre 12.5% -> 4.7%; extra side margins 8.5% -> 4.0%;
  source only 16 px tall 5.0% -> 5.0%.
- Photos of a sign (two rendered pages, 8 lines): tight boxes 2/8 -> 7/8 exact with auto-fit;
  loose and sloppy boxes 8/8 either way.
- Speed: about 9-12 ms per inference in node (SIMD build), 13-15 ms without SIMD; preprocessing
  under 1 ms for a line, about 40 ms for a whole 1400 px photo.

## Known limitations

- **Static 32x256 input.** About 8 glyphs fit at 32 px height; longer boxes are squeezed, which
  the model never saw in training (CER 3% -> 14%). A dynamic-width model, or a static 32x512
  export (the training loader already allows widths up to 512), would fix long lines. Until
  then, box long lines in parts.
- **Synthetic training data only.** No real manga, anime or merch crops have been measured yet;
  expect a sim-to-real gap (fonts, halftones, perspective, motion blur).
- **One horizontal line per read.** No line detection, no vertical text (the manga often writes
  vertically), no punctuation. Rotation beyond a few degrees is not handled.
- **Dakuten / handakuten** (small dot or ring) are the main error source, worse when the text is
  small in the photo; the notice suggests getting closer when the box is under 32 px tall.
- Katakana and small kana come out as full-size hiragana (the glyphs are the same).
- Image decoding is the browser's: JPEG decoders differ by a level here and there between
  browsers, and very large photos are scaled to 3072 px on the long side on load.
- Needs WebAssembly (any current browser); SIMD is optional. HEIC opens only where the browser
  can decode it (iOS converts it when picking from the photo library).
