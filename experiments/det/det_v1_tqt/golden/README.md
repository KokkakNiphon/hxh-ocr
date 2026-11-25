# Detector golden vectors (det_v1_tqt)

Model: `../model.espdl` (ESP-PPQ, target esp32s3, int8 per-tensor power-of-two,
PTQ + TQT, calibration kl on val[0:64]).
Source ONNX: `det_v1.onnx`.

- `inputs_u8.npy`: uint8 [64, 640, 640], pages val[900:964] of
  `data/pages_v2`, letterboxed (long side scaled to 640 with cv2 INTER_AREA when shrinking /
  INTER_LINEAR when enlarging, placed top-left, right/bottom padded with 255).
- `input_lut_int8.npy`: int8 [256], u8 pixel -> int8 model input. The float input is
  (x / 255 - 0.5) / 0.5, quantised with exponent -7 (round half up, clip to [-128, 127]).
- `outputs_int8.npy`: int8 [64, 2, 320, 320] (NCHW), ESP-PPQ TorchExecutor simulated output
  (the .espdl semantics), stored as integers; float value = int * 2^-3.
  Channel 0 = text-kernel logit, channel 1 = vertical-orientation logit.
- Output exponent -3: the prob threshold logit > -0.8473 (prob 0.3) becomes
  int > -6.778, i.e. int >= -6.

## The detector (det_v1) for the Rust / browser side

This is the recommended int8 model (ESP-PPQ PTQ + TQT). `../../det_v1_ptq/` holds the plain-PTQ
build of the same ONNX for comparison (F1 1.2 points lower). Python references:
`hxh_ocr/det_data.py` (letterbox, normalisation), `hxh_ocr/det_post.py` (post-processing),
`scripts/eval_det.py` (crops for the recogniser).

Graph in `model.espdl`: Conv x60 (ReLU fused), Add x13, RequantizeLinear x5, Resize x10,
Concat x1. Every Resize is nearest, scale 2 (ONNX attrs coordinate_transformation_mode=asymmetric,
nearest_mode=floor: out[y, x] = in[y // 2, x // 2]) with equal input and output exponents (a pure
copy). The single Concat (4 x 16 channels -> 64, channel axis) has all inputs at the output's
exponent (a byte copy); ESP-PPQ inserted RequantizeLinear where needed, as for the Adds.
Input exponent -7, output exponent -3. The output exponent comes from calibration (the smoke model
got -4), so read it from the model; never hard-code it.

Holdout (val[900:1000], 100 pages, 1,657 lines), same post-processing:

| | P | R | F1 | orientation | e2e CER h | e2e CER v |
|---|---|---|---|---|---|---|
| fp32 ONNX | 0.928 | 0.904 | 0.916 | 99.2% | 8.6% | 16.4% |
| int8 PTQ (det_v1_ptq) | 0.923 | 0.886 | 0.904 | 99.1% | 9.7% | 16.5% |
| int8 PTQ + TQT (this) | 0.927 | 0.902 | 0.914 | 99.1% | 9.1% | 18.2% |

(Only 222 vertical lines / 780 characters in this holdout, so the v column is noisy.)

### Post-processing (hxh_ocr/det_post.py, defaults)

1. Binary map: logit > -0.8473 (prob > 0.3); with exponent -3 that is `q >= -6`.
2. Connected components, 8-connectivity, on the 320 x 320 map.
3. Drop components with < 3 pixels, or with score < 0.5, where score = mean of sigmoid(logit)
   over the component's pixels. Vertical if the mean vertical logit (channel 1) over the
   component's pixels is > 0.
4. cv2.minAreaRect over the pixel centres (x, y) of the component; add 1 to both side lengths
   (pixel extent). With side lengths wk, hk (output px), unclip each side by 2 * D,
   D = 0.7 * Dx, where Dx is the positive root of
   (8 - 4a) Dx^2 + (2 - 2a)(wk + hk) Dx - a wk hk = 0, a = 1 - 0.4^2 = 0.84
   (the exact inverse of the training shrink; 0.7 because the predicted region is fatter than
   the training kernel, tuned on val[300:600]). DBNet's D = A * 1.5 / L of the rectangle is
   almost as good (F1 0.920 vs 0.922).
5. Corners: output (u, v) -> letterbox input (2u + 0.5, 2v + 0.5) -> original page
   ((x + 0.5) / sx - 0.5, (y + 0.5) / sy - 0.5), with sx = new_w / w, sy = new_h / h from the
   letterbox (round(w * 640 / max(w, h)), same for h).
6. Order the corners clockwise from the top-left in reading orientation. Horizontal: the
   reading direction is the rectangle axis closest to +x; corners TL, TR, BR, BL. Vertical: the
   reading direction is the axis closest to +y (down); corners top-right, bottom-right,
   bottom-left, top-left (the generator's convention).
7. Drop boxes whose short side is < 3 original px.

Reading order (`reading_order`): lines of the same orientation join a block when their
thicknesses differ by at most 2x, they overlap along the reading axis (x for horizontal, y for
vertical, axis-aligned boxes) and the gap across is at most 0.8 x the thinner line's thickness;
blocks are sorted by top edge (ties: right edge, right first); inside a block horizontal lines go
top-to-bottom and vertical columns right-to-left.

Recogniser crops (`scripts/eval_det.py::crop_strip`): grow the quad by 0.3 line heights along
the reading direction and 0.2 across, perspective-warp it to an upright W x H strip at native
resolution (quad corner 0 -> (-0.5, -0.5), 1 -> (W - 0.5, -0.5), ...; edge replicate). Because
vertical quads start at the column's top-right, their strips come out rotated 90 degrees
counter-clockwise, which is what the vertical recogniser expects. Then the recogniser's own
preprocessing: height 48 (INTER_AREA), width round(w * 48 / h) capped at 768, right pad by edge
replication.
