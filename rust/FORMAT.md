# `.hxhm` model format (version 2) and engine semantics

`.hxhm` is the model file of `hxh-infer`, the no_std int8 engine for the HxH OCR line
recogniser and text-line detector. `scripts/espdl_to_hxhm.py` writes it from an ESP-PPQ
`.espdl` (ESP32-S3, int8). The file holds everything the engine needs and nothing else. Shapes, exponents,
shifts and arena offsets are precomputed, so the runtime parses nothing beyond fixed-size
records, never allocates, and reads weights zero-copy from the file buffer.

```
PyTorch -> ONNX -> ESP-PPQ (espdl_quantize_onnx, target esp32s3, w8a8) -> model.espdl
        -> scripts/espdl_to_hxhm.py -> model.hxhm -> hxh-infer (host / wasm32 / later ESP32-S3)
```

**Versions.** Version 2 adds the Resize (nearest) and Concat ops and the raw-output header
flag (for the detector, whose output is a map, not CTC logits). The engine reads versions 1
and 2. The converter writes version 1 whenever a model uses none of these, so recogniser
files stay byte-identical and older engine builds keep loading them; a version-2 op or flag
inside a version-1 file is rejected.

## Layout

All integers are little-endian. The file is: header, tensor table, op table, then data
blobs. Every table and blob starts at a 16-byte-aligned offset, and the file length is a
multiple of 16. If the buffer itself is 16-byte aligned, every weight, bias and LUT blob
is too, so a SIMD backend can load them directly.

### Header (64 bytes)

| off | type | field |
|---|---|---|
| 0 | `[u8;4]` | magic `"HXHM"` |
| 4 | u16 | version: 1, or 2 if the file uses Resize / Concat / the raw-output flag |
| 6 | u16 | header size = 64 |
| 8 | u32 | file size in bytes (trailing bytes beyond it are ignored) |
| 12 | u16 | number of tensors |
| 14 | u16 | number of ops |
| 16 | u16 | input tensor id |
| 18 | u16 | output (logits) tensor id |
| 20 | u32 | arena size in bytes |
| 24 | u32 | tensor table offset |
| 28 | u32 | op table offset |
| 32 | u32 | input LUT offset (256 bytes, see below), or 0 for none |
| 36 | u16 | `num_classes`: the decoder looks at the first `num_classes` output channels (padding classes beyond them are ignored); all channels for a raw-output model |
| 38 | u16 | CTC blank id (0) |
| 40 | u64 | MACs per inference (convs only) |
| 48 | u32 | flags. Bit 0 `INPUT_CLOBBERED`: the arena plan reuses the input's memory, so each `run` needs a fresh input. By default the input is kept intact and `run` can be repeated. Bit 1 `RAW_OUTPUT` (version 2): the output is a raw tensor (the detector's `[h, w, 2]` maps), not CTC logits; `decode` returns no ids. Other bits must be 0. |
| 52 | 12 B | reserved (0) |

### Tensor record (16 bytes)

Activations are int8, NHWC, N = 1, symmetric (zero point 0), with one power-of-two
exponent per tensor: `real = q * 2^exp`.

| off | type | field |
|---|---|---|
| 0 | u16 | h |
| 2 | u16 | w |
| 4 | u16 | c |
| 6 | i8 | exponent |
| 7 | u8 | reserved |
| 8 | u32 | arena offset (16-byte aligned) |
| 12 | u32 | size = h·w·c bytes |

The logits are a `[1, 1, T, C]` tensor (h = 1, w = T timesteps, c = classes). A
raw-output model's output is any NHWC tensor (the detector: `[1, S/2, S/2, 2]`).

### Op record (40 bytes)

| off | type | field |
|---|---|---|
| 0 | u8 | kind: 1 Conv, 2 DwConv, 3 Add, 4 Lut, 5 Requant, 6 AvgPool, 7 Resize (v2), 8 Concat (v2) |
| 1 | u8 | flags: bit 0 fused ReLU (Conv/DwConv); bit 1 "was ReduceMean" (AvgPool, informational) |
| 2 | u16 | input tensor (Concat: its first input) |
| 4 | u16 | second input (Add), else 0xFFFF |
| 6 | u16 | output tensor |
| 8 | u16 ×4 | kh, kw, sh, sw (Conv/DwConv/AvgPool). Resize: kh = kw = 0, sh, sw = the integer scale factors. Concat: kh = number of inputs n (2..64), kw = axis (1 = H, 2 = W, 3 = C), sh = sw = 0 |
| 16 | u8 ×4 | pads top, left, bottom, right (Conv/DwConv) |
| 20 | i8 | shift: Conv/DwConv `out_exp − in_exp − w_exp`; Requant `out_exp − in_exp` |
| 21 | i8 | filter exponent (Conv/DwConv) |
| 22 | u16 | `cout_padded`: output channels rounded up to a multiple of 16 |
| 24 | u32 | data offset: filter (Conv/DwConv), 256-byte table (Lut), or the n input tensor ids (Concat, u16 each) |
| 28 | u32 | data length |
| 32 | u32 | bias offset (int32 × `cout_padded`) |
| 36 | u32 | reserved (0) |

Ops are stored in execution order, which is the `.espdl` node order and is topologically
sorted.

### Blobs

- **Conv filter**: int8 `[cout_padded/16][kh][kw][cin][16]`. Output channel `n` is lane
  `n % 16` of block `n / 16`. The padding channels are zero. This is ESP-DL's
  `(N/16)HWC16` packing, extended to non-multiple-of-16 `Cout` by zero padding instead of
  ESP-DL's separate unaligned tail.
- **Depthwise filter**: int8 `[cout_padded/16][kh][kw][16]`.
- **Bias**: int32 `[cout_padded]` at exponent `in_exp + w_exp` (ESP-PPQ limits the values
  to a 20-bit range). The padding entries are 0.
- **LUT**: 256 int8 values, `y = table[x + 128]`. ESP-PPQ builds it by evaluating the
  activation on `x·2^in_exp` for x = −128..127 and quantising at the output exponent.
- **Concat inputs**: n little-endian u16 tensor ids, in concatenation order (`data_len = 2n`).
- **Input LUT** (u8 pixel → int8 input): it folds in the training normalisation
  `(p/255 − 0.5)/0.5` and the input quantisation `round_half_up(x / 2^in_exp)`, computed
  in float32 exactly as the Python pipeline does. With exponent −7, pixels 0/127/128/255
  map to −128/−1/+1/127. The int8 input is never 0.

### Arena plan

The converter assigns every tensor an arena offset by first fit, with 16-byte alignment
and lifetimes taken from op order (a Concat reads all of its n inputs). A tensor is freed
after its last reader. Add, Lut and Requant may write in place over an input that dies at
that op. In that case the record's output offset equals the input offset. Otherwise every
op's output range is disjoint from its input ranges (always for Resize and Concat, all n
inputs). `Model::parse` checks all of this, including the input-intact guarantee when flag
bit 0 is clear.

## Semantics (what the engine computes, bit-exactly)

`requant(acc, s)` = `(acc + 2^(s−1)) >> s` (arithmetic shift) for `s > 0`, or `acc << −s`
for `s ≤ 0`, then clamp to [−128, 127]. This is round-half-up: ties go toward +∞, so
−1.5 → −1 and 2.5 → 3.

| op | int8 result |
|---|---|
| Conv / DwConv | `requant(bias[n] + Σ x·w, shift)`, then `max(0, ·)` if ReLU is fused. Padded taps contribute 0. Accumulation is in i32 (`parse` bounds kh·kw·cin ≤ 2^15 and \|bias\| ≤ 2^30, so it cannot overflow). |
| Add | `clamp(a + b)`. All three exponents must be equal. ESP-DL's int8 Add ignores exponents, and ESP-PPQ ("Align to Output" + RequantizeLinear) guarantees equality. The converter rejects anything else. |
| Lut | `table[x + 128]` (ESP-DL: `table[(uint8)x ^ 0x80]`) |
| Requant | `requant(x, out_exp − in_exp)` |
| AvgPool | ESP-DL's float path: `floor(f32(sum) · (DL_SCALE(in_exp) · (1/area) · DL_RESCALE(out_exp)) + 0.5)`, clamped, all in f32. When the area is a power of two this is exact and equals a rounding shift. ReduceMean over H (or H,W) is converted to this op. |
| Resize | nearest, integer factors: `y[oy][ox][c] = x[oy / sh][ox / sw][c]`. Input and output exponents must be equal (see below), so it is a copy. |
| Concat | the inputs' bytes one after another along the axis (for C: per pixel, input 0's channels, then input 1's, ...). All input exponents must equal the output's. |
| decode | greedy CTC over the first `num_classes` channels per timestep. A tie goes to the lowest id (like `numpy.argmax`). Repeats collapse and the blank is dropped. A raw-output model decodes to nothing. |

**Resize and Concat in ESP-DL / ESP-PPQ** (esp-dl master, `dl/module/include/dl_module_concat.hpp`,
`dl_module_resize.hpp`, `dl/base/dl_base_resize.{hpp,cpp}`; esp-ppq 1.3.15):

- ESP-DL's int8 `Concat::forward` is `tool::copy_memory` of each input in turn: it never
  looks at exponents. ESP-PPQ's espdl setting has `align_concat_to = 'Align to Output'`: the
  Concat's input quantisation configs are slaved to its output config (`refine.py`,
  `align_to_output`), upstream ops with a single consumer adopt it, and the exporter's
  `InsertRequantNodePattern` puts a `RequantizeLinear` in front of any input whose producer
  keeps another exponent. The toy detector shows both (`experiments/det_engine/toy_det/espdl_nodes.txt`:
  `PPQ_Operation_3` requantises the FPN output p1 from exponent -3 to -4 for the Concat; the
  convs feeding it directly were moved to -4). The converter rejects a Concat with unequal
  exponents rather than guess. The `.espdl` axis is already NHWC (esp-ppq `ResetConcatPattern`).
- ESP-DL's int8 nearest Resize computes `round(x · output_scale / 2^output_shift)` with
  `output_shift = out_exp − in_exp` (a scale `2^-shift` when negative), i.e. a copy when the
  exponents are equal. `align_resize_to = 'Align to Output'` plus `RequantizeLinear` make them
  equal (the toy has two such requantisations before a Resize). The 2x2 case has its own
  kernel that writes each input pixel to a 2x2 block; other factors use
  `in_y = min(int(y · (1.f / scale)), in_h − 1)` in float32 ("asymmetric" coordinates, floor,
  whatever the ONNX `nearest_mode`), which ESP-PPQ's simulation reproduces with
  `F.interpolate(mode='nearest')`. The converter accepts only integer factors for which that
  float index equals `y // scale` for every output row and column (all powers of two), and
  warns where the ONNX fp32 semantics differ (asymmetric + `round_prefer_floor` at scale > 2).

## Accepted `.espdl` subset

The accepted subset is int8 (`quant_type S8`) only, with per-tensor exponents and the
S3/P4 filter packing `(N/16)HWC16[_UNALIGNED]`. Supported ops: Conv (group 1 or
depthwise, any kernel, stride and pad, no dilation, fused activation Linear or Relu),
Add (no broadcast), any unary op with a `lut` attribute, RequantizeLinear, AveragePool
(no padding, no ceil mode), ReduceMean / GlobalAveragePool over H or H,W with
keepdims, Resize (mode nearest, integer factors, equal exponents) and Concat (2..64
activation inputs along H, W or C, equal exponents). The converter rejects everything else,
with an error message, rather than guess at it.

## Golden files (test vectors)

`golden.bin` holds a 32-byte header (`"HXHG"`, u32 version 1, u32 n, u16 h, w, T, C,
max_ids, u16 0, u32 T, zero padding), then n records of
`u8 image[h·w] | i8 input[h·w] | i8 logits[T·C] | u16 n_ids | u16 ids[max_ids]`. The image
is the resized grayscale line, and the input is the int8 tensor that ESP-PPQ quantised from
it. The logits come from ESP-PPQ's `TorchExecutor` int8 simulation. The ids are greedy CTC
over the first `num_classes` channels; recogniser goldens have `max_ids = T`.
Detector goldens (`experiments/det_engine/golden_det.py`) use the same layout with
T = out_h · out_w, C = 2, `max_ids = 0` and `n_ids = 0`: the image is the letterboxed page
and the "logits" are the NHWC output maps. T does not fit u16 for the real detector
(320 · 320): the u16 field is then 0 and the u32 at offset 24 holds it (readers take the
u32 when the u16 is 0; older files have 0 there and a nonzero u16).

`golden_inter.bin` holds a 16-byte header (`"HXHI"`, u32 version, u32 vectors, u32
records), then records of `u32 vector | u16 tensor id | u16 0 | u32 len | i8 data[len]`.
These are NHWC intermediate tensors from the same simulation, used for per-layer
debugging. `tensors.txt` maps tensor ids to the `.espdl` names.

## Commands

```powershell
# from the repo root
$py = '.venv-ppq\Scripts\python.exe'
# convert only
& $py scripts\espdl_to_hxhm.py convert path\model.espdl -o path\model.hxhm [--num-classes 75] [--reuse-input]
# re-quantize like the spike + convert + golden vectors (+ copy into the rust test data)
& $py scripts\espdl_to_hxhm.py golden --onnx experiments\esp_ppq_spike\models\toy_trained_avg_op18.onnx --tqt `
    --out-dir experiments\rust_engine\toy_tqt --test-data rust\hxh-infer\tests\data\toy_tqt
& $py scripts\espdl_to_hxhm.py dump path\model.hxhm

# from rust
cargo test --release -- --nocapture          # bit-exact tests + statistics
cargo run --release -p hxh-cli -- hxh-infer\tests\data\toy_tqt\model.hxhm hxh-infer\tests\data\toy_tqt\sample.pgm `
    --vocab hxh-infer\tests\data\toy_tqt\vocab.txt --profile
cargo run --release -p hxh-cli -- hxh-infer\tests\data\toy_tqt\model.hxhm hxh-infer\tests\data\toy_tqt\golden.bin
# any other converted model + golden.bin (e.g. from experiments\rust_engine\golden_from_saved.py,
# which pairs quantize.py --save-outputs logits with the val lines): bit-exact check + statistics
$env:HXH_MODEL='D:\path\model.hxhm'; $env:HXH_GOLDEN='D:\path\golden.bin'
cargo test --release -p hxh-infer --test golden -- --ignored --nocapture external
cargo build -p hxh-infer-wasm --target wasm32-unknown-unknown --profile wasm
node hxh-infer-wasm\test_wasm.mjs target\wasm32-unknown-unknown\wasm\hxh_infer_wasm.wasm `
    hxh-infer\tests\data\toy_tqt\model.hxhm hxh-infer\tests\data\toy_tqt\golden.bin

# the text-line detector (rust/DETECT.md for the post-processing and its JS API)
$py = '.venv-ppq\Scripts\python.exe'   # (from the repo root)
& $py experiments\det_engine\toy_det.py --train-seconds 75          # toy detector -> models\toy_det.onnx
& $py experiments\det_engine\golden_det.py quantize --onnx experiments\det_engine\models\toy_det.onnx `
    --out-dir experiments\det_engine\toy_det --test-data rust\hxh-infer\tests\data\toy_det
& $py scripts\espdl_to_hxhm.py convert det.espdl -o det.hxhm --raw-output   # convert only
cargo run --release -p hxh-detect --example detect -- golden hxh-infer\tests\data\toy_det\model.hxhm `
    hxh-infer\tests\data\toy_det\golden.bin hxh-infer\tests\data\toy_det\lines_default.txt
node hxh-infer-wasm\test_detect.mjs target\web-simd\wasm32-unknown-unknown\wasm\hxh_infer_wasm.wasm
& $py experiments\det_engine\check_postprocess.py    # post-processing vs OpenCV (needs the example built)

# browser builds, as scripts\build_web.ps1 makes them (separate target dirs keep the flag sets apart)
$env:CARGO_PROFILE_WASM_OPT_LEVEL = '3'
$env:CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS = '-C target-feature=+simd128'
cargo build -j 4 -p hxh-infer-wasm --target wasm32-unknown-unknown --profile wasm --target-dir target\web-simd
Remove-Item Env:CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS
cargo build -j 4 -p hxh-infer-wasm --target wasm32-unknown-unknown --profile wasm --target-dir target\web-nosimd
# every golden set (3 toy recognisers + the toy detector + the 5 real models in experiments\rust_engine)
node hxh-infer-wasm\test_wasm.mjs target\web-simd\wasm32-unknown-unknown\wasm\hxh_infer_wasm.wasm --all
node hxh-infer-wasm\test_wasm.mjs target\web-simd\wasm32-unknown-unknown\wasm\hxh_infer_wasm.wasm --all --scalar
# dev build with the randomised scalar-vs-SIMD kernel test and the per-op profile
$env:CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS = '-C target-feature=+simd128'
cargo build -j 4 -p hxh-infer-wasm --features selftest,profile --target wasm32-unknown-unknown --profile wasm --target-dir target\dev-simd
node hxh-infer-wasm\test_wasm.mjs target\dev-simd\wasm32-unknown-unknown\wasm\hxh_infer_wasm.wasm --selftest 20000 `
    --profile ..\experiments\rust_engine\qat_v3_mnv4s_h48_ptq\model.hxhm ..\experiments\rust_engine\qat_v3_mnv4s_h48_ptq\golden.bin
```

The `wasm` profile uses opt-level "z", fat LTO, one codegen unit, panic=abort and strip.
The browser build uses `CARGO_PROFILE_WASM_OPT_LEVEL=3`. With
`-C target-feature=+simd128` the build also contains the SIMD128 kernels
(`hxh-infer/src/kernels/wasm_simd.rs`), which it uses by default; without it only the
scalar kernels exist. All builds are bit-exact. `rust/KERNELS.md` describes the kernel
contract and what a new backend (e.g. ESP32-S3) must implement.
