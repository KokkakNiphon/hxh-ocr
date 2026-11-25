# Writing a compute backend for `hxh-infer`

`hxh-infer` does all arithmetic through the `Kernels` trait (`hxh-infer/src/kernels.rs`).
Every method has a default that calls the portable scalar reference (`kernels::scalar`),
so a backend overrides only what it speeds up. Results must be **bit-exact** with the
scalar kernels: the golden files check it, and a mismatch anywhere is a bug.

| backend | where | overrides |
|---|---|---|
| `Scalar` | `kernels.rs`, module `scalar` | nothing (the reference) |
| `Simd128` | `kernels/wasm_simd.rs`, only on wasm32 with `+simd128` | conv2d, dwconv2d, avgpool, add_inplace, requant_inplace, resize_nearest, concat_copy |
| (future) ESP32-S3 TIE728 | | conv2d, dwconv2d (see the end of this file) |

`hxh_infer::Fast` names the fastest backend compiled into the build. The wasm wrapper uses
`Simd128` in a SIMD build (`hxh_set_simd(0)` switches to scalar for A/B checks).

## The trait

```rust
pub trait Kernels {
    fn conv2d(&self, g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]);
    fn dwconv2d(&self, g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]);
    fn avgpool(&self, g: &PoolGeom, x: &[i8], y: &mut [i8]);
    fn add_inplace(&self, acc: &mut [i8], other: &[i8]);
    fn lut_inplace(&self, buf: &mut [i8], table: &[i8; 256]);
    fn requant_inplace(&self, buf: &mut [i8], shift: i8);
    // format version 2 (the text-line detector)
    fn resize_nearest(&self, g: &ResizeGeom, x: &[i8], y: &mut [i8]);
    fn concat_copy(&self, x: &[i8], rows: usize, len: usize, y: &mut [i8], offset: usize, stride: usize);
}
```

A backend that accelerates only some shapes calls `scalar::conv2d` (etc.) for the rest.
A 1x1 conv (`kh = kw = sh = sw = 1`, no padding) is ~75% of the MACs of the real models;
3x3 convs (including the `cin = 1` stem) and depthwise 3x3 / 5x5 / 1x5 are the rest.

## What the engine guarantees to a backend

`Model::parse` validates everything once, so a backend can rely on the following:

- **Activations** are int8 NHWC with N = 1, zero point 0: `x[(iy * in_w + ix) * in_c + c]`.
  `x` and `y` have exactly `h * w * c` elements.
- **Conv filter**: `w[(n / 16) * kh*kw*cin*16 + ((ky * kw + kx) * cin + c) * 16 + n % 16]`,
  i.e. ESP-DL's `(N/16)HWC16`: for each block of 16 output channels, for each tap and
  input channel, 16 consecutive weights, one per output channel. A conv row segment
  `(ky, kx0..kx1, all c)` is contiguous both in the input and in the filter.
- **Depthwise filter**: `w[(n / 16) * kh*kw*16 + (ky * kw + kx) * 16 + n % 16]`.
- `cout_padded` = `out_c` rounded up to a multiple of 16. Filter and bias are padded to
  it with zeros, but `y` holds only `out_c` channels per pixel: **never write channels
  `>= out_c`**, and in a depthwise conv never *read* input channels `>= in_c` (the last
  block of the last pixel would run past the tensor).
- **Bias**: `cout_padded` little-endian i32 at exponent `in_exp + w_exp` (`Bias::get`, or
  `Bias::as_bytes` for vector loads). `|bias| <= 2^30` and `kh * kw * cin <= 2^15`, so
  `|bias + sum x*w| <= 2^30 + 2^29` fits in i32.
- **Alignment**: tensor arena offsets are 16-byte aligned. Weight, bias and LUT blobs are
  16-byte aligned relative to the file start, so they are aligned in memory only if the
  model buffer is (the wasm wrapper's buffers are 64 KiB aligned). Use unaligned loads
  unless the caller guarantees alignment.
- **Aliasing**: conv, depthwise, avgpool, resize and concat outputs never overlap their
  inputs. Add, Lut and Requant work in place (the engine copies first when the plan does not
  alias).
- **Resize**: `x` is `in_h x in_w x c`, `y` is `(in_h * sh) x (in_w * sw) x c`, factors >= 1.
- **Concat**: the engine calls `concat_copy` once per input with `rows` = product of the
  output dims before the axis (`h * w` for channels), `len` = the input's bytes per row and
  `offset` = the sum of the previous inputs' `len`; `stride` = output bytes per row.
- Geometry: `out_h = (in_h + pad_t + pad_b - kh) / sh + 1` (same for w); a window may lie
  partly or entirely in the padding (then the output is `requant(bias)`).

## Exact semantics

`requant(acc, s)`:

- `s > 0`: `(acc + 2^(s-1)) >> s` with an arithmetic shift (round half up: ties toward +inf,
  `-1.5 -> -1`, `2.5 -> 3`). Not half-to-even, not half-away-from-zero: each of those
  changes every test vector (see the journal, "hxh-infer v0").
- `s <= 0`: `acc << -s`, saturating.
- then saturate to [-128, 127], then ReLU (`max(0, .)`) if the op has it fused.

| op | result |
|---|---|
| conv2d / dwconv2d | `requant(bias[n] + sum over valid taps of x * w, shift)`; padded taps are skipped (= zero) |
| avgpool | `floor(f32(sum) * scale + 0.5)` in f32, saturated, with `scale = dl_scale(in_exp) * (1/area) * dl_rescale(out_exp)` computed in that order (`scalar::avgpool`) |
| add_inplace | `sat8(a + b)` (exponents are equal by construction) |
| lut_inplace | `table[x + 128]` |
| requant_inplace | `requant(x, shift)` without ReLU |
| resize_nearest | `y[oy][ox][c] = x[oy / sh][ox / sw][c]` (ESP-DL with equal exponents: a copy) |
| concat_copy | `y[offset + r * stride ..][..len] = x[r * len ..][..len]` for `r < rows` |

## Pitfalls (found while writing `Simd128`)

- **i16 overflow.** One int8 x int8 product always fits i16 (range [-16256, 16384]), but
  the sum of two does not: `(-128)*(-128) * 2 = 32768`. Accumulate pairs straight into
  i32: wasm `i32x4.dot_i16x8_s` on sign-extended operands, or `extmul` to i16 followed by
  an i32 pairwise add, or NEON `smlal` / `sadalp`. Do not add two i16 products in i16.
- **Rounding constant overflow.** `acc + 2^(s-1)` overflows i32 when `s >= 30` and
  `|acc|` is near its bound. Use `(acc >> s) + ((acc >> (s-1)) & 1)` (identical result),
  or 64-bit, for large shifts.
- **Negative shifts.** Clamp to [-128, 127] before shifting left (127 << 24 still fits).
- **Saturating narrows** (`i16x8.narrow_i32x4_s`, `i8x16.narrow_i16x8_s`, `sqxtn`) are
  exactly the int8 clamp. Apply ReLU after them.
- **Partial blocks.** The last output block may have fewer than 16 valid channels: store
  only those, and in depthwise mask the loads (`Simd128` copies the tail into a zeroed
  16-byte buffer).
- **Odd reduction lengths.** With two-term instructions (dot), pair the last term with
  a zero weight (the `cin = 1` stem has 3 terms per kernel row).
- **AvgPool** is float in ESP-DL. A vector version is exact only if it performs the same
  IEEE f32 operations in the same order (convert, multiply by the precomputed scale,
  add 0.5, floor, saturating truncation).
- **Resize / Concat** move a few bytes per pixel (8 to 80 in the detector): one
  `memory.copy` per segment is slow there, so `Simd128` uses 16-byte loads and stores and
  lets the last store of a segment overlap the previous one (same values). Segments
  shorter than 16 bytes go to scalar.

## Testing a backend

1. The golden files: run `Model::run(&YourBackend, ...)` on every vector of
   `hxh-infer/tests/data/*/golden.bin` and `experiments/rust_engine/*/golden.bin`, and
   compare the logits and the greedy CTC ids (see `tests/golden.rs` and
   `hxh-infer-wasm/test_wasm.mjs --all`).
2. A randomised differential test against `kernels::scalar`, for shapes the models never
   use: kernel 1..5, strides 1..3, asymmetric padding, odd channel counts, partial blocks,
   windows entirely in the padding, -128 everywhere, shifts -24..31, biases up to 2^30.
   `hxh-infer-wasm/src/selftest.rs` does this for `Simd128` (also Resize with factors 1..5
   and 1..70 channels, and Concat segments of 1..70 bytes at any offset and stride). Check
   that it catches a deliberately broken kernel before trusting a clean run (a copy kernel
   that skips its tail: 425 mismatches in 15,735 cases).

## ESP32-S3 (TIE728) backend notes

- The filter layout is ESP-DL's: one 16-byte load of `w[block][tap][c][0..16]` against one
  broadcast input byte (`EE.VLDBC.8`) is one `EE.VMULAS.S8.QACC` step for 16 output
  channels. A 1x1 conv is `cin` such steps per pixel and block; a k x k conv is a sum of
  contiguous row segments. Implement `conv2d` with a 1x1 fast path first: that is ~75%
  of the MACs.
- Depthwise: the 16 channels of one tap are a 16-byte load from the input pixel and one
  from the filter. `in_c` is a multiple of 16 in the current models, but a backend must
  still handle the partial last block (or fall back to scalar for it).
- QACC lanes hold 20 bits. The real models peak at |acc| = 73,193 against 524,287, but
  that is data-dependent: keep a check in the converter or the tests, and handle the bias
  outside QACC if it does not fit.
- Requantisation must be round-half-up as above; ESP-NN's TFLite-style requant is not
  (that is the 87 K-value mismatch in the journal). Verify the TIE728 shift/round instruction against
  `scalar::requant` on all shifts, including ties on negative values.
- Leave avgpool, add, lut, requant, resize and concat on the scalar defaults unless
  profiling says otherwise: they are a few percent of the time (in the toy detector Resize
  and Concat together are 3 %).
