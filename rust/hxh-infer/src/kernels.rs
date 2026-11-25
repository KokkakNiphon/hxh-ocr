//! Kernel interface and the portable scalar backend.
//!
//! Every method of [`Kernels`] has a default that calls the scalar implementation in
//! [`scalar`], so a SIMD backend overrides only what it accelerates:
//!
//! ```ignore
//! struct EspNn;
//! impl hxh_infer::Kernels for EspNn {
//!     fn conv2d(&self, g: &ConvGeom, x: &[i8], w: &[i8], b: Bias<'_>, y: &mut [i8]) {
//!         if g.kh == 1 && g.kw == 1 && g.in_c % 16 == 0 { /* FFI / asm */ } else { scalar::conv2d(g, x, w, b, y) }
//!     }
//! }
//! ```
//!
//! Contract shared by all implementations (and checked bit-exactly by the golden tests):
//! activations are NHWC int8 with zero point 0; conv accumulates `bias + sum(x * w)` in
//! (at least) 32 bits, then `requant(acc, shift)` = arithmetic shift right with
//! round-half-up (`(acc + 2^(shift-1)) >> shift`, or a saturating left shift when
//! `shift < 0`), saturate to int8, then ReLU (clamp at 0) if fused. Padding taps contribute
//! nothing (= zero padding). See `rust/KERNELS.md` for what a new backend must implement.
//!
//! Resize (nearest, integer factors) and Concat are pure data movement with equal exponents,
//! as in ESP-DL's int8 modules.
//!
//! Backends: [`Scalar`] (portable reference, always available) and, on wasm32 builds with
//! `simd128`, `wasm_simd::Simd128`. [`crate::Fast`] names the fastest one compiled in.

/// int32 little-endian bias vector stored in the model file (zero-copy).
///
/// The converter places it at a 16-byte-aligned file offset, so a backend that knows the
/// model buffer is 16-byte aligned may read it as `&[i32]` directly via [`Bias::as_bytes`].
#[derive(Clone, Copy, Debug)]
pub struct Bias<'a> {
    bytes: &'a [u8],
}

impl<'a> Bias<'a> {
    #[inline]
    pub fn new(bytes: &'a [u8]) -> Self {
        Bias { bytes }
    }

    /// Number of entries (`cout_padded`).
    #[inline]
    pub fn len(&self) -> usize {
        self.bytes.len() / 4
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    #[inline]
    pub fn get(&self, i: usize) -> i32 {
        let b = &self.bytes[i * 4..i * 4 + 4];
        i32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    #[inline]
    pub fn as_bytes(&self) -> &'a [u8] {
        self.bytes
    }

    #[inline]
    fn load16(&self, first: usize, acc: &mut [i32; 16]) {
        let b = &self.bytes[first * 4..first * 4 + 64];
        for (a, c) in acc.iter_mut().zip(b.chunks_exact(4)) {
            *a = i32::from_le_bytes([c[0], c[1], c[2], c[3]]);
        }
    }
}

/// Geometry of a conv or depthwise conv. Activations are NHWC (N = 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConvGeom {
    pub in_h: usize,
    pub in_w: usize,
    pub in_c: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub out_c: usize,
    /// `out_c` rounded up to a multiple of 16: the weight/bias blocking.
    pub cout_padded: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub pad_t: usize,
    pub pad_l: usize,
    /// out_exp - in_exp - w_exp.
    pub shift: i8,
    pub relu: bool,
}

/// Geometry of an average pool (no padding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolGeom {
    pub in_h: usize,
    pub in_w: usize,
    pub c: usize,
    pub out_h: usize,
    pub out_w: usize,
    pub kh: usize,
    pub kw: usize,
    pub sh: usize,
    pub sw: usize,
    pub in_exp: i8,
    pub out_exp: i8,
}

/// Geometry of a nearest-neighbour Resize by integer factors (NHWC, N = 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResizeGeom {
    pub in_h: usize,
    pub in_w: usize,
    pub c: usize,
    /// Integer scale factors: the output is `in_h * sh` x `in_w * sw`.
    pub sh: usize,
    pub sw: usize,
}

/// Compute backend. All methods default to the portable scalar kernels.
pub trait Kernels {
    /// Regular conv. `w`: `(cout_padded/16)[kh][kw][in_c][16]`, `y`: `out_h*out_w*out_c`.
    fn conv2d(&self, g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]) {
        scalar::conv2d(g, x, w, bias, y)
    }

    /// Depthwise conv (`in_c == out_c`). `w`: `(cout_padded/16)[kh][kw][16]`.
    fn dwconv2d(&self, g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]) {
        scalar::dwconv2d(g, x, w, bias, y)
    }

    /// Average pool with ESP-DL's float rescale.
    fn avgpool(&self, g: &PoolGeom, x: &[i8], y: &mut [i8]) {
        scalar::avgpool(g, x, y)
    }

    /// `acc[i] = sat8(acc[i] + other[i])` (all exponents equal).
    fn add_inplace(&self, acc: &mut [i8], other: &[i8]) {
        scalar::add_inplace(acc, other)
    }

    /// `buf[i] = table[buf[i] + 128]`.
    fn lut_inplace(&self, buf: &mut [i8], table: &[i8; 256]) {
        scalar::lut_inplace(buf, table)
    }

    /// `buf[i] = requant(buf[i], shift)` (RequantizeLinear between exponents).
    fn requant_inplace(&self, buf: &mut [i8], shift: i8) {
        scalar::requant_inplace(buf, shift)
    }

    /// Nearest Resize by integer factors: `y[oy][ox][c] = x[oy / sh][ox / sw][c]`.
    fn resize_nearest(&self, g: &ResizeGeom, x: &[i8], y: &mut [i8]) {
        scalar::resize_nearest(g, x, y)
    }

    /// One Concat input: `y[offset + r * stride ..][..len] = x[r * len ..][..len]` for
    /// `r < rows` (the engine calls it once per input, with increasing `offset`).
    fn concat_copy(&self, x: &[i8], rows: usize, len: usize, y: &mut [i8], offset: usize, stride: usize) {
        scalar::concat_copy(x, rows, len, y, offset, stride)
    }
}

/// The portable scalar backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct Scalar;

impl Kernels for Scalar {}

/// wasm32 SIMD128 backend (only in builds with `-C target-feature=+simd128`).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub mod wasm_simd;

/// Portable scalar kernels: the reference semantics.
pub mod scalar {
    use super::{Bias, ConvGeom, PoolGeom, ResizeGeom};

    /// Power-of-two requantisation of an accumulator: shift right with round-half-up
    /// (ties toward +inf, as ESP-DL on the S3 and ESP-PPQ's `ROUND_HALF_UP`), or shift
    /// left when `shift < 0`; then saturate to int8 and apply ReLU.
    #[inline(always)]
    pub fn requant(acc: i32, shift: i8, relu: bool) -> i8 {
        let v = if shift > 0 {
            (acc as i64 + (1i64 << (shift - 1))) >> shift
        } else {
            (acc as i64) << (-shift as u32)
        };
        let lo = if relu { 0 } else { -128 };
        v.clamp(lo, 127) as i8
    }

    /// ESP-DL `tool::round` on the S3: `floor(v + 0.5f)` in f32 (`floor.s`).
    #[inline]
    pub fn round_half_up_f32(v: f32) -> i32 {
        let u = v + 0.5f32;
        let t = u as i32; // truncates toward zero (saturating)
        if (t as f32) > u { t - 1 } else { t }
    }

    /// ESP-DL `DL_SCALE(e)`: 2^e as f32.
    #[inline]
    pub fn dl_scale(e: i8) -> f32 {
        if e > 0 { (1i32 << e) as f32 } else { 1.0f32 / (1i32 << -(e as i32)) as f32 }
    }

    /// ESP-DL `DL_RESCALE(e)`: 2^-e as f32.
    #[inline]
    pub fn dl_rescale(e: i8) -> f32 {
        if e > 0 { 1.0f32 / (1i32 << e) as f32 } else { (1i32 << -(e as i32)) as f32 }
    }

    /// Valid kernel-tap range `[k0, k1)` for an output position whose window starts at
    /// input coordinate `i0` (may be negative because of padding).
    #[inline(always)]
    pub(crate) fn tap_range(i0: isize, k: usize, n: usize) -> (usize, usize) {
        let k0 = (-i0).max(0) as usize;
        let k1 = (n as isize - i0).clamp(0, k as isize) as usize;
        (k0, k1.max(k0))
    }

    #[inline(always)]
    fn mac16(acc: &mut [i32; 16], xs: &[i8], ws: &[i8]) {
        for (&xv, wr) in xs.iter().zip(ws.chunks_exact(16)) {
            let xv = xv as i32;
            for j in 0..16 {
                acc[j] += xv * wr[j] as i32;
            }
        }
    }

    /// Regular conv, 16 output channels at a time (matches the weight blocking).
    pub fn conv2d(g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]) {
        let cin = g.in_c;
        let tap = cin * 16;
        let blk = g.kh * g.kw * tap;
        let nblk = g.cout_padded / 16;
        for oy in 0..g.out_h {
            let iy0 = (oy * g.sh) as isize - g.pad_t as isize;
            let (ky0, ky1) = tap_range(iy0, g.kh, g.in_h);
            for ox in 0..g.out_w {
                let ix0 = (ox * g.sw) as isize - g.pad_l as isize;
                let (kx0, kx1) = tap_range(ix0, g.kw, g.in_w);
                let yp = &mut y[(oy * g.out_w + ox) * g.out_c..][..g.out_c];
                for b in 0..nblk {
                    let mut acc = [0i32; 16];
                    bias.load16(b * 16, &mut acc);
                    let wb = &w[b * blk..][..blk];
                    for ky in ky0..ky1 {
                        let iy = (iy0 + ky as isize) as usize;
                        for kx in kx0..kx1 {
                            let ix = (ix0 + kx as isize) as usize;
                            let xs = &x[(iy * g.in_w + ix) * cin..][..cin];
                            let ws = &wb[(ky * g.kw + kx) * tap..][..tap];
                            mac16(&mut acc, xs, ws);
                        }
                    }
                    let n = (g.out_c - b * 16).min(16);
                    for (o, &a) in yp[b * 16..b * 16 + n].iter_mut().zip(acc.iter()) {
                        *o = requant(a, g.shift, g.relu);
                    }
                }
            }
        }
    }

    /// Depthwise conv, 16 channels at a time.
    pub fn dwconv2d(g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]) {
        let c = g.in_c;
        let blk = g.kh * g.kw * 16;
        let nblk = g.cout_padded / 16;
        for oy in 0..g.out_h {
            let iy0 = (oy * g.sh) as isize - g.pad_t as isize;
            let (ky0, ky1) = tap_range(iy0, g.kh, g.in_h);
            for ox in 0..g.out_w {
                let ix0 = (ox * g.sw) as isize - g.pad_l as isize;
                let (kx0, kx1) = tap_range(ix0, g.kw, g.in_w);
                let yp = &mut y[(oy * g.out_w + ox) * c..][..c];
                for b in 0..nblk {
                    let n = (c - b * 16).min(16);
                    let mut acc = [0i32; 16];
                    bias.load16(b * 16, &mut acc);
                    let wb = &w[b * blk..][..blk];
                    for ky in ky0..ky1 {
                        let iy = (iy0 + ky as isize) as usize;
                        for kx in kx0..kx1 {
                            let ix = (ix0 + kx as isize) as usize;
                            let ws = &wb[(ky * g.kw + kx) * 16..][..16];
                            let xs = &x[(iy * g.in_w + ix) * c + b * 16..][..n];
                            if n == 16 {
                                for j in 0..16 {
                                    acc[j] += xs[j] as i32 * ws[j] as i32;
                                }
                            } else {
                                for j in 0..n {
                                    acc[j] += xs[j] as i32 * ws[j] as i32;
                                }
                            }
                        }
                    }
                    for (o, &a) in yp[b * 16..b * 16 + n].iter_mut().zip(acc.iter()) {
                        *o = requant(a, g.shift, g.relu);
                    }
                }
            }
        }
    }

    /// Average pool as ESP-DL computes it for int8: int32 window sum, then
    /// `tool::round(sum * (DL_SCALE(in_exp) * (1.f / area) * DL_RESCALE(out_exp)))` in f32,
    /// saturated to int8. Exact (and equal to a rounding shift) when `area` is a power of 2.
    pub fn avgpool(g: &PoolGeom, x: &[i8], y: &mut [i8]) {
        let area = g.kh * g.kw;
        let scale = dl_scale(g.in_exp) * (1.0f32 / area as f32) * dl_rescale(g.out_exp);
        let c = g.c;
        let mut out = 0;
        for oy in 0..g.out_h {
            for ox in 0..g.out_w {
                for ch in 0..c {
                    let mut sum = 0i32;
                    for ky in 0..g.kh {
                        let row = (oy * g.sh + ky) * g.in_w;
                        for kx in 0..g.kw {
                            sum += x[(row + ox * g.sw + kx) * c + ch] as i32;
                        }
                    }
                    y[out] = round_half_up_f32(sum as f32 * scale).clamp(-128, 127) as i8;
                    out += 1;
                }
            }
        }
    }

    pub fn add_inplace(acc: &mut [i8], other: &[i8]) {
        for (a, &b) in acc.iter_mut().zip(other) {
            *a = a.saturating_add(b);
        }
    }

    pub fn lut_inplace(buf: &mut [i8], table: &[i8; 256]) {
        for v in buf.iter_mut() {
            *v = table[(*v as u8 ^ 0x80) as usize];
        }
    }

    pub fn requant_inplace(buf: &mut [i8], shift: i8) {
        for v in buf.iter_mut() {
            *v = requant(*v as i32, shift, false);
        }
    }

    /// Nearest Resize by integer factors (ESP-DL with equal exponents: a copy). Each output
    /// row is built once from its input row, then repeated `sh - 1` times.
    pub fn resize_nearest(g: &ResizeGeom, x: &[i8], y: &mut [i8]) {
        let c = g.c;
        let row = g.in_w * g.sw * c;
        for iy in 0..g.in_h {
            let y0 = iy * g.sh * row;
            let dst = &mut y[y0..y0 + row];
            for (ix, px) in x[iy * g.in_w * c..][..g.in_w * c].chunks_exact(c).enumerate() {
                for d in dst[ix * g.sw * c..][..g.sw * c].chunks_exact_mut(c) {
                    d.copy_from_slice(px);
                }
            }
            for r in 1..g.sh {
                y.copy_within(y0..y0 + row, y0 + r * row);
            }
        }
    }

    pub fn concat_copy(x: &[i8], rows: usize, len: usize, y: &mut [i8], offset: usize, stride: usize) {
        for r in 0..rows {
            y[offset + r * stride..][..len].copy_from_slice(&x[r * len..][..len]);
        }
    }
}
