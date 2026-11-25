//! wasm32 SIMD128 backend (compiled only with `target_feature = "simd128"`).
//!
//! Same integer semantics as [`super::scalar`], bit for bit. The core primitive is
//! `i32x4.dot_i16x8_s`: two int8 x int8 products summed into an i32 lane, exact for every
//! input (each product is in [-16256, 16384] and the pair sum fits easily in i32, while an
//! i16 accumulator could not hold two (-128)·(-128) products). The int8 operands are
//! sign-extended to i16 first, and the two reduction terms of a lane are interleaved:
//!
//! * regular / pointwise conv: input channels `c, c+1` of one pixel (the x pair is a
//!   2-byte splat), against weight rows `w[c][0..16]` and `w[c+1][0..16]` byte-interleaved
//!   (`[w_c[0], w_c+1[0], w_c[1], w_c+1[1], ...]`); four dots cover the 16 output channels
//!   of a weight block. A conv row segment `(ky, kx0..kx1, 0..cin)` is contiguous in both
//!   NHWC input and the `(N/16)HWC16` filter, so every conv is a sum of such segments.
//!   Pixels whose windows are not clipped horizontally (all of a 1x1 conv) go through the
//!   weight-stationary [`conv_chunk`]: each group of two channel pairs is interleaved once
//!   and swept over up to [`CHUNK`] pixels with accumulators in an L1 stack array. That
//!   keeps the 1x1 conv near one `pmaddwd` per cycle on x86-64. Border pixels take
//!   [`conv_pixel`] one at a time.
//! * depthwise: two kernel taps of the same 16 channels, x and w both interleaved per tap
//!   pair. The interleaved weights of all tap pairs are cached per channel block and reused
//!   for every pixel with the same (clipped) window.
//! * AvgPool: i16 window sums, then the same f32 operations as the scalar code; Add and
//!   RequantizeLinear: saturating i8 / i16 lane arithmetic.
//! * Resize (nearest) and Concat: 16-byte vector moves per pixel / row segment (the last move
//!   of a segment overlaps the previous one), instead of one small `memory.copy` each.
//!
//! An odd tail term is paired with a zero weight. Requantisation uses the overflow-free form
//! `(acc >> s) + ((acc >> (s-1)) & 1)` (= `(acc + 2^(s-1)) >> s`) when `2^(s-1)` could
//! overflow i32, clamps before a left shift, and saturates to int8 with the two narrowing
//! instructions; ReLU is a `max(0, ·)` after saturation, as in ESP-DL.
//!
//! Every kernel is checked against the scalar reference by the golden files and by a
//! randomised differential test (`hxh-infer-wasm` feature `selftest`).

use core::arch::wasm32::*;

use super::scalar::{self, tap_range};
use super::{Bias, ConvGeom, Kernels, PoolGeom, ResizeGeom};

/// wasm32 SIMD128 kernels for conv, depthwise conv, AvgPool, Add, RequantizeLinear, Resize
/// and Concat (Lut falls back to the scalar default).
#[derive(Clone, Copy, Debug, Default)]
pub struct Simd128;

impl Kernels for Simd128 {
    fn conv2d(&self, g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]) {
        conv2d(g, x, w, bias, y)
    }

    fn dwconv2d(&self, g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]) {
        dwconv2d(g, x, w, bias, y)
    }

    fn avgpool(&self, g: &PoolGeom, x: &[i8], y: &mut [i8]) {
        avgpool(g, x, y)
    }

    fn add_inplace(&self, acc: &mut [i8], other: &[i8]) {
        add_inplace(acc, other)
    }

    fn requant_inplace(&self, buf: &mut [i8], shift: i8) {
        requant_inplace(buf, shift)
    }

    fn resize_nearest(&self, g: &ResizeGeom, x: &[i8], y: &mut [i8]) {
        resize_nearest(g, x, y)
    }

    fn concat_copy(&self, x: &[i8], rows: usize, len: usize, y: &mut [i8], offset: usize, stride: usize) {
        concat_copy(x, rows, len, y, offset, stride)
    }
}

/// Output pixels per weight-stationary conv chunk: the interleaved weights of two input
/// channel pairs are built once and swept over up to this many pixels, whose accumulators
/// live in a stack array (32 x 64 B, in L1). Rebuilding the interleave for every pixel or
/// every pair of pixels costs about as much as the multiplies themselves.
const CHUNK: usize = 32;

/// Largest depthwise window handled by the SIMD path (kh * kw); larger ones use scalar.
const DW_MAX_TAPS: usize = 64;

/// 16 int32 accumulators (output channels `b*16 .. b*16+16`).
type Acc = [v128; 4];

#[inline(always)]
unsafe fn ld(p: *const i8) -> v128 {
    // SAFETY: the caller guarantees 16 readable bytes (v128_load is unaligned).
    unsafe { v128_load(p.cast::<v128>()) }
}

/// Byte-interleave `a` and `b` and sign-extend to i16: element `2k` / `2k+1` of result `j`
/// is `a[4j+k]` / `b[4j+k]`, i.e. the operand pairs of lanes `4j..4j+4` for `i32x4_dot_i16x8`.
#[inline(always)]
fn interleave_i16(a: v128, b: v128) -> [v128; 4] {
    let lo = i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b);
    let hi = i8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b);
    [
        i16x8_extend_low_i8x16(lo),
        i16x8_extend_high_i8x16(lo),
        i16x8_extend_low_i8x16(hi),
        i16x8_extend_high_i8x16(hi),
    ]
}

#[inline(always)]
fn mac(acc: &mut Acc, xv: v128, wv: &[v128; 4]) {
    for j in 0..4 {
        acc[j] = i32x4_add(acc[j], i32x4_dot_i16x8(xv, wv[j]));
    }
}

#[inline(always)]
unsafe fn load_bias(p: *const u8) -> Acc {
    // SAFETY: the caller guarantees 64 readable bytes (16 little-endian i32).
    unsafe { [0, 16, 32, 48].map(|o| v128_load(p.add(o).cast::<v128>())) }
}

/// Per-layer requantisation constants.
#[derive(Clone, Copy)]
struct Rq {
    shift: i32,
    /// `2^(shift-1)` when adding it cannot overflow i32 (`shift <= 29`).
    round: Option<v128>,
    relu: bool,
}

impl Rq {
    fn new(shift: i8, relu: bool) -> Self {
        let shift = shift as i32;
        // |acc| <= 2^30 + 2^29 (Model::parse bounds bias and taps), so adding up to 2^28 is safe
        let round = (1..=29).contains(&shift).then(|| i32x4_splat(1 << (shift - 1)));
        Rq { shift, round, relu }
    }

    /// `requant` without the final clamp (the narrowing below saturates).
    #[inline(always)]
    fn shift4(&self, a: v128) -> v128 {
        let s = self.shift;
        if s > 0 {
            match self.round {
                Some(r) => i32x4_shr(i32x4_add(a, r), s as u32),
                // floor(a / 2^s) + bit (s-1) of a == (a + 2^(s-1)) >> s, without the overflow
                None => i32x4_add(i32x4_shr(a, s as u32), v128_and(i32x4_shr(a, (s - 1) as u32), i32x4_splat(1))),
            }
        } else {
            // clamp first so the left shift (by at most 24) cannot overflow: 127 << 24 < 2^31
            let a = i32x4_max(i32x4_min(a, i32x4_splat(127)), i32x4_splat(-128));
            i32x4_shl(a, (-s) as u32)
        }
    }

    /// Requantise 16 accumulators to 16 int8 (saturated, then ReLU if fused).
    #[inline(always)]
    fn pack(&self, acc: &Acc) -> v128 {
        let r = acc.map(|a| self.shift4(a));
        let v = i8x16_narrow_i16x8(i16x8_narrow_i32x4(r[0], r[1]), i16x8_narrow_i32x4(r[2], r[3]));
        if self.relu { i8x16_max(v, i8x16_splat(0)) } else { v }
    }
}

/// Store the first `n` (1..=16) bytes of `v` at `out`.
#[inline(always)]
unsafe fn store(out: *mut i8, v: v128, n: usize) {
    // SAFETY: the caller guarantees `n` writable bytes at `out`.
    unsafe {
        if n == 16 {
            v128_store(out.cast::<v128>(), v);
        } else {
            let mut t = [0i8; 16];
            v128_store(t.as_mut_ptr().cast::<v128>(), v);
            core::ptr::copy_nonoverlapping(t.as_ptr(), out, n);
        }
    }
}

// ------------------------------------------------------------------------- conv2d

/// `[x_i, x_i+1]` as an i16 pair repeated in every i32 lane (the x operand of a dot).
#[inline(always)]
unsafe fn x_pair(p: *const i8) -> v128 {
    // SAFETY: the caller guarantees 2 readable bytes.
    unsafe { i16x8_extend_low_i8x16(v128_load16_splat(p.cast::<u16>())) }
}

/// `acc += sum_{i < len} x[i] * w[i][0..16]` (one pixel, one row segment).
#[inline(always)]
unsafe fn dot_seg(acc: &mut Acc, x: *const i8, w: *const i8, len: usize) {
    let mut i = 0;
    // SAFETY (whole body): the caller guarantees `len` readable bytes at x and `len * 16`
    // at w; every access below is inside those ranges.
    unsafe {
        while i + 2 <= len {
            mac(acc, x_pair(x.add(i)), &interleave_i16(ld(w.add(i * 16)), ld(w.add(i * 16 + 16))));
            i += 2;
        }
        if i < len {
            // odd tail: pair with a zero weight
            mac(acc, i16x8_splat(*x.add(i) as i16), &interleave_i16(ld(w.add(i * 16)), i8x16_splat(0)));
        }
    }
}

/// Shared per-layer constants of a conv.
struct ConvCtx {
    w: *const i8,
    bias: *const u8,
    /// bytes per weight block (16 output channels)
    blk: usize,
    nblk: usize,
    out_c: usize,
    /// input bytes between kernel rows (`in_w * cin`) and filter bytes between rows (`kw * cin * 16`)
    x_row: usize,
    w_row: usize,
    rq: Rq,
}

/// All output channels of one pixel whose (clipped) window has `nrows` kernel rows, each a
/// contiguous segment of `len` input bytes starting at `x` (next row `x_row` further),
/// against the filter at `w_off` inside each block; `out_c` bytes are written at `y`.
#[inline(always)]
unsafe fn conv_pixel(c: &ConvCtx, x: *const i8, y: *mut i8, w_off: usize, nrows: usize, len: usize) {
    for b in 0..c.nblk {
        // SAFETY: b < nblk; bias holds nblk*16 i32, the filter nblk*blk bytes, and the caller
        // guarantees the segments (rows, len) and out_c bytes at y.
        unsafe {
            let mut acc = load_bias(c.bias.add(b * 64));
            let wb = c.w.add(b * c.blk + w_off);
            for r in 0..nrows {
                dot_seg(&mut acc, x.add(r * c.x_row), wb.add(r * c.w_row), len);
            }
            store(y.add(b * 16), c.rq.pack(&acc), (c.out_c - b * 16).min(16));
        }
    }
}

/// `npx <= CHUNK` pixels whose windows have the same shape, with inputs `xstep` bytes apart
/// (`x0` is the first pixel's first segment) and outputs `out_c` bytes apart from `y0`.
/// Same contract as [`conv_pixel`], but weight-stationary: each group of 2 input-channel
/// pairs is interleaved once and applied to all pixels.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_chunk(
    c: &ConvCtx,
    x0: *const i8,
    xstep: usize,
    y0: *mut i8,
    npx: usize,
    w_off: usize,
    nrows: usize,
    len: usize,
) {
    let mut acc = [[i32x4_splat(0); 4]; CHUNK];
    let acc = &mut acc[..npx];
    for b in 0..c.nblk {
        // SAFETY: as in conv_pixel; pixel p reads its segments at x0 + p*xstep (+ r*x_row) and
        // writes out_c bytes at y0 + p*out_c. Pointers past the last pixel are only formed
        // with wrapping_add and never dereferenced.
        unsafe {
            let bv = load_bias(c.bias.add(b * 64));
            acc.fill(bv);
            let wb = c.w.add(b * c.blk + w_off);
            for r in 0..nrows {
                let xr = x0.add(r * c.x_row);
                let wr = wb.add(r * c.w_row);
                let mut i = 0;
                while i + 4 <= len {
                    let w0 = interleave_i16(ld(wr.add(i * 16)), ld(wr.add(i * 16 + 16)));
                    let w1 = interleave_i16(ld(wr.add(i * 16 + 32)), ld(wr.add(i * 16 + 48)));
                    let mut xp = xr.add(i);
                    for a in acc.iter_mut() {
                        let (xa, xb) = (x_pair(xp), x_pair(xp.add(2)));
                        for j in 0..4 {
                            a[j] = i32x4_add(a[j], i32x4_add(i32x4_dot_i16x8(xa, w0[j]), i32x4_dot_i16x8(xb, w1[j])));
                        }
                        xp = xp.wrapping_add(xstep);
                    }
                    i += 4;
                }
                if i + 2 <= len {
                    let wv = interleave_i16(ld(wr.add(i * 16)), ld(wr.add(i * 16 + 16)));
                    let mut xp = xr.add(i);
                    for a in acc.iter_mut() {
                        mac(a, x_pair(xp), &wv);
                        xp = xp.wrapping_add(xstep);
                    }
                    i += 2;
                }
                if i < len {
                    // odd tail: pair with a zero weight
                    let wv = interleave_i16(ld(wr.add(i * 16)), i8x16_splat(0));
                    let mut xp = xr.add(i);
                    for a in acc.iter_mut() {
                        mac(a, i16x8_splat(*xp as i16), &wv);
                        xp = xp.wrapping_add(xstep);
                    }
                }
            }
            let n = (c.out_c - b * 16).min(16);
            for (p, a) in acc.iter().enumerate() {
                store(y0.add(p * c.out_c + b * 16), c.rq.pack(a), n);
            }
        }
    }
}

/// Regular conv (any kernel, stride and padding), bit-exact with [`scalar::conv2d`].
pub fn conv2d(g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]) {
    let cin = g.in_c;
    let tap = cin * 16;
    let w_row = g.kw * tap;
    let blk = g.kh * w_row;
    let nblk = g.cout_padded / 16;
    // These make every raw access below in bounds (Model::parse guarantees them; checked
    // again because the trait is public).
    assert!(g.cout_padded.is_multiple_of(16) && g.out_c <= g.cout_padded && g.cout_padded < g.out_c + 16);
    assert!(x.len() >= g.in_h * g.in_w * cin && w.len() >= nblk * blk && bias.len() >= g.cout_padded);
    assert!(y.len() >= g.out_h * g.out_w * g.out_c);
    let c = ConvCtx {
        w: w.as_ptr(),
        bias: bias.as_bytes().as_ptr(),
        blk,
        nblk,
        out_c: g.out_c,
        x_row: g.in_w * cin,
        w_row,
        rq: Rq::new(g.shift, g.relu),
    };
    let (xp, yp) = (x.as_ptr(), y.as_mut_ptr());
    let oc = g.out_c;

    let pointwise = g.kh == 1 && g.kw == 1 && g.sh == 1 && g.sw == 1 && g.pad_t == 0 && g.pad_l == 0;
    if pointwise && g.out_h == g.in_h && g.out_w == g.in_w {
        // 1x1: one flat list of pixels, each a single segment of cin bytes
        let npix = g.out_h * g.out_w;
        for p in (0..npix).step_by(CHUNK) {
            // SAFETY: pixel q < npix reads x[q*cin..][..cin] and writes y[q*oc..][..oc].
            unsafe { conv_chunk(&c, xp.add(p * cin), cin, yp.add(p * oc), CHUNK.min(npix - p), 0, 1, cin) }
        }
        return;
    }

    // Output columns whose window lies fully inside the input horizontally.
    let ox_lo = g.pad_l.div_ceil(g.sw).min(g.out_w);
    let ox_hi = if g.in_w + g.pad_l >= g.kw { ((g.in_w + g.pad_l - g.kw) / g.sw + 1).min(g.out_w) } else { 0 };
    let ox_hi = ox_hi.max(ox_lo);
    for oy in 0..g.out_h {
        let iy0 = (oy * g.sh) as isize - g.pad_t as isize;
        let (ky0, ky1) = tap_range(iy0, g.kh, g.in_h);
        let nrows = ky1 - ky0;
        // first valid input row (only dereferenced when nrows > 0)
        let iy = if nrows > 0 { (iy0 + ky0 as isize) as usize } else { 0 };
        let yrow = oy * g.out_w;
        // one pixel with its own clipped horizontal range
        let single = |ox: usize| {
            let ix0 = (ox * g.sw) as isize - g.pad_l as isize;
            let (kx0, kx1) = tap_range(ix0, g.kw, g.in_w);
            let len = (kx1 - kx0) * cin;
            let start = if nrows > 0 && len > 0 { (iy * g.in_w + (ix0 + kx0 as isize) as usize) * cin } else { 0 };
            // SAFETY: rows iy..iy+nrows and columns ix0+kx0..ix0+kx1 are inside the input.
            unsafe { conv_pixel(&c, xp.add(start), yp.add((yrow + ox) * oc), (ky0 * g.kw + kx0) * tap, nrows, len) }
        };
        for ox in 0..ox_lo {
            single(ox);
        }
        let len = g.kw * cin;
        for ox in (ox_lo..ox_hi).step_by(CHUNK) {
            let start = if nrows > 0 { (iy * g.in_w + ox * g.sw - g.pad_l) * cin } else { 0 };
            let npx = CHUNK.min(ox_hi - ox);
            // SAFETY: for ox in ox_lo..ox_hi the window columns ox*sw-pad_l .. +kw are inside the input.
            unsafe { conv_chunk(&c, xp.add(start), g.sw * cin, yp.add((yrow + ox) * oc), npx, ky0 * w_row, nrows, len) }
        }
        for ox in ox_hi..g.out_w {
            single(ox);
        }
    }
}

// ----------------------------------------------------------------------- dwconv2d

/// Load 16 channels, or only the first `n` (zero-filled) for a partial last block.
#[inline(always)]
unsafe fn ld_n<const FULL: bool>(p: *const i8, n: usize) -> v128 {
    // SAFETY: the caller guarantees 16 (FULL) or n readable bytes at p.
    unsafe {
        if FULL {
            ld(p)
        } else {
            let mut t = [0i8; 16];
            core::ptr::copy_nonoverlapping(p, t.as_mut_ptr(), n);
            ld(t.as_ptr())
        }
    }
}

/// Depthwise conv, bit-exact with [`scalar::dwconv2d`].
pub fn dwconv2d(g: &ConvGeom, x: &[i8], w: &[i8], bias: Bias<'_>, y: &mut [i8]) {
    let ch = g.in_c;
    let nblk = g.cout_padded / 16;
    let taps = g.kh * g.kw;
    if taps > DW_MAX_TAPS {
        return scalar::dwconv2d(g, x, w, bias, y);
    }
    assert!(g.out_c == ch && g.cout_padded.is_multiple_of(16) && ch <= g.cout_padded && g.cout_padded < ch + 16);
    assert!(x.len() >= g.in_h * g.in_w * ch && w.len() >= nblk * taps * 16 && bias.len() >= g.cout_padded);
    assert!(y.len() >= g.out_h * g.out_w * ch);
    let rq = Rq::new(g.shift, g.relu);
    for b in 0..nblk {
        let n = (ch - b * 16).min(16);
        if n == 16 {
            dw_block::<true>(g, x, w, bias, y, &rq, b, n);
        } else {
            dw_block::<false>(g, x, w, bias, y, &rq, b, n);
        }
    }
}

/// One block of 16 channels over all pixels. The tap list (and the interleaved weights of
/// its pairs) depends only on the clipped window, so it is rebuilt only when that changes.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn dw_block<const FULL: bool>(
    g: &ConvGeom,
    x: &[i8],
    w: &[i8],
    bias: Bias<'_>,
    y: &mut [i8],
    rq: &Rq,
    b: usize,
    n: usize,
) {
    let ch = g.in_c;
    let (xp, yp) = (x.as_ptr(), y.as_mut_ptr());
    // SAFETY: b < nblk (bias and filter sizes asserted by the caller)
    let bv = unsafe { load_bias(bias.as_bytes().as_ptr().add(b * 64)) };
    let wb = w[b * g.kh * g.kw * 16..].as_ptr();
    let zero = i8x16_splat(0);
    // valid taps of the current window: input offset relative to the window origin
    let mut xoff = [0isize; DW_MAX_TAPS];
    let mut wv = [[zero; 4]; DW_MAX_TAPS.div_ceil(2)];
    let mut ntaps = 0;
    let mut key = (usize::MAX, 0, 0, 0);
    for oy in 0..g.out_h {
        let iy0 = (oy * g.sh) as isize - g.pad_t as isize;
        let (ky0, ky1) = tap_range(iy0, g.kh, g.in_h);
        for ox in 0..g.out_w {
            let ix0 = (ox * g.sw) as isize - g.pad_l as isize;
            let (kx0, kx1) = tap_range(ix0, g.kw, g.in_w);
            if key != (ky0, ky1, kx0, kx1) {
                key = (ky0, ky1, kx0, kx1);
                ntaps = 0;
                let mut wo = [0usize; DW_MAX_TAPS];
                for ky in ky0..ky1 {
                    for kx in kx0..kx1 {
                        xoff[ntaps] = (ky * g.in_w + kx) as isize * ch as isize;
                        wo[ntaps] = (ky * g.kw + kx) * 16;
                        ntaps += 1;
                    }
                }
                for k in 0..ntaps.div_ceil(2) {
                    // SAFETY: tap offsets are inside block b of the filter (kh*kw*16 bytes)
                    let (a, c) = unsafe {
                        (ld(wb.add(wo[2 * k])), if 2 * k + 1 < ntaps { ld(wb.add(wo[2 * k + 1])) } else { zero })
                    };
                    wv[k] = interleave_i16(a, c);
                }
            }
            let mut acc = bv;
            // window origin (may lie in the padding); every valid tap lands inside x
            let base = (iy0 * g.in_w as isize + ix0) * ch as isize + (b * 16) as isize;
            let mut k = 0;
            // SAFETY: base + xoff[t] is the start of channel block b of a valid input pixel;
            // it has 16 channels left (FULL) or n.
            unsafe {
                while k + 1 < ntaps {
                    let xa = ld_n::<FULL>(xp.offset(base + xoff[k]), n);
                    let xb = ld_n::<FULL>(xp.offset(base + xoff[k + 1]), n);
                    let xv = interleave_i16(xa, xb);
                    let wk = &wv[k / 2];
                    for j in 0..4 {
                        acc[j] = i32x4_add(acc[j], i32x4_dot_i16x8(xv[j], wk[j]));
                    }
                    k += 2;
                }
                if k < ntaps {
                    let xv = interleave_i16(ld_n::<FULL>(xp.offset(base + xoff[k]), n), zero);
                    let wk = &wv[k / 2];
                    for j in 0..4 {
                        acc[j] = i32x4_add(acc[j], i32x4_dot_i16x8(xv[j], wk[j]));
                    }
                }
                store(yp.add((oy * g.out_w + ox) * ch + b * 16), rq.pack(&acc), n);
            }
        }
    }
}

// ------------------------------------------------------------------------ avgpool

/// Average pool, bit-exact with [`scalar::avgpool`]: the window sum is exact in i16 lanes
/// (area <= 256), and the f32 steps (convert, multiply by the same scale, + 0.5, floor,
/// saturating truncation) are the IEEE operations the scalar code performs one by one.
pub fn avgpool(g: &PoolGeom, x: &[i8], y: &mut [i8]) {
    let area = g.kh * g.kw;
    let c = g.c;
    if area > 256 || c < 16 {
        return scalar::avgpool(g, x, y);
    }
    assert!(g.out_h >= 1 && g.out_w >= 1 && (g.out_h - 1) * g.sh + g.kh <= g.in_h && (g.out_w - 1) * g.sw + g.kw <= g.in_w);
    assert!(x.len() >= g.in_h * g.in_w * c && y.len() >= g.out_h * g.out_w * c);
    let scale = scalar::dl_scale(g.in_exp) * (1.0f32 / area as f32) * scalar::dl_rescale(g.out_exp);
    let (sv, half) = (f32x4_splat(scale), f32x4_splat(0.5));
    let quarter = |s: v128| -> v128 {
        let v = f32x4_add(f32x4_mul(f32x4_convert_i32x4(s), sv), half);
        i32x4_trunc_sat_f32x4(f32x4_floor(v))
    };
    for oy in 0..g.out_h {
        for ox in 0..g.out_w {
            let o = (oy * g.out_w + ox) * c;
            // 16-channel groups; the last one may overlap the previous (recomputing equal values)
            let mut ch = 0;
            loop {
                let ch0 = ch.min(c - 16);
                let (mut lo, mut hi) = (i16x8_splat(0), i16x8_splat(0));
                for ky in 0..g.kh {
                    let row = (oy * g.sh + ky) * g.in_w + ox * g.sw;
                    for kx in 0..g.kw {
                        // SAFETY: window inside the input (asserted above), channels ch0..ch0+16 <= c
                        let v = unsafe { ld(x.as_ptr().add((row + kx) * c + ch0)) };
                        lo = i16x8_add(lo, i16x8_extend_low_i8x16(v));
                        hi = i16x8_add(hi, i16x8_extend_high_i8x16(v));
                    }
                }
                let q = [
                    quarter(i32x4_extend_low_i16x8(lo)),
                    quarter(i32x4_extend_high_i16x8(lo)),
                    quarter(i32x4_extend_low_i16x8(hi)),
                    quarter(i32x4_extend_high_i16x8(hi)),
                ];
                let r = i8x16_narrow_i16x8(i16x8_narrow_i32x4(q[0], q[1]), i16x8_narrow_i32x4(q[2], q[3]));
                // SAFETY: output channels ch0..ch0+16 of pixel o are inside y (asserted above)
                unsafe { v128_store(y.as_mut_ptr().add(o + ch0).cast::<v128>(), r) };
                if ch0 + 16 >= c {
                    break;
                }
                ch += 16;
            }
        }
    }
}

// ------------------------------------------------------------------- elementwise

/// `acc[i] = sat8(acc[i] + other[i])`.
pub fn add_inplace(acc: &mut [i8], other: &[i8]) {
    let n = acc.len().min(other.len());
    let mut i = 0;
    // SAFETY: i + 16 <= n for every vector access.
    unsafe {
        while i + 16 <= n {
            let p = acc.as_mut_ptr().add(i);
            v128_store(p.cast::<v128>(), i8x16_add_sat(ld(p), ld(other.as_ptr().add(i))));
            i += 16;
        }
    }
    scalar::add_inplace(&mut acc[i..n], &other[i..n]);
}

/// `buf[i] = requant(buf[i], shift)` (RequantizeLinear), computed in i16 lanes.
pub fn requant_inplace(buf: &mut [i8], shift: i8) {
    let s = shift as i32;
    if s >= 8 {
        // (x + 2^(s-1)) >> s == 0 for every int8 x once s >= 8
        buf.fill(0);
        return;
    }
    let one = |v: v128| -> v128 {
        if s > 0 {
            i16x8_shr(i16x8_add(v, i16x8_splat(1 << (s - 1))), s as u32)
        } else {
            // |x << 8| >= 256 for x != 0, so shifting by 8 saturates exactly like any larger shift
            i16x8_shl(v, (-s).min(8) as u32)
        }
    };
    let mut i = 0;
    let n = buf.len();
    // SAFETY: i + 16 <= n for every vector access.
    unsafe {
        while i + 16 <= n {
            let p = buf.as_mut_ptr().add(i);
            let v = ld(p);
            let r = i8x16_narrow_i16x8(one(i16x8_extend_low_i8x16(v)), one(i16x8_extend_high_i8x16(v)));
            v128_store(p.cast::<v128>(), r);
            i += 16;
        }
    }
    scalar::requant_inplace(&mut buf[i..], shift);
}

// ------------------------------------------------------------------ data movement

/// Copy `len >= 16` bytes from `src` to `dst` (non-overlapping) with 16-byte vector moves;
/// when `len` is not a multiple of 16 the last move overlaps the previous one (it rewrites
/// bytes with the values they already hold).
#[inline(always)]
unsafe fn copy_v(src: *const i8, dst: *mut i8, len: usize) {
    let mut i = 0;
    // SAFETY: the caller guarantees len >= 16 readable bytes at src and writable at dst.
    unsafe {
        while i + 16 <= len {
            v128_store(dst.add(i).cast::<v128>(), ld(src.add(i)));
            i += 16;
        }
        if i < len {
            let j = len - 16;
            v128_store(dst.add(j).cast::<v128>(), ld(src.add(j)));
        }
    }
}

/// Nearest Resize by integer factors, bit-exact with [`scalar::resize_nearest`]: each input
/// pixel (>= 16 channels) is loaded once per 16 channels and stored `sw` times, then the row
/// is repeated `sh - 1` times with `copy_within`.
pub fn resize_nearest(g: &ResizeGeom, x: &[i8], y: &mut [i8]) {
    let c = g.c;
    if c < 16 {
        return scalar::resize_nearest(g, x, y);
    }
    let row = g.in_w * g.sw * c;
    assert!(x.len() >= g.in_h * g.in_w * c && y.len() >= g.in_h * g.sh * row);
    for iy in 0..g.in_h {
        let y0 = iy * g.sh * row;
        // SAFETY: input row iy has in_w * c bytes; output row y0 has in_w * sw * c (asserted).
        unsafe {
            let xp = x.as_ptr().add(iy * g.in_w * c);
            let yp = y.as_mut_ptr().add(y0);
            for ix in 0..g.in_w {
                if c == 16 {
                    let v = ld(xp.add(ix * 16));
                    for k in 0..g.sw {
                        v128_store(yp.add((ix * g.sw + k) * 16).cast::<v128>(), v);
                    }
                } else {
                    for k in 0..g.sw {
                        copy_v(xp.add(ix * c), yp.add((ix * g.sw + k) * c), c);
                    }
                }
            }
        }
        for r in 1..g.sh {
            y.copy_within(y0..y0 + row, y0 + r * row);
        }
    }
}

/// One Concat input, bit-exact with [`scalar::concat_copy`].
pub fn concat_copy(x: &[i8], rows: usize, len: usize, y: &mut [i8], offset: usize, stride: usize) {
    if len < 16 || rows == 0 {
        return scalar::concat_copy(x, rows, len, y, offset, stride);
    }
    assert!(x.len() >= rows * len && y.len() >= offset + (rows - 1) * stride + len);
    for r in 0..rows {
        // SAFETY: segment r is inside x and y (asserted above).
        unsafe { copy_v(x.as_ptr().add(r * len), y.as_mut_ptr().add(offset + r * stride), len) };
    }
}
