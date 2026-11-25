//! Randomised differential test (cargo feature `selftest`): the SIMD128 kernels against the
//! scalar reference over random geometries (kernel 1..5, stride 1..3, asymmetric padding,
//! odd channel counts, partial output blocks, windows fully inside the padding), random
//! data including -128 everywhere, shifts from -24 to 31 and biases up to 2^30, plus
//! AvgPool (windows up to 18x18, exponents -8..4), Add, RequantizeLinear, nearest Resize
//! (factors 1..5, 1..70 channels) and Concat segments (1..70 bytes, any stride). Exercises
//! code paths the golden models never reach.

/// Run `iters` rounds (each: one conv or depthwise case, one Add + RequantizeLinear case,
/// one AvgPool case, one Resize + Concat case). Returns the number of cases whose outputs differ (0 = all identical),
/// or -1 in a build without simd128. `hxh_selftest_cases` tells how many cases ran.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_selftest(seed: u32, iters: u32) -> i32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        imp::run(seed, iters)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let _ = (seed, iters);
        -1
    }
}

/// Cases compared by all `hxh_selftest` calls so far (degenerate random geometries are
/// skipped and not counted).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_selftest_cases() -> u32 {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        imp::cases()
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        0
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod imp {
    use core::sync::atomic::{AtomicU32, Ordering::Relaxed};

    static CASES: AtomicU32 = AtomicU32::new(0);

    pub fn cases() -> u32 {
        CASES.load(Relaxed)
    }

    /// Count a compared case and pass its result through.
    fn ran(ok: bool) -> bool {
        CASES.fetch_add(1, Relaxed);
        ok
    }

    use hxh_infer::kernels::{scalar, wasm_simd};
    use hxh_infer::{Bias, ConvGeom, PoolGeom, ResizeGeom};

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        /// uniform in lo..=hi
        fn range(&mut self, lo: i64, hi: i64) -> i64 {
            lo + (self.next() % (hi - lo + 1) as u64) as i64
        }
        fn chance(&mut self, pct: u64) -> bool {
            self.next() % 100 < pct
        }
        /// int8 data: uniform, or mostly extreme values
        fn fill_i8(&mut self, buf: &mut [i8]) {
            let mode = self.range(0, 3);
            for v in buf.iter_mut() {
                *v = match mode {
                    0 => -128,
                    1 => [-128, 127, -127, 0][self.range(0, 3) as usize],
                    _ => self.next() as i8,
                };
            }
        }
    }

    const GUARD: usize = 32;
    const SENTINEL: i8 = 0x5a;

    /// Scratch buffers (bump-allocated once per call, never freed).
    struct Bufs {
        x: &'static mut [i8],
        w: &'static mut [i8],
        bias: &'static mut [u8],
        y0: &'static mut [i8],
        y1: &'static mut [i8],
    }

    fn alloc(len: usize) -> &'static mut [i8] {
        let p = crate::hxh_alloc(len);
        assert!(!p.is_null());
        // SAFETY: fresh, zeroed, never-freed memory of `len` bytes, handed out once.
        unsafe { core::slice::from_raw_parts_mut(p.cast::<i8>(), len) }
    }

    pub fn run(seed: u32, iters: u32) -> i32 {
        let mut r = Rng(0x9E37_79B9_7F4A_7C15 ^ seed as u64);
        let bias = alloc(128 * 4);
        let mut b = Bufs {
            x: alloc(64 * 1024),
            w: alloc(256 * 1024),
            // SAFETY: the same fresh memory viewed as bytes
            bias: unsafe { core::slice::from_raw_parts_mut(bias.as_mut_ptr().cast::<u8>(), bias.len()) },
            y0: alloc(64 * 1024),
            y1: alloc(64 * 1024),
        };
        let mut bad = 0;
        for _ in 0..iters {
            bad += !conv_case(&mut r, &mut b) as i32;
            bad += !eltwise_case(&mut r, &mut b) as i32;
            bad += !pool_case(&mut r, &mut b) as i32;
            bad += !move_case(&mut r, &mut b) as i32;
        }
        bad
    }

    fn conv_case(r: &mut Rng, b: &mut Bufs) -> bool {
        let Bufs { x, w, bias, y0, y1 } = b;
        let dw = r.chance(35);
        let pointwise = !dw && r.chance(30);
        let (kh, kw) = if pointwise { (1, 1) } else { (r.range(1, 5) as usize, r.range(1, 5) as usize) };
        let (sh, sw) = if pointwise || r.chance(40) { (1, 1) } else { (r.range(1, 3) as usize, r.range(1, 3) as usize) };
        let in_h = r.range(1, 7) as usize;
        let in_w = r.range(1, 24) as usize;
        let pads = if pointwise {
            [0; 4]
        } else {
            [r.range(0, kh as i64 - 1), r.range(0, kw as i64 - 1), r.range(0, kh as i64 - 1), r.range(0, kw as i64 - 1)]
                .map(|v| v as usize)
        };
        let (hp, wp) = (in_h + pads[0] + pads[2], in_w + pads[1] + pads[3]);
        if hp < kh || wp < kw {
            return true; // no valid output: skipped
        }
        let (out_h, out_w) = ((hp - kh) / sh + 1, (wp - kw) / sw + 1);
        let in_c = if r.chance(20) { r.range(1, 3) } else { r.range(1, 72) } as usize;
        let out_c = if dw { in_c } else { r.range(1, 72) as usize };
        let cout_padded = out_c.div_ceil(16) * 16;
        let cin_w = if dw { 1 } else { in_c };
        let shift = match r.range(0, 9) {
            0 => r.range(-24, -1),
            1 => r.range(20, 31),
            _ => r.range(0, 14),
        } as i8;
        let g = ConvGeom {
            in_h,
            in_w,
            in_c,
            out_h,
            out_w,
            out_c,
            cout_padded,
            kh,
            kw,
            sh,
            sw,
            pad_t: pads[0],
            pad_l: pads[1],
            shift,
            relu: r.chance(50),
        };
        let (nx, nw, ny) = (in_h * in_w * in_c, cout_padded * kh * kw * cin_w, out_h * out_w * out_c);
        if nx > x.len() || nw > w.len() || ny + GUARD > y0.len() {
            return true;
        }
        r.fill_i8(&mut x[..nx]);
        r.fill_i8(&mut w[..nw]);
        // zero the padding output channels of the filter, as the converter does
        for (i, v) in w[..nw].iter_mut().enumerate() {
            if (i / (kh * kw * cin_w * 16)) * 16 + i % 16 >= out_c {
                *v = 0;
            }
        }
        let big = r.chance(15);
        for n in 0..cout_padded {
            let v: i32 = if n >= out_c {
                0
            } else if big {
                r.range(-(1 << 30), 1 << 30) as i32
            } else {
                r.range(-(1 << 20), 1 << 20) as i32
            };
            bias[n * 4..n * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let bias = Bias::new(&bias[..cout_padded * 4]);
        y0[..ny + GUARD].fill(SENTINEL);
        y1[..ny + GUARD].fill(SENTINEL);
        if dw {
            scalar::dwconv2d(&g, &x[..nx], &w[..nw], bias, &mut y0[..ny]);
            wasm_simd::dwconv2d(&g, &x[..nx], &w[..nw], bias, &mut y1[..ny]);
        } else {
            scalar::conv2d(&g, &x[..nx], &w[..nw], bias, &mut y0[..ny]);
            wasm_simd::conv2d(&g, &x[..nx], &w[..nw], bias, &mut y1[..ny]);
        }
        ran(y0[..ny + GUARD] == y1[..ny + GUARD])
    }

    fn eltwise_case(r: &mut Rng, b: &mut Bufs) -> bool {
        let Bufs { x, y0, y1, .. } = b;
        let n = r.range(0, 200) as usize;
        r.fill_i8(&mut x[..n]);
        r.fill_i8(&mut y0[..n]);
        y1[..n].copy_from_slice(&y0[..n]);
        scalar::add_inplace(&mut y0[..n], &x[..n]);
        wasm_simd::add_inplace(&mut y1[..n], &x[..n]);
        let mut ok = y0[..n] == y1[..n];
        let shift = r.range(-24, 31) as i8;
        y1[..n].copy_from_slice(&x[..n]);
        y0[..n].copy_from_slice(&x[..n]);
        scalar::requant_inplace(&mut y0[..n], shift);
        wasm_simd::requant_inplace(&mut y1[..n], shift);
        ok &= y0[..n] == y1[..n];
        ran(ok)
    }

    fn pool_case(r: &mut Rng, b: &mut Bufs) -> bool {
        let Bufs { x, y0, y1, .. } = b;
        // mostly small windows, sometimes big ones (area up to 324: > 256 takes the scalar path)
        let (kh, kw) = if r.chance(15) { (r.range(1, 18) as usize, r.range(1, 18) as usize) } else { (r.range(1, 4) as usize, r.range(1, 4) as usize) };
        let (sh, sw) = (r.range(1, 3) as usize, r.range(1, 3) as usize);
        let (out_h, out_w) = (r.range(1, 4) as usize, r.range(1, 8) as usize);
        let (in_h, in_w) = ((out_h - 1) * sh + kh + r.range(0, 1) as usize, (out_w - 1) * sw + kw + r.range(0, 1) as usize);
        let c = r.range(1, 72) as usize;
        let g = PoolGeom { in_h, in_w, c, out_h, out_w, kh, kw, sh, sw, in_exp: r.range(-8, 4) as i8, out_exp: r.range(-8, 4) as i8 };
        let (nx, ny) = (in_h * in_w * c, out_h * out_w * c);
        if nx > x.len() || ny + GUARD > y0.len() {
            return true;
        }
        r.fill_i8(&mut x[..nx]);
        y0[..ny + GUARD].fill(SENTINEL);
        y1[..ny + GUARD].fill(SENTINEL);
        scalar::avgpool(&g, &x[..nx], &mut y0[..ny]);
        wasm_simd::avgpool(&g, &x[..nx], &mut y1[..ny]);
        ran(y0[..ny + GUARD] == y1[..ny + GUARD])
    }

    fn move_case(r: &mut Rng, b: &mut Bufs) -> bool {
        let Bufs { x, y0, y1, .. } = b;
        // Resize: random factors and channel counts (c < 16 takes the scalar path)
        let g = ResizeGeom {
            in_h: r.range(1, 6) as usize,
            in_w: r.range(1, 9) as usize,
            c: r.range(1, 70) as usize,
            sh: r.range(1, 5) as usize,
            sw: r.range(1, 5) as usize,
        };
        let (nx, ny) = (g.in_h * g.in_w * g.c, g.in_h * g.sh * g.in_w * g.sw * g.c);
        if nx > x.len() || ny + GUARD > y0.len() {
            return true;
        }
        r.fill_i8(&mut x[..nx]);
        y0[..ny + GUARD].fill(SENTINEL);
        y1[..ny + GUARD].fill(SENTINEL);
        scalar::resize_nearest(&g, &x[..nx], &mut y0[..ny]);
        wasm_simd::resize_nearest(&g, &x[..nx], &mut y1[..ny]);
        let mut ok = y0[..ny + GUARD] == y1[..ny + GUARD];
        // Concat: one input segment of `len` bytes per row into rows of `stride` bytes
        let rows = r.range(0, 40) as usize;
        let len = r.range(1, 70) as usize;
        let offset = r.range(0, 20) as usize;
        let stride = offset + len + r.range(0, 30) as usize;
        let ny = if rows == 0 { 0 } else { (rows - 1) * stride + offset + len };
        if rows * len <= x.len() && ny + GUARD <= y0.len() {
            r.fill_i8(&mut x[..rows * len]);
            y0[..ny + GUARD].fill(SENTINEL);
            y1[..ny + GUARD].fill(SENTINEL);
            scalar::concat_copy(&x[..rows * len], rows, len, &mut y0[..ny], offset, stride);
            wasm_simd::concat_copy(&x[..rows * len], rows, len, &mut y1[..ny], offset, stride);
            ok &= y0[..ny + GUARD] == y1[..ny + GUARD];
        }
        ran(ok)
    }
}
