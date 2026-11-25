//! Float helpers for `no_std` (core has no `sqrt`, `floor` or `exp`). Every function uses
//! only IEEE `+ - * /`, comparisons and integer casts, so it gives the same result on every
//! target: host tests and the browser build compute bit-identical boxes.

/// `floor(x)` (exact).
#[inline]
pub fn floor(x: f64) -> f64 {
    // values beyond 2^52 are integers already (NaN and infinities pass through)
    if x.is_nan() || x.abs() >= 4_503_599_627_370_496.0 {
        return x;
    }
    let t = x as i64 as f64;
    if t > x { t - 1.0 } else { t }
}

/// Round half up: `floor(x + 0.5)`.
#[inline]
pub fn round(x: f64) -> f64 {
    floor(x + 0.5)
}

/// Square root to about 1 ulp: Newton's iteration from a bit-level first guess. Software
/// on every target (core has no stable `sqrt`; wasm's `f64.sqrt` intrinsic is unstable in
/// core), so host and wasm give bit-identical results.
#[inline(never)] // called rarely; keeps the wasm small
pub fn sqrt(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 || x == f64::INFINITY {
        return x;
    }
    // subnormals: scale into the normal range first (sqrt(x * 2^108) = sqrt(x) * 2^54)
    if x < f64::MIN_POSITIVE {
        return sqrt(x * 3.245_185_536_584_267_3e32) / 18_014_398_509_481_984.0;
    }
    // halving the exponent gives a first guess within ~6 %; 6 steps reach full precision
    let mut y = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51));
    for _ in 0..6 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// `e^x` to about 1 ulp (range reduction by ln 2, Taylor series of degree 14).
#[inline(never)] // 256 calls per map (the probability table); keeps the wasm small
pub fn exp(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let x = x.clamp(-700.0, 700.0);
    const LOG2_E: f64 = core::f64::consts::LOG2_E;
    const LN2_HI: f64 = 6.931_471_803_691_238e-1;
    const LN2_LO: f64 = 1.908_214_929_270_587_7e-10;
    let k = round(x * LOG2_E);
    let r = (x - k * LN2_HI) - k * LN2_LO; // |r| <= ~0.347
    let (mut sum, mut term) = (1.0f64, 1.0f64);
    for i in 1..=14 {
        term *= r / i as f64;
        sum += term;
    }
    // 2^k for k in [-1010, 1010] (x is clamped), as two normal factors
    let k = k as i32;
    let (k1, k2) = (k / 2, k - k / 2);
    sum * f64::from_bits(((k1 + 1023) as u64) << 52) * f64::from_bits(((k2 + 1023) as u64) << 52)
}

/// Logistic sigmoid `1 / (1 + e^-x)`.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + exp(-x))
}

/// `2^e` for a tensor exponent.
#[inline]
pub fn pow2(e: i32) -> f64 {
    f64::from_bits(((e.clamp(-1022, 1023) + 1023) as u64) << 52)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_round_sqrt_exp() {
        for &(x, f) in &[(1.5, 1.0), (-1.5, -2.0), (-0.0, 0.0), (2.0, 2.0), (-2.0, -2.0), (1e300, 1e300), (-3.0000001, -4.0)] {
            assert_eq!(floor(x), f, "floor({x})");
        }
        assert_eq!(round(2.5), 3.0);
        assert_eq!(round(-2.5), -2.0);
        for i in 0..20000 {
            let x = i as f64 * 0.731 + 1e-3;
            assert!((sqrt(x) - x.sqrt()).abs() <= f64::EPSILON * x.sqrt(), "sqrt({x})");
        }
        for x in [1e-300, 1e300, 4.9e-324, 2.0, 0.25] {
            assert!((sqrt(x) - x.sqrt()).abs() <= f64::EPSILON * x.sqrt(), "sqrt({x})");
        }
        assert_eq!((sqrt(0.0), sqrt(16.0), sqrt(1e6)), (0.0, 4.0, 1e3));
        for i in -400..400 {
            let x = i as f64 * 0.0731;
            let (a, b) = (exp(x), x.exp());
            assert!((a - b).abs() <= 4.0 * f64::EPSILON * b, "exp({x}) = {a} vs {b}");
        }
        assert_eq!(pow2(-3), 0.125);
        assert!((sigmoid(-0.847_297_860_387_203_9) - 0.3).abs() < 1e-12);
    }
}
