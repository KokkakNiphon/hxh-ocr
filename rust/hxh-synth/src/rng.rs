//! Small self-contained PRNG (SplitMix64-seeded xoshiro256++) plus the distributions the
//! generator needs. Kept in-crate so a (seed, split, index) key always yields the same page,
//! whatever versions of external crates are in use.

pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    /// Seed from a key such as `[seed, split, index]`.
    pub fn from_key(key: &[u64]) -> Self {
        let mut h = 0x243F_6A88_85A3_08D3u64;
        for &k in key {
            let mut t = h ^ k.wrapping_mul(0x9E37_79B9_7F4A_7C15);
            h = splitmix(&mut t);
        }
        let mut x = h;
        let s = [splitmix(&mut x), splitmix(&mut x), splitmix(&mut x), splitmix(&mut x)];
        Rng { s, spare: None }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let r = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    }

    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f64()
    }

    pub fn chance(&mut self, p: f64) -> bool {
        self.f64() < p
    }

    /// Uniform integer in [0, n), n > 0.
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Uniform integer in [lo, hi); lo if the range is empty.
    pub fn int(&mut self, lo: i64, hi: i64) -> i64 {
        if hi <= lo { lo } else { lo + self.below((hi - lo) as usize) as i64 }
    }

    pub fn normal(&mut self, mu: f64, sigma: f64) -> f64 {
        if let Some(z) = self.spare.take() {
            return mu + sigma * z;
        }
        loop {
            let u = 2.0 * self.f64() - 1.0;
            let v = 2.0 * self.f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let m = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * m);
                return mu + sigma * u * m;
            }
        }
    }

    /// Gamma(shape, scale), Marsaglia-Tsang (shape >= 1).
    pub fn gamma(&mut self, shape: f64, scale: f64) -> f64 {
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = self.normal(0.0, 1.0);
            let v = 1.0 + c * x;
            if v <= 0.0 {
                continue;
            }
            let v = v * v * v;
            let u = self.f64();
            if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
                return d * v * scale;
            }
        }
    }

    pub fn poisson(&mut self, lam: f64) -> usize {
        let l = (-lam).exp();
        let (mut k, mut p) = (0, 1.0);
        loop {
            p *= self.f64();
            if p <= l {
                return k;
            }
            k += 1;
        }
    }

    /// Index drawn with probability proportional to `w`.
    pub fn weighted(&mut self, w: &[f64]) -> usize {
        let total: f64 = w.iter().sum();
        let mut u = self.f64() * total;
        for (i, &x) in w.iter().enumerate() {
            if u < x {
                return i;
            }
            u -= x;
        }
        w.len() - 1
    }

    pub fn log_uniform(&mut self, lo: f64, hi: f64) -> f64 {
        if hi <= lo { lo } else { self.uniform(lo.ln(), hi.ln()).exp() }
    }

    pub fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len())]
    }
}
