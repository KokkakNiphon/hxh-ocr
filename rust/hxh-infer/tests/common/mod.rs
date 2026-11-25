//! Test helpers: golden-file readers and a deliberately naive reference implementation
//! (independent of the optimised kernels) with switchable rounding and statistics.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use hxh_infer::{Model, OpKind};

/// Golden sets in tests/data: three toy recognisers (CTC logits) and a toy text detector
/// (raw output, format version 2: Resize nearest x2/x4/x8 and a 6-input channel Concat).
pub const MODELS: &[&str] = &["toy_tqt", "toy_mean", "toy_random", "toy_det"];

pub fn data_dir(model: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join(model)
}

pub fn read(model: &str, file: &str) -> Vec<u8> {
    let p = data_dir(model).join(file);
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (run scripts/espdl_to_hxhm.py golden)", p.display()))
}

/// golden.bin written by scripts/espdl_to_hxhm.py (see FORMAT.md). `t` x `c` is the output
/// (timesteps x classes, or out_h*out_w x channels for a raw-output model); `max_ids` is the
/// id capacity of a record (`t` for CTC models, 0 for raw-output models).
pub struct Golden {
    pub n: usize,
    pub h: usize,
    pub w: usize,
    pub t: usize,
    pub c: usize,
    pub max_ids: usize,
    rec: usize,
    data: Vec<u8>,
}

impl Golden {
    pub fn load(model: &str) -> Self {
        Self::from_bytes(read(model, "golden.bin"))
    }

    pub fn from_bytes(data: Vec<u8>) -> Self {
        assert_eq!(&data[0..4], b"HXHG");
        let u16_at = |o: usize| u16::from_le_bytes([data[o], data[o + 1]]) as usize;
        let n = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let (h, w, c, max_ids) = (u16_at(12), u16_at(14), u16_at(18), u16_at(20));
        // T is a u16 field; a raw-output golden may store 0 there and the true size in the
        // u32 at 24 (out_h * out_w can exceed 65535)
        let t = if u16_at(16) == 0 { u32::from_le_bytes(data[24..28].try_into().unwrap()) as usize } else { u16_at(16) };
        let rec = 2 * h * w + t * c + 2 + 2 * max_ids;
        assert_eq!(data.len(), 32 + n * rec, "golden.bin size");
        Golden { n, h, w, t, c, max_ids, rec, data }
    }
    fn rec(&self, i: usize) -> &[u8] {
        &self.data[32 + i * self.rec..][..self.rec]
    }
    pub fn image(&self, i: usize) -> &[u8] {
        &self.rec(i)[..self.h * self.w]
    }
    pub fn input(&self, i: usize) -> Vec<i8> {
        self.rec(i)[self.h * self.w..2 * self.h * self.w].iter().map(|&b| b as i8).collect()
    }
    pub fn logits(&self, i: usize) -> Vec<i8> {
        let o = 2 * self.h * self.w;
        self.rec(i)[o..o + self.t * self.c].iter().map(|&b| b as i8).collect()
    }
    pub fn ids(&self, i: usize) -> Vec<u16> {
        let r = self.rec(i);
        let o = 2 * self.h * self.w + self.t * self.c;
        let n = u16::from_le_bytes([r[o], r[o + 1]]) as usize;
        (0..n).map(|k| u16::from_le_bytes([r[o + 2 + 2 * k], r[o + 3 + 2 * k]])).collect()
    }
}

/// golden_inter.bin: (vector, tensor id, int8 NHWC data) records.
pub fn load_inter(model: &str) -> Vec<(usize, u16, Vec<i8>)> {
    let d = read(model, "golden_inter.bin");
    assert_eq!(&d[0..4], b"HXHI");
    let n = u32::from_le_bytes(d[12..16].try_into().unwrap()) as usize;
    let mut o = 16;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let vi = u32::from_le_bytes(d[o..o + 4].try_into().unwrap()) as usize;
        let tid = u16::from_le_bytes([d[o + 4], d[o + 5]]);
        let len = u32::from_le_bytes(d[o + 8..o + 12].try_into().unwrap()) as usize;
        out.push((vi, tid, d[o + 12..o + 12 + len].iter().map(|&b| b as i8).collect()));
        o += 12 + len;
    }
    out
}

// ------------------------------------------------------------------ naive reference

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Round {
    /// floor(v + 0.5): ESP-DL on the S3, ESP-PPQ ROUND_HALF_UP.
    HalfUp,
    /// ties to even: ONNX QuantizeLinear, ESP-DL on the P4.
    HalfEven,
    /// ties away from zero: C roundf().
    HalfAway,
    /// plain arithmetic shift (floor), like DL_RIGHT_SHIFT.
    Floor,
}

#[derive(Default, Debug, Clone)]
pub struct Stats {
    pub conv_out: u64,
    /// requantisations that hit an exact .5 tie
    pub conv_ties: u64,
    /// ties on negative values (half-up and half-away-from-zero disagree there)
    pub conv_neg_ties: u64,
    /// ties on even quotients (half-up and half-to-even disagree there)
    pub conv_even_ties: u64,
    /// results outside [-128, 127] before saturation
    pub conv_sat: u64,
    pub max_abs_acc: i64,
    /// largest |running sum| in the order bias, then taps (ky, kx, c)
    pub max_abs_running: i64,
    /// largest |bias| + sum |x*w| (order-independent worst case for these inputs)
    pub max_abs_bound: i64,
    pub max_abs_bias: i64,
    pub requant_out: u64,
    pub requant_ties: u64,
    pub requant_neg_ties: u64,
    pub add_out: u64,
    pub add_sat: u64,
    pub pool_out: u64,
    pub pool_ties: u64,
}

fn round_shift(acc: i64, s: i32, mode: Round) -> (i64, bool, bool, bool) {
    if s <= 0 {
        return (acc << (-s), false, false, false);
    }
    let q = acc >> s;
    let r = acc - (q << s);
    let half = 1i64 << (s - 1);
    let tie = r == half;
    let up = match mode {
        Round::HalfUp => r >= half,
        Round::HalfEven => r > half || (tie && q & 1 == 1),
        Round::HalfAway => r > half || (tie && q >= 0),
        Round::Floor => false,
    };
    (q + up as i64, tie, tie && q < 0, tie && q & 1 == 0)
}

fn floor_f32(v: f32) -> i64 {
    let t = v as i64;
    if (t as f32) > v { t - 1 } else { t }
}

/// Run the model op by op with straightforward loops; returns the logits.
pub fn reference(m: &Model, input: &[i8], mode: Round, st: &mut Stats) -> Vec<i8> {
    let mut acts: HashMap<u16, Vec<i8>> = HashMap::new();
    let in_id = (0..m.n_tensors()).find(|&i| m.tensor(i) == m.input()).unwrap() as u16;
    acts.insert(in_id, input.to_vec());
    let mut last = 0u16;
    for k in 0..m.n_ops() {
        let op = m.op(k);
        let x = m.tensor(op.in0 as usize);
        let y = m.tensor(op.out as usize);
        let xa = acts[&op.in0].clone();
        let mut ya = vec![0i8; y.size];
        match op.kind {
            OpKind::Conv | OpKind::DwConv => {
                let dw = op.kind == OpKind::DwConv;
                let cin = if dw { 1 } else { x.c };
                let w = m.weights(&op);
                let b = m.bias(&op);
                let wt = |n: usize, ky: usize, kx: usize, c: usize| -> i64 {
                    let blk = n / 16 * op.kh * op.kw * cin * 16;
                    w[blk + ((ky * op.kw + kx) * cin + c) * 16 + n % 16] as i64
                };
                for oy in 0..y.h {
                    for ox in 0..y.w {
                        for n in 0..y.c {
                            let bias = b.get(n) as i64;
                            let mut acc = bias;
                            let mut bound = bias.abs();
                            st.max_abs_bias = st.max_abs_bias.max(bias.abs());
                            for ky in 0..op.kh {
                                let iy = (oy * op.sh + ky) as isize - op.pad[0] as isize;
                                if iy < 0 || iy >= x.h as isize {
                                    continue;
                                }
                                for kx in 0..op.kw {
                                    let ix = (ox * op.sw + kx) as isize - op.pad[1] as isize;
                                    if ix < 0 || ix >= x.w as isize {
                                        continue;
                                    }
                                    let px = (iy as usize * x.w + ix as usize) * x.c;
                                    for c in 0..cin {
                                        let xv = xa[px + if dw { n } else { c }] as i64;
                                        let p = xv * wt(n, ky, kx, c);
                                        acc += p;
                                        bound += p.abs();
                                        st.max_abs_running = st.max_abs_running.max(acc.abs());
                                    }
                                }
                            }
                            st.max_abs_acc = st.max_abs_acc.max(acc.abs());
                            st.max_abs_bound = st.max_abs_bound.max(bound);
                            let (v, tie, neg, even) = round_shift(acc, op.shift as i32, mode);
                            st.conv_out += 1;
                            st.conv_ties += tie as u64;
                            st.conv_neg_ties += neg as u64;
                            st.conv_even_ties += even as u64;
                            st.conv_sat += !(-128..=127).contains(&v) as u64;
                            let lo = if op.relu() { 0 } else { -128 };
                            ya[(oy * y.w + ox) * y.c + n] = v.clamp(lo, 127) as i8;
                        }
                    }
                }
            }
            OpKind::Add => {
                let za = &acts[&op.in1.unwrap()];
                for i in 0..y.size {
                    let s = xa[i] as i32 + za[i] as i32;
                    st.add_out += 1;
                    st.add_sat += !(-128..=127).contains(&s) as u64;
                    ya[i] = s.clamp(-128, 127) as i8;
                }
            }
            OpKind::Lut => {
                let t = m.lut(&op);
                for i in 0..y.size {
                    ya[i] = t[(xa[i] as i32 + 128) as usize];
                }
            }
            OpKind::Requant => {
                for i in 0..y.size {
                    let (v, tie, neg, _) = round_shift(xa[i] as i64, op.shift as i32, mode);
                    st.requant_out += 1;
                    st.requant_ties += tie as u64;
                    st.requant_neg_ties += neg as u64;
                    ya[i] = v.clamp(-128, 127) as i8;
                }
            }
            OpKind::Resize => {
                for oy in 0..y.h {
                    for ox in 0..y.w {
                        for c in 0..y.c {
                            ya[(oy * y.w + ox) * y.c + c] = xa[((oy / op.sh) * x.w + ox / op.sw) * x.c + c];
                        }
                    }
                }
            }
            OpKind::Concat => {
                // gather: output index (h, w, c) -> the input that covers it along the axis
                let axis = op.concat_axis() - 1;
                let ins: Vec<_> = (0..op.concat_n()).map(|i| m.concat_input(&op, i)).collect();
                for oy in 0..y.h {
                    for ox in 0..y.w {
                        for oc in 0..y.c {
                            let mut idx = [oy, ox, oc];
                            let mut src = None;
                            for &id in &ins {
                                let t = m.tensor(id as usize);
                                let d = [t.h, t.w, t.c][axis];
                                if idx[axis] < d {
                                    src = Some((id, t));
                                    break;
                                }
                                idx[axis] -= d;
                            }
                            let (id, t) = src.unwrap();
                            ya[(oy * y.w + ox) * y.c + oc] = acts[&id][(idx[0] * t.w + idx[1]) * t.c + idx[2]];
                        }
                    }
                }
            }
            OpKind::AvgPool => {
                let area = (op.kh * op.kw) as f32;
                let s_in = 2f32.powi(x.exp as i32);
                let s_out = 2f32.powi(-(y.exp as i32));
                let scale = s_in * (1.0 / area) * s_out;
                for oy in 0..y.h {
                    for ox in 0..y.w {
                        for c in 0..y.c {
                            let mut sum = 0i64;
                            for ky in 0..op.kh {
                                for kx in 0..op.kw {
                                    sum += xa[((oy * op.sh + ky) * x.w + ox * op.sw + kx) * x.c + c] as i64;
                                }
                            }
                            let v = sum as f32 * scale;
                            st.pool_out += 1;
                            st.pool_ties += (v - floor_f32(v) as f32 == 0.5) as u64;
                            ya[(oy * y.w + ox) * y.c + c] = floor_f32(v + 0.5).clamp(-128, 127) as i8;
                        }
                    }
                }
            }
        }
        acts.insert(op.out, ya);
        last = op.out;
    }
    acts.remove(&last).unwrap()
}
