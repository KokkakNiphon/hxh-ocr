//! Bit-exactness against ESP-PPQ's int8 simulation (golden vectors from
//! scripts/espdl_to_hxhm.py), plus robustness checks. Run with `--nocapture` to see the
//! per-model statistics.

mod common;

use std::cell::Cell;

use common::{Golden, MODELS, Round, Stats, load_inter, read, reference};
use hxh_infer::{Bias, ConvGeom, Error, Kernels, Model, OpKind, Scalar, ctc_greedy, kernels::scalar};

#[test]
fn logits_and_ctc_bit_exact_vs_esp_ppq() {
    for &name in MODELS {
        let bytes = read(name, "model.hxhm");
        let m = Model::parse(&bytes).unwrap();
        let g = Golden::load(name);
        let mut arena = vec![0u8; m.arena_size()];
        let mut ids = vec![0u16; g.t];
        let (mut ok_logits, mut ok_ids, mut diff_values) = (0, 0, 0);
        for i in 0..g.n {
            m.set_input_u8(g.image(i), &mut arena).unwrap();
            let logits = m.run(&Scalar, &mut arena).unwrap();
            let want = g.logits(i);
            let d = logits.iter().zip(&want).filter(|(a, b)| a != b).count();
            diff_values += d;
            ok_logits += (d == 0) as usize;
            let n = m.decode(&arena, &mut ids);
            ok_ids += (ids[..n] == g.ids(i)[..]) as usize;
        }
        println!(
            "{name}: logits bit-exact {ok_logits}/{n} vectors ({diff_values} differing values), \
             greedy CTC identical {ok_ids}/{n}",
            n = g.n
        );
        assert_eq!(ok_logits, g.n, "{name}: logits differ");
        assert_eq!(ok_ids, g.n, "{name}: decodes differ");
    }
}

#[test]
fn input_lut_reproduces_ppq_input_quantisation() {
    for &name in MODELS {
        let bytes = read(name, "model.hxhm");
        let m = Model::parse(&bytes).unwrap();
        let g = Golden::load(name);
        let mut arena = vec![0u8; m.arena_size()];
        for i in 0..g.n {
            m.set_input_u8(g.image(i), &mut arena).unwrap();
            assert_eq!(m.input_mut(&mut arena).unwrap().to_vec(), g.input(i), "{name} vector {i}");
        }
        // (p/255 - 0.5)/0.5 * 2^7, half up: 127 -> -0.502 -> -1 and 128 -> +0.502 -> 1, so the
        // int8 input never takes the value 0.
        let lut = m.input_lut().unwrap();
        assert_eq!((lut[0], lut[127], lut[128], lut[255]), (-128, -1, 1, 127), "{name}: pixels 0/127/128/255");
    }
}

#[test]
fn intermediate_tensors_match_esp_ppq() {
    for &name in MODELS {
        let bytes = read(name, "model.hxhm");
        let m = Model::parse(&bytes).unwrap();
        let g = Golden::load(name);
        let inter = load_inter(name);
        let vectors: std::collections::BTreeSet<usize> = inter.iter().map(|r| r.0).collect();
        let mut arena = vec![0u8; m.arena_size()];
        let mut checked = 0;
        for &vi in &vectors {
            m.set_input_u8(g.image(vi), &mut arena).unwrap();
            let mut outs: Vec<(u16, Vec<i8>)> = Vec::new();
            m.run_with_hook(&Scalar, &mut arena, |_, op, y| outs.push((op.out, y.to_vec()))).unwrap();
            for (_, tid, want) in inter.iter().filter(|r| r.0 == vi) {
                let got = &outs.iter().find(|o| o.0 == *tid).expect("tensor produced").1;
                let bad = got.iter().zip(want).filter(|(a, b)| a != b).count();
                assert_eq!(bad, 0, "{name} vector {vi}: tensor t{tid} differs in {bad} of {} values", want.len());
                checked += 1;
            }
        }
        println!("{name}: {checked} intermediate tensors identical");
    }
}

/// The naive reference must agree with the golden data too; its statistics show which
/// semantics the test vectors actually exercise, and what goes wrong with the wrong rounding.
#[test]
fn reference_agrees_and_rounding_mode_matters() {
    for &name in MODELS {
        let bytes = read(name, "model.hxhm");
        let m = Model::parse(&bytes).unwrap();
        let g = Golden::load(name);
        // the naive reference is slow without optimisations: fewer vectors in debug builds
        let n = if cfg!(debug_assertions) { 6.min(g.n) } else { g.n };
        let mut st = Stats::default();
        for i in 0..n {
            assert_eq!(reference(&m, &g.input(i), Round::HalfUp, &mut st), g.logits(i), "{name}: reference vector {i}");
        }
        println!(
            "{name} ({n} vectors): conv outputs {} | exact .5 ties {} (negative {}, on even quotient {}) | saturated {} | \
             max |acc| {} (bound {}, running {}) vs QACC 20-bit limit 524287 | max |bias| {}",
            st.conv_out,
            st.conv_ties,
            st.conv_neg_ties,
            st.conv_even_ties,
            st.conv_sat,
            st.max_abs_acc,
            st.max_abs_bound,
            st.max_abs_running,
            st.max_abs_bias
        );
        println!(
            "{name}: requant outputs {} (ties {}, negative {}) | add outputs {} (saturated {}) | \
             avgpool outputs {} (ties {})",
            st.requant_out, st.requant_ties, st.requant_neg_ties, st.add_out, st.add_sat, st.pool_out, st.pool_ties
        );
        for mode in [Round::HalfEven, Round::HalfAway, Round::Floor] {
            let mut st2 = Stats::default();
            let (mut bad_vec, mut bad_val, mut bad_ids) = (0, 0, 0);
            for i in 0..n {
                let got = reference(&m, &g.input(i), mode, &mut st2);
                let d = got.iter().zip(g.logits(i)).filter(|(a, b)| **a != *b).count();
                bad_val += d;
                bad_vec += (d > 0) as usize;
                if !m.raw_output() {
                    let mut ids = vec![0u16; g.t];
                    let k = ctc_greedy(&got, g.t, g.c, g.c, 0, &mut ids);
                    bad_ids += (ids[..k] != g.ids(i)[..]) as usize;
                }
            }
            println!(
                "{name}: with {mode:?} rounding instead of half-up: {bad_vec}/{n} vectors differ \
                 ({bad_val} logit values), {bad_ids} decodes differ"
            );
        }
    }
}

/// The toy detector (format version 2, raw output) uses the new ops, and decode() is a no-op.
#[test]
fn detector_model_is_v2_raw_output_with_resize_and_concat() {
    let bytes = read("toy_det", "model.hxhm");
    let m = Model::parse(&bytes).unwrap();
    assert_eq!(m.version(), 2);
    assert!(m.raw_output());
    let kinds: Vec<OpKind> = (0..m.n_ops()).map(|i| m.op(i).kind).collect();
    assert!(kinds.iter().filter(|&&k| k == OpKind::Resize).count() >= 4);
    let concat = (0..m.n_ops()).map(|i| m.op(i)).find(|o| o.kind == OpKind::Concat).unwrap();
    assert_eq!((concat.concat_n(), concat.concat_axis()), (6, 3));
    let scales: Vec<(usize, usize)> = (0..m.n_ops()).map(|i| m.op(i)).filter(|o| o.kind == OpKind::Resize).map(|o| (o.sh, o.sw)).collect();
    for s in [(2, 2), (4, 4), (8, 8)] {
        assert!(scales.contains(&s), "resize scale {s:?} not exercised");
    }
    let o = m.output();
    assert_eq!((o.h, o.w, o.c), (64, 64, 2));
    let g = Golden::load("toy_det");
    let mut arena = vec![0u8; m.arena_size()];
    m.set_input_u8(g.image(0), &mut arena).unwrap();
    m.run(&Scalar, &mut arena).unwrap();
    let mut ids = [0u16; 8];
    assert_eq!(m.decode(&arena, &mut ids), 0);
    // a version-2 op inside a file that claims version 1 is rejected
    let mut b = bytes.clone();
    b[4] = 1;
    assert!(Model::parse(&b).is_err());
}

#[test]
fn repeated_runs_are_identical_unless_the_input_is_reused() {
    let g = Golden::load("toy_tqt");
    let bytes = read("toy_tqt", "model.hxhm");
    let m = Model::parse(&bytes).unwrap();
    assert!(m.input_preserved());
    let mut arena = vec![0u8; m.arena_size()];
    m.set_input_u8(g.image(1), &mut arena).unwrap();
    for _ in 0..3 {
        assert_eq!(m.run(&Scalar, &mut arena).unwrap(), &g.logits(1)[..]);
    }
    // converted with --reuse-input: the input region is recycled, a fresh input is needed
    let bytes = read("toy_tqt", "model_reuse_input.hxhm");
    let m = Model::parse(&bytes).unwrap();
    assert!(!m.input_preserved());
    m.set_input_u8(g.image(1), &mut arena).unwrap();
    assert_eq!(m.run(&Scalar, &mut arena).unwrap(), &g.logits(1)[..]);
}

#[test]
fn arena_too_small_is_an_error() {
    let bytes = read("toy_tqt", "model.hxhm");
    let m = Model::parse(&bytes).unwrap();
    let mut arena = vec![0u8; m.arena_size() - 1];
    assert!(matches!(m.run(&Scalar, &mut arena), Err(Error::ArenaTooSmall { .. })));
    let mut arena = vec![0u8; m.arena_size()];
    assert!(matches!(m.set_input_u8(&[0u8; 10], &mut arena), Err(Error::InputSize { .. })));
}

#[test]
fn corrupt_files_are_rejected_or_run_without_panicking() {
    let bytes = read("toy_tqt", "model.hxhm");
    assert_eq!(Model::parse(&bytes[..10]).unwrap_err(), Error::Truncated);
    let mut b = bytes.clone();
    b[0] = b'X';
    assert_eq!(Model::parse(&b).unwrap_err(), Error::BadMagic);
    for cut in (0..bytes.len()).step_by(97) {
        assert!(Model::parse(&bytes[..cut]).is_err(), "truncated at {cut} accepted");
    }
    // Deterministic byte flips in the header and tables: parse must either reject the file
    // or produce a model that runs without panicking (out-of-bounds, overflow, aliasing).
    let tables_end = 64 + 21 * 16 + 20 * 40 + 64;
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut arena = vec![0u8; 1 << 20];
    let (mut rejected, mut ran) = (0, 0);
    let flips = if cfg!(debug_assertions) { 300 } else { 3000 };
    for _ in 0..flips {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        let mut b = bytes.clone();
        let pos = (seed % tables_end as u64) as usize;
        b[pos] ^= 1 << ((seed >> 32) % 8);
        match Model::parse(&b) {
            Err(_) => rejected += 1,
            Ok(m) => {
                if m.arena_size() <= arena.len() {
                    m.run(&Scalar, &mut arena).unwrap();
                    ran += 1;
                }
            }
        }
    }
    println!("byte flips: {rejected} rejected, {ran} accepted and ran");
}

#[test]
fn ctc_greedy_collapses_repeats_drops_blank_and_breaks_ties_low() {
    // T = 6, C = 3: argmaxes 1,1,0,1,2,2 -> [1, 1, 2]; a tie (row 5: 2 vs 2) picks the lower id
    let l: [i8; 18] = [0, 5, 1, 0, 5, 1, 9, 0, 0, 0, 3, 1, 0, 1, 4, 0, 2, 2];
    let mut out = [0u16; 8];
    let n = ctc_greedy(&l, 6, 3, 3, 0, &mut out);
    assert_eq!(&out[..n], &[1, 1, 2, 1]);
    // output buffer too small: truncated, no panic
    let mut small = [0u16; 2];
    assert_eq!(ctc_greedy(&l, 6, 3, 3, 0, &mut small), 2);
    // padded classes beyond num_classes are ignored
    let l2: [i8; 4] = [0, 1, 5, 9];
    let n = ctc_greedy(&l2, 1, 4, 2, 0, &mut out);
    assert_eq!(&out[..n], &[1]);
}

#[test]
fn requant_rounds_half_up_and_saturates() {
    assert_eq!(scalar::requant(3, 1, false), 2); // 1.5 -> 2
    assert_eq!(scalar::requant(-3, 1, false), -1); // -1.5 -> -1 (half up, not away from zero)
    assert_eq!(scalar::requant(5, 1, false), 3); // 2.5 -> 3 (not to even)
    assert_eq!(scalar::requant(-5, 2, false), -1); // -1.25 -> -1
    assert_eq!(scalar::requant(-6, 2, false), -1); // -1.5 -> -1
    assert_eq!(scalar::requant(-7, 2, false), -2); // -1.75 -> -2
    assert_eq!(scalar::requant(1000, 2, false), 127);
    assert_eq!(scalar::requant(-1000, 2, false), -128);
    assert_eq!(scalar::requant(-1000, 2, true), 0);
    assert_eq!(scalar::requant(3, -2, false), 12);
    assert_eq!(scalar::requant(100, -2, false), 127);
    assert_eq!(scalar::round_half_up_f32(-2.5), -2);
    assert_eq!(scalar::round_half_up_f32(2.5), 3);
    assert_eq!(scalar::round_half_up_f32(-0.5), 0);
}

/// A backend that overrides only conv2d is used for regular convs and falls back to the
/// scalar defaults for everything else, with identical results.
#[test]
fn custom_backend_overrides_are_used() {
    struct Counting(Cell<usize>);
    impl Kernels for Counting {
        fn conv2d(&self, g: &ConvGeom, x: &[i8], w: &[i8], b: Bias<'_>, y: &mut [i8]) {
            self.0.set(self.0.get() + 1);
            scalar::conv2d(g, x, w, b, y)
        }
    }
    let bytes = read("toy_tqt", "model.hxhm");
    let m = Model::parse(&bytes).unwrap();
    let g = Golden::load("toy_tqt");
    let mut arena = vec![0u8; m.arena_size()];
    let k = Counting(Cell::new(0));
    m.set_input_u8(g.image(0), &mut arena).unwrap();
    let logits = m.run(&k, &mut arena).unwrap().to_vec();
    assert_eq!(logits, g.logits(0));
    let n_conv = (0..m.n_ops()).filter(|&i| m.op(i).kind == OpKind::Conv).count();
    assert_eq!(k.0.get(), n_conv);
}

/// Opt-in check of any converted model against its golden file, with the same statistics:
///   $env:HXH_MODEL='..\experiments\rust_engine\mnv4s_h32_ptq\model.hxhm'
///   $env:HXH_GOLDEN='..\experiments\rust_engine\mnv4s_h32_ptq\golden.bin'
///   cargo test --release -p hxh-infer --test golden -- --ignored --nocapture external
#[test]
#[ignore]
fn external_model_bit_exact_and_stats() {
    let (Ok(mp), Ok(gp)) = (std::env::var("HXH_MODEL"), std::env::var("HXH_GOLDEN")) else {
        panic!("set HXH_MODEL and HXH_GOLDEN");
    };
    let bytes = std::fs::read(&mp).unwrap();
    let m = Model::parse(&bytes).unwrap();
    let g = Golden::from_bytes(std::fs::read(&gp).unwrap());
    let mut arena = vec![0u8; m.arena_size()];
    let mut st = Stats::default();
    let (mut ok, mut ok_ref) = (0, 0);
    for i in 0..g.n {
        m.set_input_u8(g.image(i), &mut arena).unwrap();
        ok += (m.run(&Scalar, &mut arena).unwrap() == &g.logits(i)[..]) as usize;
        ok_ref += (reference(&m, &g.input(i), Round::HalfUp, &mut st) == g.logits(i)) as usize;
    }
    println!("{mp}: engine bit-exact {ok}/{n}, naive reference {ok_ref}/{n}", n = g.n);
    println!("{st:?}");
    assert_eq!(ok, g.n);
}
