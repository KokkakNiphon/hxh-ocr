//! Host CLI for hxh-infer.
//!
//! ```text
//! hxh-cli <model.hxhm> <image.pgm | golden.bin> [--iters N] [--vocab vocab.txt] [--profile]
//! ```
//! * PGM (binary P5, 8-bit, already the model's input size): prints the decoded class ids
//!   (and text with `--vocab`) and the single-thread time per inference / MAC/s;
//!   `--profile` adds a per-op time table.
//! * golden.bin (from scripts/espdl_to_hxhm.py or experiments/det_engine/golden_det.py): runs
//!   every vector, checks the int8 output and the decoded ids (none for a raw-output detector)
//!   bit-exactly, and reports timing.

use std::process::ExitCode;
use std::time::Instant;

use hxh_infer::{Model, OpKind, Scalar};

fn read_pgm(b: &[u8]) -> Result<(usize, usize, &[u8]), String> {
    // P5 <ws> width <ws> height <ws> maxval <single ws> data, '#' comments allowed
    let mut fields = Vec::new();
    let mut i = 2;
    while fields.len() < 3 {
        while i < b.len() && (b[i].is_ascii_whitespace() || b[i] == b'#') {
            if b[i] == b'#' {
                while i < b.len() && b[i] != b'\n' {
                    i += 1;
                }
            }
            i += 1;
        }
        let s = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        let v: usize = std::str::from_utf8(&b[s..i]).ok().and_then(|t| t.parse().ok()).ok_or("bad PGM header")?;
        fields.push(v);
    }
    let (w, h, maxv) = (fields[0], fields[1], fields[2]);
    if maxv != 255 {
        return Err("only 8-bit PGM is supported".into());
    }
    let data = b.get(i + 1..i + 1 + w * h).ok_or("PGM data truncated")?;
    Ok((w, h, data))
}

struct Golden<'a> {
    n: usize,
    h: usize,
    w: usize,
    t: usize,
    c: usize,
    max_ids: usize,
    rec: usize,
    body: &'a [u8],
}

impl<'a> Golden<'a> {
    fn parse(b: &'a [u8]) -> Result<Self, String> {
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize;
        let n = u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize;
        let (h, w, c, max_ids) = (u16_at(12), u16_at(14), u16_at(18), u16_at(20));
        // T = 0 in the u16 field: the u32 at offset 24 holds it (raw-output goldens)
        let t = if u16_at(16) == 0 { u32::from_le_bytes([b[24], b[25], b[26], b[27]]) as usize } else { u16_at(16) };
        let rec = 2 * h * w + t * c + 2 + 2 * max_ids;
        if b.len() != 32 + n * rec {
            return Err("golden file size mismatch".into());
        }
        Ok(Golden { n, h, w, t, c, max_ids, rec, body: &b[32..] })
    }
    fn image(&self, i: usize) -> &'a [u8] {
        &self.body[i * self.rec..][..self.h * self.w]
    }
    fn logits(&self, i: usize) -> &'a [u8] {
        &self.body[i * self.rec + 2 * self.h * self.w..][..self.t * self.c]
    }
    fn ids(&self, i: usize) -> Vec<u16> {
        let o = i * self.rec + 2 * self.h * self.w + self.t * self.c;
        let n = (u16::from_le_bytes([self.body[o], self.body[o + 1]]) as usize).min(self.max_ids);
        (0..n).map(|k| u16::from_le_bytes([self.body[o + 2 + 2 * k], self.body[o + 3 + 2 * k]])).collect()
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: hxh-cli <model.hxhm> <image.pgm | golden.bin> [--iters N] [--vocab vocab.txt] [--profile]");
        return ExitCode::from(2);
    }
    let mut iters = 200usize;
    let mut vocab: Option<Vec<String>> = None;
    let mut profile = false;
    let mut k = 3;
    while k < args.len() {
        match args[k].as_str() {
            "--iters" => {
                iters = args.get(k + 1).and_then(|s| s.parse().ok()).unwrap_or(iters);
                k += 1;
            }
            "--vocab" => {
                let text = std::fs::read_to_string(&args[k + 1]).expect("read vocab");
                vocab = Some(text.lines().map(str::to_owned).collect());
                k += 1;
            }
            "--profile" => profile = true,
            a => {
                eprintln!("unknown argument {a}");
                return ExitCode::from(2);
            }
        }
        k += 1;
    }

    let model_bytes = std::fs::read(&args[1]).expect("read model");
    let model = match Model::parse(&model_bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("bad model: {e:?}");
            return ExitCode::FAILURE;
        }
    };
    let inp = model.input();
    let out = model.output();
    let mut arena = vec![0u8; model.arena_size()];
    let mut ids = vec![0u16; out.h * out.w];
    println!(
        "model {} (v{}{}): {} B, {} ops, input {}x{}x{} (exp {}), output {}x{} (exp {}), arena {} B, {:.2} M MACs",
        args[1],
        model.version(),
        if model.raw_output() { ", raw output" } else { "" },
        model_bytes.len(),
        model.n_ops(),
        inp.h,
        inp.w,
        inp.c,
        inp.exp,
        out.h * out.w,
        out.c,
        out.exp,
        model.arena_size(),
        model.macs() as f64 / 1e6
    );

    let data = std::fs::read(&args[2]).expect("read input");
    if data.starts_with(b"HXHG") {
        let g = Golden::parse(&data).expect("golden");
        assert_eq!((g.h, g.w, g.t, g.c), (inp.h, inp.w, out.h * out.w, out.c), "golden shape mismatch");
        let (mut ok_logits, mut ok_ids) = (0, 0);
        let mut times = Vec::new();
        for i in 0..g.n {
            model.set_input_u8(g.image(i), &mut arena).unwrap();
            let t0 = Instant::now();
            let logits = model.run(&Scalar, &mut arena).unwrap();
            times.push(t0.elapsed().as_secs_f64());
            if logits.iter().zip(g.logits(i)).all(|(&a, &b)| a == b as i8) {
                ok_logits += 1;
            }
            let n = model.decode(&arena, &mut ids);
            if ids[..n] == g.ids(i)[..] {
                ok_ids += 1;
            }
        }
        let med = median(&mut times);
        println!("golden {}: logits bit-exact {ok_logits}/{n}, CTC ids identical {ok_ids}/{n}", args[2], n = g.n);
        println!(
            "time per inference: median {:.3} ms, min {:.3} ms -> {:.2} GMAC/s (single thread, scalar)",
            med * 1e3,
            times[0] * 1e3,
            model.macs() as f64 / med / 1e9
        );
        return if ok_logits == g.n && ok_ids == g.n { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }

    if !data.starts_with(b"P5") {
        eprintln!("input must be a binary PGM (P5) or a golden.bin");
        return ExitCode::from(2);
    }
    let (w, h, pixels) = read_pgm(&data).expect("pgm");
    if (h, w) != (inp.h, inp.w) || inp.c != 1 {
        eprintln!("image is {w}x{h}, model wants {}x{} (resize first)", inp.w, inp.h);
        return ExitCode::from(2);
    }
    model.set_input_u8(pixels, &mut arena).unwrap();
    let mut times = Vec::with_capacity(iters);
    for it in 0..iters.max(1) + 3 {
        if !model.input_preserved() {
            model.set_input_u8(pixels, &mut arena).unwrap(); // the previous run overwrote it
        }
        let t0 = Instant::now();
        model.run(&Scalar, &mut arena).unwrap();
        if it >= 3 {
            times.push(t0.elapsed().as_secs_f64()); // first 3 runs are warm-up
        }
    }
    let n = model.decode(&arena, &mut ids);
    let ids = &ids[..n];
    println!("ids ({n}): {ids:?}");
    if let Some(v) = &vocab {
        let text: String = ids.iter().map(|&i| v.get(i as usize).map_or("?", |s| s.as_str())).collect();
        println!("text: {text}");
    }
    let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let med = median(&mut times);
    println!(
        "time per inference over {} runs: median {:.3} ms, min {:.3} ms -> {:.2} GMAC/s (single thread, scalar)",
        times.len(),
        med * 1e3,
        min * 1e3,
        model.macs() as f64 / med / 1e9
    );
    if profile {
        print_profile(&model, &mut arena, iters.max(1));
    }
    ExitCode::SUCCESS
}

/// Per-op wall time (hook timestamps between consecutive ops), averaged over `iters` runs.
fn print_profile(model: &Model, arena: &mut [u8], iters: usize) {
    let n = model.n_ops();
    let mut acc = vec![0f64; n];
    for _ in 0..iters {
        let mut last = Instant::now();
        model
            .run_with_hook(&Scalar, arena, |i, _, _| {
                let now = Instant::now();
                acc[i] += (now - last).as_secs_f64();
                last = now;
            })
            .unwrap();
    }
    let total: f64 = acc.iter().sum::<f64>() / iters as f64;
    println!("per-op profile ({iters} runs, total {:.3} ms):", total * 1e3);
    for (i, t) in acc.iter().enumerate() {
        let op = model.op(i);
        let (x, y) = (model.tensor(op.in0 as usize), model.tensor(op.out as usize));
        let t = t / iters as f64;
        let macs = match op.kind {
            OpKind::Conv => (y.size * op.kh * op.kw * x.c) as f64,
            OpKind::DwConv => (y.size * op.kh * op.kw) as f64,
            _ => 0.0,
        };
        let rate = if macs > 0.0 { format!("{:6.2} GMAC/s", macs / t / 1e9) } else { String::new() };
        println!(
            "  op{i:<3} {:8} k{}x{} s{}x{} {:>4}x{:<4}x{:<4} -> {:>4}x{:<4}x{:<4} {:7.1} us {:5.1}% {:>6.2} M MACs {rate}",
            format!("{:?}", op.kind),
            op.kh,
            op.kw,
            op.sh,
            op.sw,
            x.h,
            x.w,
            x.c,
            y.h,
            y.w,
            y.c,
            t * 1e6,
            100.0 * t / total,
            macs / 1e6
        );
    }
}
