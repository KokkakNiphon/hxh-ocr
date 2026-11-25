//! Host tool for the detector post-processing (std; the library itself is no_std).
//!
//! ```text
//! cargo run --release -p hxh-detect --example detect -- <command> ...
//!   page <model.hxhm> <image.pgm> [--crops DIR] [params]
//!       full pipeline on one grayscale image: letterbox -> model -> lines (reading order)
//!       -> crops (DIR/line_NN.pgm)
//!   golden <model.hxhm> <golden.bin> <out.txt> [--frame IN CONTENT_W CONTENT_H ORIG_W ORIG_H] [params]
//!       run every golden vector through the model (scalar kernels) and the post-processing;
//!       write the expected lines + crop checksums for hxh-infer-wasm/test_detect.mjs
//!   maps <maps.bin> <out.txt> [params]
//!       components / hulls / min-area rects / lines of raw int8 maps, for
//!       experiments/det_engine/check_postprocess.py (OpenCV comparison)
//!   letterbox <in.pgm> <size> <out.pgm>
//! params: --bin-thresh P --min-score P --min-side PX --unclip R --shrink R --gap G
//! ```
//! Line records are printed like the wasm export (16 numbers, see rust/DETECT.md), as f32.

use std::fmt::Write as _;
use std::process::ExitCode;

use hxh_detect::{Frame, Line, MapInfo, Params, Scratch, crop, crop_size, detect, for_each_component, kernel_threshold, letterbox};
use hxh_infer::{Model, Scalar};

fn read_pgm(b: &[u8]) -> Result<(usize, usize, Vec<u8>), String> {
    if !b.starts_with(b"P5") {
        return Err("not a binary PGM (P5)".into());
    }
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
        fields.push(std::str::from_utf8(&b[s..i]).ok().and_then(|t| t.parse::<usize>().ok()).ok_or("bad PGM header")?);
    }
    let (w, h) = (fields[0], fields[1]);
    let data = b.get(i + 1..i + 1 + w * h).ok_or("PGM data truncated")?;
    Ok((w, h, data.to_vec()))
}

fn write_pgm(path: &str, w: usize, h: usize, px: &[u8]) {
    let mut v = format!("P5\n{w} {h}\n255\n").into_bytes();
    v.extend_from_slice(&px[..w * h]);
    std::fs::write(path, v).expect("write pgm");
}

/// FNV-1a 32-bit (the node test computes the same over the wasm crops).
fn fnv1a(b: &[u8]) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &v in b {
        h = (h ^ v as u32).wrapping_mul(0x0100_0193);
    }
    h
}

/// The 16-number record of the wasm export `hxh_det_lines_ptr`, as f32.
fn record(l: &Line) -> [f32; 16] {
    l.record()
}

fn parse_params(args: &[String]) -> (Params, Vec<String>) {
    let mut p = Params::default();
    let mut rest = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let v = || args.get(i + 1).and_then(|s| s.parse::<f64>().ok()).expect("number after flag");
        match args[i].as_str() {
            "--bin-thresh" => p.bin_thresh = v(),
            "--min-score" => p.min_score = v(),
            "--min-side" => p.min_side = v(),
            "--unclip" => p.unclip_ratio = v(),
            "--shrink" => p.shrink_ratio = v(),
            "--gap" => p.block_gap = v(),
            _ => {
                rest.push(args[i].clone());
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    (p, rest)
}

struct Runner {
    bytes: Vec<u8>,
}

impl Runner {
    fn model(&self) -> Model<'_> {
        Model::parse(&self.bytes).expect("parse model")
    }
}

fn map_info(m: &Model) -> MapInfo {
    let o = m.output();
    MapInfo { h: o.h, w: o.w, c: o.c, exp: o.exp }
}

fn cmd_page(args: &[String], p: &Params) -> ExitCode {
    let r = Runner { bytes: std::fs::read(&args[0]).expect("read model") };
    let m = r.model();
    let (w, h, img) = read_pgm(&std::fs::read(&args[1]).expect("read image")).expect("pgm");
    let crops = args.iter().position(|a| a == "--crops").map(|i| args[i + 1].clone());
    let inp = m.input();
    let mut lb = vec![0u8; inp.h * inp.w];
    let mut tmp = vec![0f64; 2 * inp.w];
    let t0 = std::time::Instant::now();
    let (cw, ch) = letterbox(&img, w, h, &mut lb, inp.w, inp.h, &mut tmp);
    let t1 = std::time::Instant::now();
    let mut arena = vec![0u8; m.arena_size()];
    m.set_input_u8(&lb, &mut arena).unwrap();
    m.run(&Scalar, &mut arena).unwrap();
    let t2 = std::time::Instant::now();
    let mi = map_info(&m);
    let fr = Frame { in_w: inp.w, in_h: inp.h, content_w: cw, content_h: ch, orig_w: w, orig_h: h };
    let mut buf = vec![0u8; Scratch::bytes_needed(mi.h, mi.w)];
    let mut s = Scratch::new(&mut buf, mi.h, mi.w).unwrap();
    let mut lines = vec![Line::default(); 1024];
    let n = detect(m.logits(&arena), &mi, &fr, p, &mut s, &mut lines);
    let t3 = std::time::Instant::now();
    println!(
        "{w}x{h} -> letterbox {}x{} (content {cw}x{ch}); {n} lines; letterbox {:.1} ms, model {:.1} ms (scalar), post-processing {:.2} ms",
        inp.w,
        inp.h,
        (t1 - t0).as_secs_f64() * 1e3,
        (t2 - t1).as_secs_f64() * 1e3,
        (t3 - t2).as_secs_f64() * 1e3
    );
    if let Some(d) = &crops {
        std::fs::create_dir_all(d).unwrap();
        write_pgm(&format!("{d}/letterbox.pgm"), inp.w, inp.h, &lb);
    }
    for (i, l) in lines[..n].iter().enumerate() {
        let (cw, ch) = crop_size(&l.quad);
        println!(
            "  #{i:<3} block {:<2} {} score {:.3} {:>5.1} x {:<5.1} quad {:?}",
            l.block,
            if l.vertical { "v" } else { "h" },
            l.score,
            l.length,
            l.thickness,
            l.quad.map(|p| [(p[0] * 10.0).round() / 10.0, (p[1] * 10.0).round() / 10.0])
        );
        if let Some(d) = &crops {
            let mut out = vec![0u8; cw * ch];
            crop(&img, w, h, &l.quad, &mut out, cw, ch);
            write_pgm(&format!("{d}/line_{i:02}.pgm"), cw, ch, &out);
        }
    }
    ExitCode::SUCCESS
}

fn cmd_golden(args: &[String], p: &Params) -> ExitCode {
    let r = Runner { bytes: std::fs::read(&args[0]).expect("read model") };
    let m = r.model();
    let g = std::fs::read(&args[1]).expect("read golden");
    let u16_at = |o: usize| u16::from_le_bytes([g[o], g[o + 1]]) as usize;
    let n_vec = u32::from_le_bytes(g[8..12].try_into().unwrap()) as usize;
    let (h, w, c, max_ids) = (u16_at(12), u16_at(14), u16_at(18), u16_at(20));
    let t = if u16_at(16) == 0 { u32::from_le_bytes(g[24..28].try_into().unwrap()) as usize } else { u16_at(16) };
    let rec = 2 * h * w + t * c + 2 + 2 * max_ids;
    let inp = m.input();
    assert_eq!((inp.h, inp.w), (h, w), "golden does not match the model");
    let fr = match args.iter().position(|a| a == "--frame") {
        Some(i) => {
            let v: Vec<usize> = args[i + 1..i + 6].iter().map(|s| s.parse().unwrap()).collect();
            Frame { in_w: v[0], in_h: v[0], content_w: v[1], content_h: v[2], orig_w: v[3], orig_h: v[4] }
        }
        None => Frame::identity(w, h),
    };
    let mi = map_info(&m);
    let mut arena = vec![0u8; m.arena_size()];
    let mut buf = vec![0u8; Scratch::bytes_needed(mi.h, mi.w)];
    let mut lines = vec![Line::default(); 1024];
    let mut txt = String::new();
    writeln!(
        txt,
        "# hxh-detect golden lines: params {:?} frame {:?}\n# per vector: 'vector i n'; per line: 16 f32 (wasm record) + FNV-1a of its crop from the golden image",
        p, fr
    )
    .unwrap();
    let mut total = 0;
    for i in 0..n_vec {
        let img = &g[32 + i * rec..32 + i * rec + h * w];
        m.set_input_u8(img, &mut arena).unwrap();
        let out = m.run(&Scalar, &mut arena).unwrap();
        assert_eq!(out, crate_as_i8(&g[32 + i * rec + 2 * h * w..][..t * c]), "vector {i}: output differs from golden");
        let mut s = Scratch::new(&mut buf, mi.h, mi.w).unwrap();
        let n = detect(m.logits(&arena), &mi, &fr, p, &mut s, &mut lines);
        total += n;
        writeln!(txt, "vector {i} {n}").unwrap();
        for l in &lines[..n] {
            let (cw, ch) = crop_size(&l.quad);
            let mut cr = vec![0u8; cw * ch];
            // the golden image is the letterbox; with the identity frame it is the original too
            crop(img, w, h, &l.quad, &mut cr, cw, ch);
            let rec = record(l);
            let nums: Vec<String> = rec.iter().map(|v| format!("{v:?}")).collect();
            writeln!(txt, "{} {:08x}", nums.join(" "), fnv1a(&cr)).unwrap();
        }
    }
    std::fs::write(&args[2], txt).expect("write");
    println!("{n_vec} vectors, {total} lines -> {}", args[2]);
    ExitCode::SUCCESS
}

fn crate_as_i8(b: &[u8]) -> &[i8] {
    // SAFETY: same size and alignment, every bit pattern valid
    unsafe { std::slice::from_raw_parts(b.as_ptr().cast::<i8>(), b.len()) }
}

fn cmd_maps(args: &[String], p: &Params) -> ExitCode {
    let d = std::fs::read(&args[0]).expect("read maps");
    let n = u32::from_le_bytes(d[0..4].try_into().unwrap()) as usize;
    let mut o = 4;
    let mut txt = String::new();
    for k in 0..n {
        let u16_at = |o: usize| u16::from_le_bytes([d[o], d[o + 1]]) as usize;
        let (h, w, c, exp) = (u16_at(o), u16_at(o + 2), u16_at(o + 4), d[o + 6] as i8);
        o += 8;
        let map = crate_as_i8(&d[o..o + h * w * c]).to_vec();
        o += h * w * c;
        let mi = MapInfo { h, w, c, exp };
        let mut buf = vec![0u8; Scratch::bytes_needed(h, w)];
        let mut s = Scratch::new(&mut buf, h, w).unwrap();
        let q_thr = kernel_threshold(exp, p.bin_thresh);
        writeln!(txt, "map {k} {h} {w} {c} {exp} thr {q_thr}").unwrap();
        for_each_component(&map, &mi, q_thr, &mut s, |cp| {
            let r = cp.rect;
            write!(
                txt,
                "comp {} {} {} {:.17e} {} {} {} {} rect {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} {:.17e} hull {}",
                cp.first, cp.pixels, cp.vert_sum, cp.prob_sum, cp.bbox[0], cp.bbox[1], cp.bbox[2], cp.bbox[3], r.c[0], r.c[1], r.u[0], r.u[1], r.w, r.h,
                cp.hull.len()
            )
            .unwrap();
            for q in cp.hull {
                write!(txt, " {} {}", q[0], q[1]).unwrap();
            }
            txt.push('\n');
        });
        let mut lines = vec![Line::default(); 4096];
        let fr = Frame::identity(w, h);
        let nl = detect(&map, &mi, &fr, p, &mut s, &mut lines);
        for l in &lines[..nl] {
            let nums: Vec<String> = record(l).iter().map(|v| format!("{v:?}")).collect();
            writeln!(txt, "line {}", nums.join(" ")).unwrap();
        }
    }
    std::fs::write(&args[1], txt).expect("write");
    ExitCode::SUCCESS
}

fn cmd_letterbox(args: &[String]) -> ExitCode {
    let (w, h, img) = read_pgm(&std::fs::read(&args[0]).expect("read image")).expect("pgm");
    let size: usize = args[1].parse().unwrap();
    let mut dst = vec![0u8; size * size];
    let mut tmp = vec![0f64; 2 * size];
    let (nw, nh) = letterbox(&img, w, h, &mut dst, size, size, &mut tmp);
    write_pgm(&args[2], size, size, &dst);
    println!("{nw} {nh}");
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (p, rest) = parse_params(&args);
    match rest.first().map(String::as_str) {
        Some("page") if rest.len() >= 3 => cmd_page(&rest[1..], &p),
        Some("golden") if rest.len() >= 4 => cmd_golden(&rest[1..], &p),
        Some("maps") if rest.len() >= 3 => cmd_maps(&rest[1..], &p),
        Some("letterbox") if rest.len() >= 4 => cmd_letterbox(&rest[1..]),
        _ => {
            eprintln!("usage: detect (page <model> <img.pgm> [--crops DIR] | golden <model> <golden.bin> <out.txt> [--frame ..] | maps <in> <out> | letterbox <in.pgm> <size> <out.pgm>) [params]");
            ExitCode::from(2)
        }
    }
}
