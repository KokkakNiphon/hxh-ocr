//! Text-line detection exports (`hxh_det_*`): letterbox the page for the detector model,
//! turn its output into lines (`hxh-detect`), and cut rectified line crops. The JS API is
//! documented in `rust/DETECT.md`.
//!
//! Buffers come from `hxh_alloc` and grow on demand (a bigger image or crop allocates a new
//! buffer of twice the size; the old one is not reused). Line records are `[f32; 16]`
//! ([`hxh_detect::Line::record`]).

use core::ptr::null_mut;

use hxh_detect::{Frame, Line, MapInfo, Params, Scratch, crop, crop_size, detect, letterbox};

use crate::{G2, hxh_alloc, st};

/// Most lines one detection returns (the rest are dropped).
pub const MAX_LINES: usize = 1024;
/// f32 per line record.
pub const REC: usize = 16;
/// Largest crop side (a bigger quad is refused).
const MAX_CROP_SIDE: usize = 16384;

pub(crate) struct Det {
    /// the original grayscale page
    img: *mut u8,
    img_cap: usize,
    img_w: usize,
    img_h: usize,
    frame: Option<Frame>,
    params: [f64; 8],
    scratch: *mut u8,
    scratch_cap: usize,
    lb_tmp: *mut f64,
    lb_tmp_cap: usize,
    lines: *mut Line,
    recs: *mut f32,
    n: usize,
    crop: *mut u8,
    crop_cap: usize,
    crop_w: usize,
    crop_h: usize,
    quad: [f32; 8],
}

pub(crate) const DET_INIT: Det = Det {
    img: null_mut(),
    img_cap: 0,
    img_w: 0,
    img_h: 0,
    frame: None,
    // Params::default(): bin_thresh, min_score, min_side, unclip_ratio, shrink_ratio, block_gap
    params: [0.3, 0.5, 3.0, 1.5, 0.0, 1.0, 0.0, 0.0],
    scratch: null_mut(),
    scratch_cap: 0,
    lb_tmp: null_mut(),
    lb_tmp_cap: 0,
    lines: null_mut(),
    recs: null_mut(),
    n: 0,
    crop: null_mut(),
    crop_cap: 0,
    crop_w: 0,
    crop_h: 0,
    quad: [0.0; 8],
};

#[allow(clippy::mut_from_ref)]
fn det() -> &'static mut Det {
    // SAFETY: single-threaded wasm; no export holds this borrow across another export call.
    unsafe { &mut *G2.0.get() }
}

/// Make `*buf` hold at least `need` bytes (doubling; the old buffer is abandoned).
fn ensure(buf: &mut *mut u8, cap: &mut usize, need: usize) -> bool {
    if need <= *cap && !buf.is_null() {
        return true;
    }
    let size = need.max(*cap * 2).max(1);
    let p = hxh_alloc(size);
    if p.is_null() {
        return false;
    }
    *buf = p;
    *cap = size;
    true
}

fn params(d: &Det) -> Params {
    let p = d.params;
    Params { bin_thresh: p[0], min_score: p[1], min_side: p[2], unclip_ratio: p[3], shrink_ratio: p[4], block_gap: p[5] }
}

/// Post-processing parameters, 8 f64 (a `Float64Array`, JS numbers stay exact) the caller
/// may edit before `hxh_det_detect*`:
/// `[bin_thresh 0.3, min_score 0.5, min_side 3 (letterbox px), unclip_ratio 1.5,
/// shrink_ratio 0 (off; > 0 = exact inverse of the training shrink), block_gap 1, -, -]`.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_params_ptr() -> *mut f64 {
    det().params.as_mut_ptr()
}

/// Buffer for the original grayscale page (`w x h` bytes, row-major), to be filled by the
/// caller. Forgets the previous frame and lines. Null on failure.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_image(w: u32, h: u32) -> *mut u8 {
    let d = det();
    let (w, h) = (w as usize, h as usize);
    if w == 0 || h == 0 || !ensure(&mut d.img, &mut d.img_cap, w * h) {
        return null_mut();
    }
    (d.img_w, d.img_h, d.frame, d.n) = (w, h, None, 0);
    d.img
}

/// Letterbox the page (from `hxh_det_image`) into the model's input buffer
/// (`hxh_image_ptr`): long side scaled to the input size (area average when shrinking,
/// bilinear when growing), top-left, padded with 255. Sets the frame used by
/// `hxh_det_detect`. Returns 0, or -2 (no model), -30 (no page), -33 (out of memory).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_letterbox() -> i32 {
    let s = st();
    let Some(m) = s.model else { return -2 };
    let d = det();
    if d.img.is_null() || d.img_w == 0 {
        return -30;
    }
    let (iw, ih) = (m.input().w, m.input().h);
    let mut tmp_bytes = d.lb_tmp.cast::<u8>();
    if !ensure(&mut tmp_bytes, &mut d.lb_tmp_cap, 16 * iw) {
        return -33;
    }
    d.lb_tmp = tmp_bytes.cast::<f64>();
    // SAFETY: img holds img_w * img_h bytes, the image buffer iw * ih (allocated by hxh_load),
    // lb_tmp lb_tmp_cap bytes (64 KiB aligned, so aligned for f64).
    let (src, dst, tmp) = unsafe {
        (
            core::slice::from_raw_parts(d.img, d.img_w * d.img_h),
            core::slice::from_raw_parts_mut(s.image, iw * ih),
            core::slice::from_raw_parts_mut(d.lb_tmp, d.lb_tmp_cap / 8),
        )
    };
    let (cw, ch) = letterbox(src, d.img_w, d.img_h, dst, iw, ih, tmp);
    d.frame = Some(Frame { in_w: iw, in_h: ih, content_w: cw, content_h: ch, orig_w: d.img_w, orig_h: d.img_h });
    0
}

/// Set the frame by hand (for maps not produced via `hxh_det_letterbox`): model input
/// `in_w x in_h`, the resized page inside it `content_w x content_h` (top-left), original
/// page `orig_w x orig_h`. Returns 0 or -34 (a zero size).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_set_frame(in_w: u32, in_h: u32, content_w: u32, content_h: u32, orig_w: u32, orig_h: u32) -> i32 {
    let v = [in_w, in_h, content_w, content_h, orig_w, orig_h].map(|x| x as usize);
    if v.contains(&0) {
        return -34;
    }
    det().frame = Some(Frame { in_w: v[0], in_h: v[1], content_w: v[2], content_h: v[3], orig_w: v[4], orig_h: v[5] });
    0
}

/// Lines from the output of the last `hxh_run` of the loaded detector (a raw-output model,
/// channel 0 kernel logits, channel 1 vertical logits). Uses the frame of
/// `hxh_det_letterbox` / `hxh_det_set_frame`, or the identity of the model input if none is
/// set. Returns the line count (>= 0) or -2 (no model), -33 (out of memory), -35 (not a
/// raw-output model).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_detect() -> i32 {
    let s = st();
    let Some(m) = s.model else { return -2 };
    if !m.raw_output() {
        return -35;
    }
    let o = m.output();
    let frame = det().frame.unwrap_or(Frame::identity(m.input().w, m.input().h));
    // SAFETY: the output tensor lies inside the arena allocated by hxh_load.
    let map = unsafe { core::slice::from_raw_parts(s.arena.add(o.offset).cast::<i8>(), o.size) };
    run_detect(map, MapInfo { h: o.h, w: o.w, c: o.c, exp: o.exp }, frame)
}

/// Lines from any int8 NHWC map `h x w x c` at exponent `exp` in wasm memory (e.g. tests,
/// or a detector run elsewhere). Needs a frame (`hxh_det_letterbox` / `hxh_det_set_frame`).
/// Returns the line count or -32 (no frame), -33, -34 (bad size).
///
/// # Safety
/// `map` must point to `h * w * c` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hxh_det_detect_map(map: *const i8, h: u32, w: u32, c: u32, exp: i32) -> i32 {
    let (h, w, c) = (h as usize, w as usize, c as usize);
    if h == 0 || w == 0 || c == 0 || !(-24..=24).contains(&exp) || map.is_null() {
        return -34;
    }
    let Some(frame) = det().frame else { return -32 };
    // SAFETY: guaranteed by the caller.
    let map = unsafe { core::slice::from_raw_parts(map, h * w * c) };
    run_detect(map, MapInfo { h, w, c, exp: exp as i8 }, frame)
}

fn run_detect(map: &[i8], mi: MapInfo, frame: Frame) -> i32 {
    let d = det();
    d.n = 0;
    if !ensure(&mut d.scratch, &mut d.scratch_cap, Scratch::bytes_needed(mi.h, mi.w)) {
        return -33;
    }
    if d.lines.is_null() {
        let l = hxh_alloc(MAX_LINES * core::mem::size_of::<Line>()).cast::<Line>();
        let r = hxh_alloc(MAX_LINES * REC * 4).cast::<f32>();
        if l.is_null() || r.is_null() {
            return -33;
        }
        for i in 0..MAX_LINES {
            // SAFETY: fresh allocation of MAX_LINES Lines (64 KiB aligned).
            unsafe { l.add(i).write(Line::default()) };
        }
        (d.lines, d.recs) = (l, r);
    }
    // SAFETY: buffers allocated above with these sizes; they do not overlap.
    let (buf, lines, recs) = unsafe {
        (
            core::slice::from_raw_parts_mut(d.scratch, d.scratch_cap),
            core::slice::from_raw_parts_mut(d.lines, MAX_LINES),
            core::slice::from_raw_parts_mut(d.recs, MAX_LINES * REC),
        )
    };
    let Some(mut sc) = Scratch::new(buf, mi.h, mi.w) else { return -33 };
    let n = detect(map, &mi, &frame, &params(d), &mut sc, lines);
    for (l, r) in lines[..n].iter().zip(recs.chunks_exact_mut(REC)) {
        r.copy_from_slice(&l.record());
    }
    d.n = n;
    n as i32
}

/// The line records of the last detection: `count x 16` f32, in reading order (see
/// rust/DETECT.md for the fields).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_lines_ptr() -> *const f32 {
    det().recs
}

/// Number of lines of the last detection.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_line_count() -> u32 {
    det().n as u32
}

fn do_crop(q: [[f64; 2]; 4], ow: usize, oh: usize) -> *const u8 {
    let d = det();
    if d.img.is_null() || d.img_w == 0 || ow == 0 || oh == 0 || ow > MAX_CROP_SIDE || oh > MAX_CROP_SIDE {
        return core::ptr::null();
    }
    if !ensure(&mut d.crop, &mut d.crop_cap, ow * oh) {
        return core::ptr::null();
    }
    // SAFETY: img holds img_w * img_h bytes, crop at least ow * oh.
    let (img, out) =
        unsafe { (core::slice::from_raw_parts(d.img, d.img_w * d.img_h), core::slice::from_raw_parts_mut(d.crop, ow * oh)) };
    crop(img, d.img_w, d.img_h, &q, out, ow, oh);
    (d.crop_w, d.crop_h) = (ow, oh);
    d.crop
}

/// Rectified crop of line `i` of the last detection from the original page: an upright u8
/// strip of `record[14] x record[15]` pixels (= `hxh_det_crop_width/height`), vertical lines
/// turned 90 degrees counter-clockwise. The buffer is reused by the next crop. Null if `i`
/// is out of range or there is no page.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_crop(i: u32) -> *const u8 {
    let d = det();
    let i = i as usize;
    if i >= d.n {
        return core::ptr::null();
    }
    // SAFETY: lines holds MAX_LINES entries, the first n written by the last detection.
    let l = unsafe { *d.lines.add(i) };
    let (w, h) = crop_size(&l.quad);
    do_crop(l.quad, w, h)
}

/// 8 f32 (`x0 y0 x1 y1 x2 y2 x3 y3`, original-page pixels) for `hxh_det_crop_quad`.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_quad_ptr() -> *mut f32 {
    det().quad.as_mut_ptr()
}

/// Rectified crop of the quad at `hxh_det_quad_ptr` (q0 -> q1 = reading direction) into
/// `out_w x out_h` pixels; 0 for either size means native ([`crop_size`]). Null on failure.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_crop_quad(out_w: u32, out_h: u32) -> *const u8 {
    let f = det().quad.map(|v| v as f64);
    let q = [[f[0], f[1]], [f[2], f[3]], [f[4], f[5]], [f[6], f[7]]];
    let (nw, nh) = crop_size(&q);
    let ow = if out_w == 0 { nw } else { out_w as usize };
    let oh = if out_h == 0 { nh } else { out_h as usize };
    do_crop(q, ow, oh)
}

/// Width of the last crop.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_crop_width() -> u32 {
    det().crop_w as u32
}

/// Height of the last crop.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_crop_height() -> u32 {
    det().crop_h as u32
}

/// Width of the page content inside the letterbox (0 before `hxh_det_letterbox` /
/// `hxh_det_set_frame`).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_content_width() -> u32 {
    det().frame.map_or(0, |f| f.content_w as u32)
}

/// Height of the page content inside the letterbox.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_det_content_height() -> u32 {
    det().frame.map_or(0, |f| f.content_h as u32)
}
