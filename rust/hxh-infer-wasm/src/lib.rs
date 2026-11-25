//! wasm32 wrapper around `hxh-infer` and `hxh-detect`: raw `extern "C"` exports, no
//! wasm-bindgen, no allocator. Memory comes from a bump allocator on `memory.grow` (page
//! granular, so every buffer is 64 KiB aligned) and is never freed: an instance loads one
//! model. A full-page reader instantiates the module twice (detector + recogniser); the
//! `hxh_det_*` exports (letterbox, line detection, crops) are documented in rust/DETECT.md.
//!
//! JS usage (see `test_wasm.mjs`):
//! ```text
//! const p = hxh_alloc(model.length); new Uint8Array(memory.buffer, p, model.length).set(model);
//! if (hxh_load(p, model.length) !== 0) throw ...;     // validates, allocates arena/buffers
//! new Uint8Array(memory.buffer, hxh_image_ptr(), hxh_image_width() * hxh_image_height()).set(gray);
//! const n = hxh_run();                                 // >= 0: number of decoded ids
//! const ids = new Uint16Array(memory.buffer, hxh_ids_ptr(), n);
//! ```
//! Re-read `memory.buffer` after any `hxh_alloc`/`hxh_load`: growing memory detaches old views.
//!
//! Built with `-C target-feature=+simd128`, runs use the SIMD128 kernels
//! (`hxh_infer::Simd128`, bit-exact with the scalar reference); `hxh_set_simd(0)` switches
//! back to the scalar kernels (for A/B checks). Without the feature only scalar exists.
//! The `selftest` cargo feature adds `hxh_selftest`, a randomised scalar-vs-SIMD comparison
//! of the kernels over many geometries (see `test_wasm.mjs --selftest`), and `profile` adds
//! `hxh_profile`, per-op timing with a clock imported from JS (`test_wasm.mjs --profile`).
#![cfg_attr(target_arch = "wasm32", no_std)]

use core::cell::UnsafeCell;
use core::ptr::null_mut;

use hxh_infer::{Error, Model, Scalar};

mod detect;
#[cfg(all(feature = "profile", target_arch = "wasm32"))]
mod profile;
#[cfg(feature = "selftest")]
mod selftest;

#[cfg(target_arch = "wasm32")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}

struct State {
    model: Option<Model<'static>>,
    arena: *mut u8,
    image: *mut u8,
    ids: *mut u16,
    ids_cap: usize,
    /// run with the SIMD128 kernels (only meaningful when compiled with simd128)
    simd: bool,
}

struct Global(UnsafeCell<State>);
// SAFETY: wasm32-unknown-unknown without threads is single-threaded.
unsafe impl Sync for Global {}

/// State of the text-line detection exports (`detect.rs`).
struct Global2(UnsafeCell<detect::Det>);
// SAFETY: as above.
unsafe impl Sync for Global2 {}
static G2: Global2 = Global2(UnsafeCell::new(detect::DET_INIT));

static G: Global = Global(UnsafeCell::new(State {
    model: None,
    arena: null_mut(),
    image: null_mut(),
    ids: null_mut(),
    ids_cap: 0,
    simd: SIMD_BUILD,
}));

/// True when this module was compiled with the wasm `simd128` target feature.
const SIMD_BUILD: bool = cfg!(all(target_arch = "wasm32", target_feature = "simd128"));

#[allow(clippy::mut_from_ref)]
fn st() -> &'static mut State {
    // SAFETY: single-threaded, and no export re-enters another while holding this borrow.
    unsafe { &mut *G.0.get() }
}

fn err_code(e: Error) -> i32 {
    match e {
        Error::Truncated => -10,
        Error::BadMagic => -11,
        Error::BadVersion(_) => -12,
        Error::BadHeader => -13,
        Error::BadTensor(_) => -14,
        Error::BadOp(_) => -15,
        Error::ArenaTooSmall { .. } => -16,
        Error::InputSize { .. } => -17,
        Error::NoInputLut => -18,
    }
}

/// Allocate `len` bytes (64 KiB aligned, zeroed, never freed). Returns 0 on failure.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_alloc(len: usize) -> *mut u8 {
    #[cfg(target_arch = "wasm32")]
    {
        let pages = len.div_ceil(65536).max(1);
        let old = core::arch::wasm32::memory_grow(0, pages);
        if old == usize::MAX { null_mut() } else { (old * 65536) as *mut u8 }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = len;
        null_mut()
    }
}

/// Load a `.hxhm` model that the caller copied to `ptr` (memory from `hxh_alloc`; it must
/// stay untouched afterwards). Returns 0, or a negative error code.
///
/// # Safety
/// `ptr..ptr+len` must be readable for the lifetime of the module instance.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hxh_load(ptr: *const u8, len: usize) -> i32 {
    // SAFETY: guaranteed by the caller.
    let bytes: &'static [u8] = unsafe { core::slice::from_raw_parts(ptr, len) };
    let m = match Model::parse(bytes) {
        Ok(m) => m,
        Err(e) => return err_code(e),
    };
    if m.input().c != 1 || m.input_lut().is_none() {
        return -20; // the u8 grayscale API needs a 1-channel input and an input LUT
    }
    let s = st();
    let o = m.output();
    s.arena = hxh_alloc(m.arena_size());
    s.image = hxh_alloc(m.input().size);
    // a raw-output model (the detector) has no CTC ids
    s.ids_cap = if m.raw_output() { 0 } else { o.h * o.w };
    s.ids = hxh_alloc(s.ids_cap * 2).cast::<u16>();
    if s.arena.is_null() || s.image.is_null() || s.ids.is_null() {
        return -1;
    }
    s.model = Some(m);
    0
}

/// Buffer for the u8 grayscale input image (`hxh_image_height() x hxh_image_width()`).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_image_ptr() -> *mut u8 {
    st().image
}

#[unsafe(no_mangle)]
pub extern "C" fn hxh_image_width() -> u32 {
    st().model.map_or(0, |m| m.input().w as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn hxh_image_height() -> u32 {
    st().model.map_or(0, |m| m.input().h as u32)
}

/// Run on the image buffer. Returns the number of decoded class ids (>= 0; always 0 for a
/// raw-output model such as the detector) or an error (< 0).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_run() -> i32 {
    let s = st();
    let Some(m) = s.model else { return -2 };
    // SAFETY: the buffers were allocated by hxh_load with exactly these sizes.
    let (arena, image, ids) = unsafe {
        (
            core::slice::from_raw_parts_mut(s.arena, m.arena_size()),
            core::slice::from_raw_parts(s.image, m.input().size),
            core::slice::from_raw_parts_mut(s.ids, s.ids_cap),
        )
    };
    if let Err(e) = m.set_input_u8(image, arena) {
        return err_code(e);
    }
    if let Err(e) = run_with_hook(&m, arena, s.simd, |_, _, _| {}) {
        return err_code(e);
    }
    m.decode(arena, ids) as i32
}

/// Run the model with the selected kernels (SIMD128 only exists in a simd128 build).
#[inline(always)]
fn run_with_hook<F: FnMut(usize, &hxh_infer::Op, &[i8])>(
    m: &Model,
    arena: &mut [u8],
    simd: bool,
    hook: F,
) -> Result<(), Error> {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if simd {
        return m.run_with_hook(&hxh_infer::Simd128, arena, hook).map(|_| ());
    }
    let _ = simd;
    m.run_with_hook(&Scalar, arena, hook).map(|_| ())
}

/// Choose the kernels for later runs: nonzero = SIMD128 (the default in a SIMD build),
/// 0 = scalar reference. Returns 1 if SIMD is now active, 0 if not (always 0 in a build
/// without simd128).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_set_simd(on: i32) -> i32 {
    let s = st();
    s.simd = SIMD_BUILD && on != 0;
    s.simd as i32
}

/// Decoded class ids of the last run (u16, blank removed; count = `hxh_run`'s result).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_ids_ptr() -> *const u16 {
    st().ids
}

/// int8 output of the last run: CTC logits `[timesteps][channels]`, or the raw output tensor
/// (NHWC, `hxh_out_height x hxh_out_width x hxh_channels`) of a raw-output model.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_logits_ptr() -> *const i8 {
    let s = st();
    match s.model {
        Some(m) => s.arena.wrapping_add(m.output().offset).cast::<i8>(),
        None => core::ptr::null(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn hxh_timesteps() -> u32 {
    st().model.map_or(0, |m| (m.output().h * m.output().w) as u32)
}

/// Output height and width (NHWC; a recogniser has height 1 and width = timesteps).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_out_height() -> u32 {
    st().model.map_or(0, |m| m.output().h as u32)
}

#[unsafe(no_mangle)]
pub extern "C" fn hxh_out_width() -> u32 {
    st().model.map_or(0, |m| m.output().w as u32)
}

/// 1 if the loaded model has a raw output (detector), 0 for CTC logits.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_raw_output() -> u32 {
    st().model.map_or(0, |m| m.raw_output() as u32)
}

/// Channels per timestep in the logits (may include padding classes).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_channels() -> u32 {
    st().model.map_or(0, |m| m.output().c as u32)
}

/// Classes considered by the decoder (blank = 0).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_num_classes() -> u32 {
    st().model.map_or(0, |m| m.num_classes() as u32)
}

/// Logit value = int8 * 2^exp.
#[unsafe(no_mangle)]
pub extern "C" fn hxh_logits_exp() -> i32 {
    st().model.map_or(0, |m| m.output().exp as i32)
}

/// MACs per inference (f64 to stay out of BigInt land in JS).
#[unsafe(no_mangle)]
pub extern "C" fn hxh_macs() -> f64 {
    st().model.map_or(0.0, |m| m.macs() as f64)
}
