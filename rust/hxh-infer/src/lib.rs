//! `hxh-infer`: a lean `no_std`, allocation-free int8 inference engine for the HxH OCR
//! line recogniser and text-line detector.
//!
//! * Models are `.hxhm` files (see `rust/FORMAT.md`), produced from ESP-PPQ `.espdl`
//!   exports by `scripts/espdl_to_hxhm.py`. [`Model::parse`] validates the whole file once
//!   and then reads everything zero-copy from the caller's `&[u8]`.
//! * Inference runs into a caller-provided arena (`&mut [u8]`, at least
//!   [`Model::arena_size`] bytes). Tensor offsets inside the arena are precomputed by the
//!   converter, so the engine never allocates.
//! * Integer semantics follow ESP-DL on the ESP32-S3 (and ESP-PPQ's simulation of it):
//!   power-of-two requantisation by an arithmetic shift with round-half-up, int8
//!   saturation, int32 biases at exponent `in + w`, zero padding, 256-entry LUT ops,
//!   RequantizeLinear, saturating Add on equal exponents, AveragePool/ReduceMean with
//!   ESP-DL's float rescale, and (format version 2) nearest Resize by integer factors and
//!   Concat, both pure copies on equal exponents. See [`kernels::scalar`].
//! * Recogniser models end in CTC logits ([`Model::decode`]); raw-output models (e.g. the
//!   text-line detector, [`Model::raw_output`]) leave their int8 output tensor in the arena
//!   for the caller ([`Model::logits`]).
//! * All arithmetic goes through the [`Kernels`] trait. [`Scalar`] is the portable
//!   reference backend. On wasm32 with the `simd128` target feature, `Simd128` (bit-exact
//!   SIMD128 kernels) is also available, and [`Fast`] names the fastest backend in the build.
//!   An ESP32-S3 SIMD backend (TIE728 asm) only needs to override the methods it
//!   accelerates (typically `conv2d` and `dwconv2d`); see `rust/KERNELS.md`.
//!
//! ```no_run
//! # fn demo(model_bytes: &[u8], pixels: &[u8]) -> Result<(), hxh_infer::Error> {
//! use hxh_infer::{Model, Scalar};
//! let model = Model::parse(model_bytes)?;
//! let mut arena = [0u8; 128 * 1024]; // >= model.arena_size()
//! model.set_input_u8(pixels, &mut arena)?; // grayscale, already resized to the input
//! model.run(&Scalar, &mut arena)?;
//! let mut ids = [0u16; 256];
//! let n = model.decode(&arena, &mut ids); // greedy CTC, blank dropped
//! # let _ = &ids[..n]; Ok(()) }
//! ```
#![no_std]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod ctc;
mod engine;
pub mod format;
pub mod kernels;

pub use ctc::ctc_greedy;
pub use format::{Error, Model, Op, OpKind, TensorInfo};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use kernels::wasm_simd::Simd128;
pub use kernels::{Bias, ConvGeom, Kernels, PoolGeom, ResizeGeom, Scalar};

/// The fastest backend compiled into this build: [`Simd128`] on wasm32 with the `simd128`
/// target feature, otherwise [`Scalar`]. Every backend is bit-exact with [`Scalar`].
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub type Fast = Simd128;
/// The fastest backend compiled into this build: `Simd128` on wasm32 with the `simd128`
/// target feature, otherwise [`Scalar`]. Every backend is bit-exact with [`Scalar`].
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
pub type Fast = Scalar;

/// Reinterpret bytes as int8 (same size and alignment, every bit pattern valid).
#[inline]
pub(crate) fn as_i8(b: &[u8]) -> &[i8] {
    // SAFETY: u8 and i8 have identical size/alignment and no invalid bit patterns.
    unsafe { core::slice::from_raw_parts(b.as_ptr().cast::<i8>(), b.len()) }
}

/// Mutable variant of [`as_i8`].
#[inline]
pub(crate) fn as_i8_mut(b: &mut [u8]) -> &mut [i8] {
    // SAFETY: as above; the borrow is carried over unchanged.
    unsafe { core::slice::from_raw_parts_mut(b.as_mut_ptr().cast::<i8>(), b.len()) }
}
