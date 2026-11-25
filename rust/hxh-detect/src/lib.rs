//! `hxh-detect`: post-processing of the HxH OCR text-line detector, `no_std`, no allocation,
//! no dependencies (it compiles into the same wasm module as `hxh-infer`; the JS API is in
//! `rust/DETECT.md`).
//!
//! The detector (DBNet-lite, run by `hxh-infer`) maps a letterboxed page (long side scaled
//! to the input size, top-left, padded with 255) to an int8 `[h, w, 2]` map at stride 2:
//! channel 0 the text-kernel logit, channel 1 the vertical-orientation logit. From there:
//!
//! 1. [`detect::kernel_threshold`]: `prob > bin_thresh` as an integer comparison on the logit.
//! 2. [`detect::for_each_component`]: 8-connected components (BFS), per component the mean
//!    probability, the vertical-logit sum, and the convex hull of its pixel squares.
//! 3. [`geom::min_area_rect`] (rotating calipers) and [`geom::unclip`] (DBNet ratio, or the
//!    exact inverse of the training shrink).
//! 4. [`detect::detect`]: filters, orientation, quads in original-image pixels (clockwise from
//!    the top-left in reading orientation, the `hxh-synth` convention), and
//!    [`detect::reading_order`] (blocks top to bottom; horizontal lines top to bottom,
//!    vertical columns right to left).
//! 5. [`image::crop`]: an upright, rectified u8 strip per line at native resolution (vertical
//!    lines turned 90 degrees counter-clockwise), and [`image::letterbox`] for the detector
//!    input.
//!
//! ```
//! use hxh_detect::{Frame, Line, MapInfo, Params, Scratch, detect};
//! let (h, w) = (8, 8);
//! let mut map = vec![-100i8; h * w * 2];
//! for x in 1..7 { map[(3 * w + x) * 2] = 100; map[(4 * w + x) * 2] = 100; }
//! let mut buf = vec![0u8; Scratch::bytes_needed(h, w)];
//! let mut s = Scratch::new(&mut buf, h, w).unwrap();
//! let mut lines = [Line::default(); 16];
//! let mi = MapInfo { h, w, c: 2, exp: -4 };
//! let n = detect(&map, &mi, &Frame::identity(w, h), &Params::default(), &mut s, &mut lines);
//! assert_eq!(n, 1);
//! assert!(!lines[0].vertical && lines[0].length > lines[0].thickness);
//! ```
#![cfg_attr(not(test), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod detect;
pub mod geom;
pub mod image;
pub mod math;

pub use detect::{Component, Frame, Line, MapInfo, Params, Scratch, detect, for_each_component, kernel_threshold, reading_order};
pub use geom::{P, RotRect, convex_hull, convex_hull_sorted_yx, min_area_rect, rect_to_quad, unclip};
pub use image::{crop, crop_size, letterbox, letterbox_dims};
