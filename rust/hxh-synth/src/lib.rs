//! hxh-synth: synthetic manga / anime-style pages with Hunter-alphabet text lines and
//! line-level ground truth, for training the text-line detector (DBNet-style segmenter).
//!
//! * `glyphs` / `text`: glyph traces and text data embedded from `assets/glyphs.json`
//!   (exported from the Python sources by `scripts/export_glyphs.py`), label normalisation,
//!   text sampler.
//! * `render`: text styles, line layout and stroke primitives (port of `hxh_ocr/render.py`, v4).
//! * `page`: page composition and ground-truth quads (see its module docs); `web`: the
//!   web / document / app-screenshot layout.
//! * `fonts` / `foreign`: system fonts read at run time and the unlabelled foreign-text and
//!   icon distractors (see `FOREIGN.md`).
//! * `targets`: DBNet targets from quads (exact rule in its module docs).
//! * `preview`: contact sheets.
//! * v5 (model family, photo-style pages): `scenes` (name cards, lyric videos, anime frames with
//!   screens, avatars / round icons / toolbar strips as hard negatives) and `photo` (a page
//!   photographed off a screen).
//!
//! Every page is seeded from (seed, split, index) with the in-crate `rng::Rng`, so a set can be
//! regenerated or extended exactly whatever the thread count.

pub mod codec;
pub mod digits;
pub mod fonts;
pub mod foreign;
pub mod geom;
pub mod glyphs;
pub mod page;
pub mod photo;
pub mod preview;
pub mod raster;
pub mod render;
pub mod rng;
pub mod scenes;
pub mod targets;
pub mod text;
pub mod web;
