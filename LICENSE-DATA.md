# Licence for the datasets

The synthetic datasets of HxH OCR are licensed under the
**Creative Commons Attribution 4.0 International licence (CC BY 4.0)**:
https://creativecommons.org/licenses/by/4.0/ (full legal code:
https://creativecommons.org/licenses/by/4.0/legalcode).

Copyright 2026 KokkakNiphon.

## What this covers

- The line images and labels of `synth_v4` and `lines_v5b` (Hugging Face:
  `KokkakNiphon/hxh-ocr-lines`, configs `synth_v4` and `v5`).
- The page images and line annotations of `pages_v3o`, `pages_v2`, `pages_foreign_open` and
  `pages_v5` (Hugging Face: `KokkakNiphon/hxh-ocr-pages`, configs `v3o`, `v2`,
  `foreign_open` and `v5`).
- Data you generate yourself is not covered by this file. The generators are code, under the
  [Apache License 2.0](LICENSE).

The code and the model weights are **not** under this licence; they are under Apache-2.0.

## You may

share and adapt the datasets for any purpose, including commercially, as long as you give
appropriate credit, link to the licence and say if you changed anything.

Suggested credit:

> HxH OCR synthetic datasets by KokkakNiphon, https://github.com/KokkakNiphon/hxh-ocr,
> licensed under CC BY 4.0.

## Notes

- Every image is generated. None comes from the manga, the anime or any other publication.
- The glyphs are our own vector tracing of the Hunter × Hunter alphabet. Hunter × Hunter and
  its alphabet are © Yoshihiro Togashi / Shueisha. This is an unofficial fan project, not
  affiliated with or endorsed by them, and this licence grants no rights in their work.
- `synth_v4` and `pages_v2` use no fonts. The unlabelled ordinary writing and icons in
  `pages_v3o`, `pages_foreign_open`, `pages_v5` and `lines_v5b` were rasterised only from
  openly licensed fonts (SIL OFL 1.1 and Apache-2.0; files, versions and hashes in
  `rust/hxh-synth/fonts.lock.json`). No font files are distributed. The earlier sets
  `pages_v3` and `pages_foreign`, rendered with Windows system fonts, are not published.
- The CC BY 4.0 licence comes with no warranty. The data is provided as is.
