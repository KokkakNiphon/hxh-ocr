# Contributing

Thanks for your interest. Issues and pull requests are welcome.

**Most wanted: real images.** Every number in this project is measured on generated images.
If you have photos or scans of Hunter script that you made yourself, or know of material that
may be shared, open an issue first and describe it. Please do not attach or link screenshots,
scans or crops from the manga or the anime: they are copyrighted, and this repository only
holds synthetic data.

**Bugs and wrong readings.** Open an issue with what you did, what you expected and what you
got. For a misread line, a crop you made yourself (or a generated one) and the expected kana
help most.

**Pull requests.**

- Keep a change focused, and say in the description what it fixes and how you checked it.
- The int8 engine must stay bit-exact with ESP-PPQ. If you touch `rust/hxh-infer`,
  `rust/hxh-detect` or the wasm wrapper, run `cargo test --release --manifest-path rust/Cargo.toml`
  and, after building the wasm, the golden vectors of the toy models in node
  (`node rust/hxh-infer-wasm/test_wasm.mjs web/hxh_infer.wasm <model.hxhm> <golden.bin>` for
  each set in `rust/hxh-infer/tests/data/`), and include the result.
- A change to the generators or the training recipe should come with the numbers it moves,
  measured the way the journal measures them (`docs/research/JOURNAL.md`).
- Do not commit datasets, checkpoints, fonts or anything under `data/` or `runs/`.

By contributing, you agree that your code is licensed under the Apache License 2.0 and any
data under CC BY 4.0, like the rest of the project.
