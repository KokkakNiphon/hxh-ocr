# Research journal: Hunter x Hunter alphabet OCR

## 2026-11-24: glyph set and synthetic line generator

**Goal.** A tiny OCR model (target: well under 1 MB of weights) that reads the
Hunter x Hunter alphabet in a mobile browser. Step one is a synthetic training set.

**What the script is.** A one-to-one substitution cipher for Japanese kana
(Hunterpedia; Omniglot). 46 kana + ー = 47 base glyphs. Voiced kana add a small
filled dot at the lower right of the base glyph; p-kana add a small hollow ring
there. Katakana and hiragana share glyphs. Words may be separated by spaces
(Omniglot's UDHR sample).

**Glyph sources.**
- Hunterpedia chart `File:Hunterjap.png` (966x526). All 47 base glyphs were traced
  by hand into stroke primitives (lines, arcs, ellipses, dots, splines) from 4-5x
  zoomed crops, then rendered back and compared side by side
  (`previews/glyph_sheet.png`). Voiced forms were checked against the chart's own
  voiced rows: dakuten = small dot, handakuten = small ring, both bottom-right.
- Omniglot chart (omniglot.com/conscripts/hunter.htm, drawn from a fan font).
  Agrees with the trace on every glyph except も: Omniglot ends it in a hollow
  ring, Hunterpedia in a filled dot. Both forms are generated (50/50). Omniglot
  also draws small ゃゅょ as smaller glyphs, so small kana are rendered at
  0.55-0.78 scale on ~65% of lines; labels always use the full-size kana.
- Not used: the fan TTF linked from Omniglot (third-party download site) and the
  web converters. A real font would be a useful extra style if obtained from a
  trustworthy source.

**Decisions.**
- Vocabulary = 73 kana + space = 74 tokens (CTC adds a blank). Small kana and
  katakana are folded into full-size hiragana in labels, because the glyphs
  cannot tell them apart reliably.
- Relative glyph sizes are kept as in the chart (い is a lone dot, て is short);
  normalising every glyph to the em box would make い a huge disc and blur the
  い vs dakuten distinction.
- Rendering is vector-based, so styles are cheap to vary: clean (font-like),
  hand (wobble, stroke jitter, rotation, scale and aspect jitter) and brush
  (pressure-tapered stamping); stroke width 0.04-0.24 em.
- Layout: horizontal and vertical (30%), monospaced cells or proportional
  spacing, occasional italic shear and curved baselines, neighbouring lines
  peeking in at the crop edges (15%).
- Backgrounds: flat, gradient, value noise, paper, manga screentone, hatching,
  random shapes, optional real photos (`--bg-dir`). Effects: outline, drop
  shadow, patchy ink, gradient ink. Degradations: blur, motion blur,
  down/up-scaling, uneven light, contrast/gamma, noise, binarisation, grayscale,
  JPEG.
- Output: line crops 64 px thick (height for horizontal lines, width for
  vertical ones), so training can choose 32/40/48 px by downscaling.

**Numbers.** First run: about 180 samples/s with 14 workers. Profiling showed
Python-level brush stamping, join circles and a looped densify; after
vectorising and pinning BLAS to one thread per worker, throughput reached
572 samples/s (train, 100k in 175 s) and 544/s (val). Dataset `data/synth_v1`:
100k train + 5k val lines, seed 0, 64 px thick.

**Open questions.**
- Sim-to-real gap: needs a small labelled set of real crops (manga panels,
  anime frames, merch photos) to measure.
- Which も form does the manga use? Check real panels.
- Vertical text in the manga: confirm whether dakuten stay bottom-right.
- Punctuation (。、！？) is not modelled yet.

**How the day went (blog notes).**
- The glyph trace came first, from one chart. Partway through, the question
  "have you checked other sources, e.g. Omniglot?" led to the cross-check. It
  confirmed 46 of the 47 base glyphs, found the も disagreement, and surfaced two
  features the generator lacked (smaller small-kana, word spaces). Lesson: get a
  second reference before investing in a trace.
- The chart was read in the browser by drawing crops onto a canvas, then
  segmenting rows and columns from pixel profiles to get one zoomed cell per
  glyph. Coordinates were read off those cells.
- The first preview showed stray "clutter" lines striking straight through the
  text. Fix: 75% of clutter now sits in the margins, parallel to the box edge.
- The first dataset run was 3x slower than the single-core estimate suggested.
  cProfile pointed at Python loops, not image ops.
- A label-length bug (random spaces pushed lines past `max_len`) was caught by
  a 50k-sample sanity check of the text sampler.
- Moving the project out of the app's temporary folder: the app refused to
  copy 105k files, so robocopy did it (exit code 1 means "copied OK", not a
  failure). A labels-vs-files check confirmed nothing was missing.

## 2026-11-24: how to train and deploy (research)

**Constraints from the owner.** Use ONNX where possible. Must run in a mobile
browser. Train on this PC (RTX 4080 SUPER 16 GB, Windows 11). Don't start from
scratch, and use current frameworks and architectures. The end goal is
inference on an ESP32-S3, so size and speed matter. Question: can the inference
engine be lean Rust?

### What the state of the art says about the recogniser

- **CTC is the right decoder for this.** SVTRv2 (ICCV 2025, "CTC Beats
  Encoder-Decoder Models in Scene Text Recognition") shows a CTC recogniser can
  match attention decoders on accuracy while being faster. CTC also decodes
  with an argmax and a collapse, which is trivial on a microcontroller. No
  autoregressive decoder needed.
- **PP-OCRv5** (CVPR 2026) is a 5M-parameter detection+recognition system that
  is CTC-based. ONNX exports of the mobile recogniser are on Hugging Face
  (~7.5 MB fp32 for single-script variants). Espressif ships **PP-OCRv6 on
  ESP-DL** (`pp_ocr_v6` example) for the ESP32-P4: about 78 ms per int8
  recognition crop on the P4. It is not listed for the S3. This is an existence
  proof that a CTC recogniser deploys through ESP-DL, but these models are far
  bigger than our 74-token task needs.
- **OpenOCR** (PyTorch, SVTRv2 / RepSVTR, ONNX export since March 2025) is the
  current PyTorch home for these recognisers. It could supply a teacher model.
- **Pretrained tiny backbone:** timm `mobilenetv4_conv_small_050.e3000_r224_in1k`
  (2.2M params including the 1000-class head, Apache-2.0). MobileNetV4's
  conv-only variants use plain convs, depthwise convs and ReLU, with no SE
  blocks and no hard-swish. That suits int8 hardware.

### ESP32-S3 facts that shape the model

- **ESP-DL v3.3.14** (latest; release notes mention a PIE kernel rework that
  makes MobileNetV2/YOLO11 over 3x faster on the S3). 68 ONNX operators,
  including Conv (groups must be 1 or C_in, so depthwise is fine), Add, Mul,
  Concat, pools, ReduceMean, Reshape, Transpose, Gemm/MatMul, Softmax,
  LayerNorm (last axis only), LSTM, GRU and Attention. Layout is NHWC.
  BatchNormalization is not listed, so it must be folded into the convs.
- **Quantisation on ESP32-series chips:** `.espdl` models must use per-tensor,
  symmetric, **power-of-two** scales (ESP-PPQ TQT tutorial). Requantisation is
  therefore a shift, which keeps a custom engine simple. The S3 rounds half
  up. ESP-PPQ (built on PPQ) does PTQ from ONNX or PyTorch. Its TQT pass
  (esp-ppq ≥ 1.2.7) fine-tunes scales and weights with unlabelled
  calibration data. Per-tensor weight scales are a known accuracy risk for
  depthwise convs, so plan for TQT or QAT.
- **Measured speed (int8, 240 MHz, octal PSRAM, from Espressif's tables):**

  | Runtime | Model | Input | S3 latency | ≈ throughput |
  |---|---|---|---|---|
  | ESP-DL | MobileNetV2 (~300M MACs) | 224² | 437.9 ms (1 core), 389.7 ms (2) | ~0.7 GMAC/s |
  | ESP-DL | espdet_pico (0.36M params, 0.17 GFLOPs) | 224² | 120-126 ms | ~0.7 GMAC/s if FLOPs = 2×MACs |
  | TFLM + ESP-NN | person detection (MobileNetV1-0.25) | 96² | 54 ms (47 ms in SRAM) | ~0.15 GMAC/s |
  | TFLM + ESP-NN | MobileNetV3-small | 224² | 729 ms | ~0.08 GMAC/s |

  The ESP-NN kernel table shows optimised int8 kernels are 4.5x to 14x faster
  than plain C on the S3 (depthwise 5x5: 4.49x; 1x1 conv: 8-14x). So scalar
  kernels, in C or Rust alike, are roughly an order of magnitude slower than
  SIMD ones. The MAC counts in the table are approximate figures from the
  model literature; the throughput column is derived, not measured.
- **Budget we can set:** about 20-40M MACs per 32x256 line gives roughly
  30-60 ms per line with ESP-DL-class kernels. A 0.2-0.5M-parameter model is
  0.2-0.5 MB int8. At 32 px line height every activation fits in the 512 KB
  internal SRAM. The zoo models above run out of PSRAM, so a small model
  should beat their per-MAC speed.

### Can the inference engine be lean Rust?

Yes. The landscape in November 2026:

| Option | Format | S3 | WASM | Status |
|---|---|---|---|---|
| edgedl | `.espdl` (ESP-PPQ output), parsed at compile time | no_std, TIE728 SIMD for 1x1/3x3 conv (channels a multiple of 16); depthwise, linear and pools scalar | not stated | 5 commits, last May 2026, 3 stars |
| nanoglia 0.3.1 | `.tflite` at runtime | ESP-NN backend, not yet run on hardware | listed | 2 weeks old |
| ember-rs + esp_nn_sys 0.2.1 | `.tflite` at compile time | ESP-NN via FFI | no | prototype, 7 ops |
| MicroFlow 0.1.3 | `.tflite` at compile time | tested on ESP32 (scalar) | no | stale since 2024 |
| Burn 0.22 | ONNX → generated Rust | no_std only on the Flex backend; no Xtensa evidence | yes | active |
| tract 0.23 | ONNX/TFLite/NNEF | needs std | yes | active, production at Sonos |

- Rust on Xtensa still uses the espup-installed `esp` toolchain (a Tier 3
  target with Espressif's LLVM fork). Xtensa `asm!` support landed in the fork,
  so hand-written SIMD kernels are possible.
- **Conclusion:** a small purpose-built engine is realistic, because our model
  needs about six ops: conv, depthwise conv, ReLU, add, pooling, and a 1x1
  classifier, plus CTC greedy decode. With ESP-DL's power-of-two per-tensor
  scheme, requantisation is a rounding shift, and the same Rust code can be
  bit-exact on host, WASM and the S3. Speed on the S3 depends on the kernels,
  not on the language: scalar Rust is about as slow as plain C, ESP-NN via FFI
  is TFLM-class, and ESP-DL's PIE kernels are the fastest measured. edgedl
  proves `.espdl` can be read from no_std Rust and that TIE728 asm kernels
  can be written in Rust.

### Browser

onnxruntime-web is at 1.30 (WASM SIMD and WebGPU backends), which gives the
quickest demo straight from the ONNX file. A tiny custom Rust engine compiled
to WASM would be a far smaller download and would match the device's int8
output exactly.

### Decisions (made with the owner, 2026-11-24)

1. **Inference engine: lean Rust plus fast kernels.** One no_std Rust int8 engine
   serves both the browser (wasm32) and the ESP32-S3. On the S3, the heavy convs
   call SIMD kernels (Espressif's, via FFI, or our own TIE728 asm). ESP-DL C++
   is used only as the speed yardstick. The model artifact chain is
   PyTorch → ONNX → ESP-PPQ (power-of-two int8).
2. **Horizontal lines first.** Vertical text comes later with an orientation
   flag. Rotating a vertical strip 90° turns ふ (⊃) into something that looks
   exactly like う (∩), so the model must be told the orientation.
3. **Teacher only if needed.** Start with the ImageNet-pretrained MobileNetV4
   student and a from-scratch ablation. Fine-tune PP-OCRv5 or OpenOCR SVTRv2 as
   a distillation teacher only if accuracy on real images lags.
4. **Training environment:** project `.venv` (inherits global site-packages)
   plus PyTorch 2.15.0+cu132 and torchvision 0.29.2+cu132, the newest builds
   on download.pytorch.org (driver 610.88 supports CUDA 13). PyPI's default
   Windows wheel is CPU-only.

### Running in parallel (2026-11-24)
- Agent A: PyTorch training pipeline (data loader, MobileNetV4 and tiny
  models, CTC training, eval, BN-folded ONNX export, plots).
- Agent B: ESP-PPQ spike, quantising a toy model with exactly our ops for
  esp32s3 and measuring the int8 error.
- Background: `data/synth_v2`, 500k train + 10k val grayscale lines (seed 1,
  JPEG q90), at about 616 lines/s alongside the agents.
- Background: CUDA PyTorch wheel download (installed after agent A finishes,
  so its CPU smoke test isn't disturbed).

### Sources
- SVTRv2: https://arxiv.org/abs/2411.15858 · OpenOCR: https://github.com/topdu/OpenOCR
- PP-OCRv5 paper: https://arxiv.org/abs/2603.24373 · ONNX: https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_rec_onnx
- ESP-DL pp_ocr_v6: https://components.espressif.com/components/espressif/pp_ocr_v6
- ESP-DL operator table: https://github.com/espressif/esp-dl/blob/master/operator_support_state.md
- ESP-DL model zoo latencies: https://github.com/espressif/esp-dl/tree/master/models · https://github.com/espressif/esp-detection
- ESP-PPQ TQT (power-of-two constraint): https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/quantize_model_with_TQT.html
- ESP-NN kernels and model timings: https://github.com/espressif/esp-nn
- MobileNetV4 small 0.5: https://huggingface.co/timm/mobilenetv4_conv_small_050.e3000_r224_in1k
- edgedl: https://github.com/g1ibby/edgedl · nanoglia: https://github.com/AlixANNERAUD/nanoglia · ember-esp-nn: https://github.com/vintcessun/ember-esp-nn · MicroFlow: https://github.com/matteocarnelos/microflow-rs
- Burn: https://github.com/tracel-ai/burn · tract: https://github.com/sonos/tract
- Rust Xtensa targets: https://doc.rust-lang.org/rustc/platform-support/xtensa.html
- onnxruntime-web releases: https://www.npmjs.com/package/onnxruntime-web


## 2026-11-24: training pipeline (built by a parallel agent)

**Files.** `hxh_ocr/data.py` (dataset, width-bucketing sampler, collate),
`hxh_ocr/model.py` (models, BN folding, param/MAC counter), `hxh_ocr/engine.py`
(EMA, predict), `hxh_ocr/metrics.py` (Levenshtein, CER, confusions),
`hxh_ocr/decode.py` (greedy CTC), `scripts/train.py`, `scripts/eval.py`
(CER, top confusions, worst-32 grid), `scripts/export_onnx.py`,
`scripts/plot_log.py`.

**Input.** Grayscale, height 32 (48 optional), aspect kept, width capped at
16×H, normalised to [-1, 1]. Horizontal lines only for now (69,970 of
synth_v1's 100k). Online augmentation: horizontal scale 0.85-1.15, random
padding, 10% invert, light noise. Batches are padded to a multiple of 8. The
CTC input length uses each line's real width, ceil(w/4), not the padded width.

**Architecture (both models).** Total stride (2,2), (2,2), then (2,1) for
every later stage, so T = W/4 timesteps while the height goes
32 → 16 → 8 → 4 → 2 → 1. Height collapses with a mean (exports as
ReduceMean). The conv-only sequence head is 2 residual blocks of
[depthwise (1,5), BN, ReLU, 1x1, BN, ReLU], then dropout 0.1, then a 1x1 conv
to 75 classes. Output `[B, 75, 1, T]`. No LSTM, attention, hard-swish or SE.

- **mnv4s (pretrained):** timm `mobilenetv4_conv_small_050`. Stem 3x3 s2 with
  the RGB kernel summed to one input channel, then blocks[0] (16 ch, s2),
  blocks[1] (32 ch, first conv set to (2,1)), and blocks[2] and blocks[3]
  (6 UIB blocks each, 48 and 64 ch, the first `dw_mid` set to (2,1)). The 480-
  and 1280-channel heads and the classifier are dropped. Pretrained weights
  load unchanged: changing a stride doesn't change weight shapes.
- **tiny (from scratch):** widths tuned to match mnv4s in params and MACs, for
  a clean pretrained-vs-scratch ablation.

| model | params | MACs @32x256 | MACs @48x384 |
|---|---|---|---|
| mnv4s (4 stages) | 323.0k (317.5k after BN fold) | 30.2M | 77.0M |
| tiny | 327.8k (324.1k) | 30.1M | 70.4M |
| mnv4s, `keep_stages=3` | 96.1k | 15.3M | 33.9M |

The fp32 ONNX is 1.27 MB, so about 0.32 MB as int8. At ~0.7 GMAC/s the 4-stage
model is ~43 ms per 32x256 line on the S3. The 3-stage fallback halves that.

**Smoke test (CPU, batch 32, 250 steps, 300 val lines).** mnv4s: loss
20.1 → 3.5, val CER 0.65 at step 250. tiny: loss 20.8 → 4.4, still on the
"all blank" plateau (CER 0.98). An early hint that pretrained weights speed up
convergence; 250 steps says nothing about final accuracy.

**ONNX.** Opset 17, BN folded. mnv4s graph: Conv ×49, Relu ×32, Add ×12,
ReduceMean ×1, and nothing else. onnxruntime matches PyTorch to 3.8e-6 (static
1x1x32x256) and 4.8e-6 (dynamic width, W = 128-512), with identical argmax.
Conv shapes: 29 1x1 convs, depthwise 3x3/5x5/(1,5) at stride 1, one depthwise
3x3 and one 5x5 at stride (2,1), one full 3x3 at (2,1), two full 3x3 at (2,2).

**Watch-outs for the S3 (to be checked by the ESP-PPQ spike):** non-square
(2,1) strides; depthwise 5x5 and (1,5) are off the fast 1x1/3x3 path;
per-tensor power-of-two weights on depthwise convs; the first conv has one
input channel; 75 output classes is not a multiple of 16; residual Adds need
a requantisation shift.

**Gotchas.** On Windows, the dynamo ONNX exporter crashes printing a ✓ to a
cp1252 console unless `PYTHONUTF8=1` is set. The first data load reads every
image header to cache sizes (`sizes.npy`, about 2 min per 100k images).

**Defaults chosen.** AdamW lr 2e-3 (wd 0.05, none on BN/bias), 500-step
warmup then cosine to 1%, bf16 autocast on CUDA with CTC loss in fp32, grad
clip 5, EMA 0.999 (used for eval and export).

**Environment.** `.venv` inherits global site-packages. It holds PyTorch
2.15.0+cu132 and torchvision 0.29.2+cu132 (verified: CUDA 13.2, RTX 4080
SUPER, bf16), plus timm 1.0.31, onnx, onnxruntime, onnxscript and matplotlib.

**Data.** `data/synth_v2` finished: 500k train in 905 s (552/s) and 10k val,
grayscale JPEG q90. Combined with synth_v1, about 420k horizontal training
lines.

## 2026-11-24: ESP-PPQ quantisation spike (built by a parallel agent)

**Question.** Does every op we plan to use quantise and export for the
ESP32-S3, and how much accuracy does int8 cost? Scripts and README are in
`experiments/esp_ppq_spike/` (`run_all.ps1` reproduces everything in about 15
min). Separate venv `.venv-ppq`: esp-ppq 1.3.15 (no torch pin), onnx 1.19.1,
global torch 2.11 CPU.

**Setup.** A toy model (19k params, ~6.4M MACs) using every planned op, BN
folded. Its random-weight version was useless for measuring error: the 75
class scores were nearly tied (median top-1 margin 0.014), so argmax
agreement meant nothing. A version trained for 7 min on CPU (fp32 greedy CER
20.4%) is the one measured. Calibration: 64 real val lines. Evaluation: 256
other val lines (1,866 characters).

**Results.**
- **Every op exported for esp32s3 with zero warnings:** a stride-2 first conv
  with one input channel; depthwise 3x3, 5x5 and (1,5); depthwise stride
  (2,1); AveragePool (4,1) and ReduceMean (identical numbers); Add; ReLU.
  Opset 13, 17 and 18 and both PyTorch exporters gave identical quantised
  output.
- **Scheme confirmed from the `.espdl`:** per-tensor everywhere (all 50
  exponent vectors have length 1; per-channel exists only for P4/S31),
  power-of-two scales stored as integer exponents, symmetric int8, biases
  int32 limited to a 20-bit range, half-up rounding, NHWC. The input exponent
  is −7, so the device input is roughly `pixel − 128`.
- **Added by the exporter:** a ReLU after an Add becomes a separate op backed
  by a 256-entry lookup table. Where a skip tensor is reused at another scale,
  a RequantizeLinear op appears (2 in the trained toy model).

| run | SNR | per-timestep argmax agree | greedy decode identical | CER (fp32 20.4%) |
|---|---|---|---|---|
| PTQ (KL calibration) | 22.6 dB | 98.42% | 65.6% | 22.5% |
| PTQ + TQT | 24.6-25.0 dB | 98.86% | 67-70% | 19.9-20.5% |
| all int16 (w16a16) | 69.9 dB | 100% | 100% | 20.4% |

- **TQT** runs on CPU with no labels in about 23 s. It recovered the full CER
  loss, entirely by fine-tuning weights and biases; no exponent moved.
  Percentile/minmax calibration and mixed int16 for only the head or the
  first layers didn't help: the error is spread across all layers.
- **Sizes:** fp32 ONNX 80 KB vs `.espdl` 38 KB, of which 21 KB is weights and
  lookup tables and about 17 KB fixed graph metadata.
- **For the browser and the Rust engine:** ESP-PPQ can export a QDQ ONNX that
  onnxruntime runs (good enough for a web demo), but it is not bit-exact with
  the S3: 86% of output values differ by a median of 2 steps (99.1% argmax
  agreement). The cause is rounding (ONNX half-to-even vs S3 half-up); with
  the simulation set to half-to-even the two agree on 16,382 of 16,384
  timesteps. A `model.json` with every tensor's scale and bit width is written
  next to each `.espdl`. The `.espdl` is a 16-byte header (`EDL2`) plus a
  FlatBuffers model mirroring ONNX with exponents (`esp-dl/fbs_loader/espdl.fbs`).

**Gotchas.** `espdl_quantize_onnx` overwrites its input `.onnx` with a
simplified copy. QDQ export of ReduceMean at opset 18 makes a model
onnxruntime rejects (use AveragePool). For w16a16, build the setting with
`espdl_setting(num_of_bits=16)`, or KL calibration clips badly at 16 bits (the
first int16 run looked worse than int8 because of this). ReLU6 (Clip) is not
fused into convs; plain ReLU is.

**Design changes for the real model.** Channel counts in multiples of 16
(ESP-DL picks its fast int8 kernels only then), with the classifier padded
75 → 80. Prefer AveragePool (h,1) to ReduceMean. Benchmark (1,5) depthwise
against stacked 3x3 on hardware, since only 1x1 and 3x3 have dedicated
kernels. Budget for TQT, and consider QAT matching the target (per-tensor,
symmetric, power-of-two, half-up). The Rust engine must read the `.espdl` or
`model.json` and round half-up to match the device.

## 2026-11-24: first GPU training runs, and why they stalled

**Attempt 1: three runs at once crashed.** mnv4s (pretrained), mnv4s from
scratch, and tiny were launched together, 4 loader workers each. All three
died at their first convolution with `CUDA error: unknown error`. Isolated
convs (fp32, bf16, with and without cuDNN 9.26) worked, and a single run
trained fine, so the code is OK. Simultaneous CUDA start-up of three
processes under Windows' WDDM driver model is the suspect. Next time, start
runs a minute apart.

**A single run learns fast.** mnv4s pretrained, 600 steps (2 min, batch 128):
val CER **8.8%**, exact-line accuracy **71%**.

**But the GPU sat at 9%.** Only ~830 img/s with 6 workers. The diagnosis
took a few steps:
1. A stdin-piped benchmark hung. Windows "spawn" workers re-import the main
   script, which is impossible for a script read from stdin. Benchmarks now
   live in `scripts/bench_loader.py` and `scripts/bench_child.py`.
2. `bench_loader.py`: workers=0 gave 1,642 img/s, but workers=4 gave 787,
   workers=8 1,294 and workers=12 1,799. Each worker was 5-10x slower than
   the main process.
3. `bench_child.py`: the same per-image work in a spawned child took 0.69 ms,
   so the child itself is fast. But the main process took 4.19 ms per image
   on files it had never read, and 0.69 ms on files it had. The workers=0
   benchmark had reused the debug run's shuffle seed, so its files were
   already cached.
4. **Root cause:** the first open of each small file on Windows costs
   milliseconds (file-system and most likely antivirus overhead). With 420k
   JPEGs read in random order, that cost hits almost every sample.

**Fix:** `scripts/pack_dataset.py` packs each split into one `pack.bin`
(JPEG bytes in labels.tsv order) plus `pack_index.npy`. `LineDataset` reads
through a memory map that each worker opens lazily, so the pack isn't copied
into every worker. Worker threads are also pinned to one OpenCV and one
PyTorch thread each, to avoid oversubscription with several runs.

**After packing** (`pack_dataset.py`: synth_v1 train 883 MB in 60 s, synth_v2
train 2,959 MB in 329 s, 16 reader threads):

| loader workers | before (files) | after (pack) |
|---|---|---|
| 0 (main process) | 1,642 img/s* | 2,373 img/s |
| 4 | 787 img/s | 9,211 img/s (11.7x) |
| 8 | 1,294 img/s | 16,240 img/s |

\*The "before" workers=0 figure read files the debug run had already cached.

**Attempt 2:** `scripts/launch_ablation.sh` starts the three runs one at a
time, each only after the previous run reaches step 100, with 4 workers each.
The staggered start fixed it: all three runs now train side by side (GPU
90%, 6.1 GB), so the crash was a start-up race between CUDA processes. The
first run alone did about 15 steps/s (2,000 steps in 133 s) once the data
was packed.

Early read: mnv4s pretrained at step 2,000 is at val CER 9.25% (68.0% exact
lines); the same model from scratch is at 21.1% CER at step 1,000.

**Bug: EMA weights + live BatchNorm statistics.** At step 5,000 the
pretrained run's val CER jumped to 26% → 50% → 36% while its training loss
kept falling smoothly (0.23 → 0.17). The other two runs looked fine. The EMA
(decay 0.999) averages weights over roughly the last 1,000 steps but copied
BN running stats from the live model, so evaluation paired averaged weights
with statistics from different weights. `scripts/diag_ema_bn.py`, on the
step-5,000 checkpoint:

| weights | BN statistics | val CER | exact lines |
|---|---|---|---|
| EMA (what eval used) | copied from live model | 35.7% | 26.8% |
| EMA | re-estimated on 100 train batches | **2.5%** | **89.0%** |
| raw | stored | 5.2% | 80.0% |
| raw | re-estimated | 4.4% | 82.1% |

Pretrained weights move fast at lr 2e-3, which makes the mismatch worst for
that run. It also corrupted "best checkpoint" selection and every learning
curve. **Fix:** `engine.recalibrate_bn` re-estimates the BN stats of a copy of
the EMA model on 50 training batches (cumulative average) before every
evaluation, and that copy is what gets saved, so the statistics folded into
the convs at export are the right ones. All three runs were restarted
(about 6 min lost each), so the blog's ablation curves are clean.

**tiny_h32 finished** (30k steps in 1,218 s, best val CER 1.77% on the
2,000-line eval subset). Full eval on all 7,068 synth_v2 horizontal val lines:
**CER 1.27%, exact lines 93.8%**. Top confusions:

- dakuten / handakuten / no mark: ぱ→ば 13, ず→す 12, ぶ→ふ 11, べ→ぺ 11,
  ぶ→ぷ 10, ぷ→ぶ 10, ぴ→び 10, ば→ぱ 10, ぺ→べ 9, が→か 9, か→が 8, じ→し 8…
- missing spaces 30, missing ら 11, missing る 9.

The marks are 1-2 px at 32 px line height (dakuten radius 0.065 em, em = 13-24
px), so this is the expected failure.

**Worst-32 grid** (`figures/2026-11-24_tiny_h32_worst32.png`): most of the
worst cases are generator bugs, not model errors.
1. **Hard binarisation** (Otsu on the whole image, 5% of samples) turns text on
   dark or patterned backgrounds into a black blob, so the label is unreadable.
2. **Neighbour lines** sometimes show as much as the real line, so which line
   is labelled is ambiguous.
3. Real errors: mark confusions and very faint text.

**Next data version (synth_v3):** binarise only when the text survives (or use
adaptive thresholding), keep neighbour lines mostly outside the crop (at most
about 35% of their height visible, main line centred), and test 48 px input
and/or tighter crops (em ≥ 0.55 of line height) for the marks.

**Generator v3 changes** (`hxh_ocr/synth.py`; preview in `previews/samples_v3.png`):
- **Tighter crops:** em size is U(0.58, 0.84)×line height on 65% of lines,
  U(0.40, 0.62) on the rest (v2: U(0.42, 0.74)). Bigger glyphs give the
  dakuten/handakuten a few more pixels.
- **Neighbour lines:** only a sliver of each neighbour (≤ 35% of its height)
  enters the crop, at least 0.12 em away from the labelled line, and none at
  all if there's no margin.
- **Binarisation:** thresholds at the midpoint between the known ink and
  background luminance, only when the background is fairly uniform (std
  < 0.1), and before the other degradations. Text always survives.

### Ablation result (synth_v1 + v2, 32 px, 30k steps × batch 128, BN-fixed eval)

| model (all ~323k params, ~30M MACs @32x256) | wall time | best val CER (2k subset) | **full synth_v2 val CER** | exact lines | CER at step 2k |
|---|---|---|---|---|---|
| mnv4s, ImageNet-pretrained | 1,727 s | 1.54% | **1.16%** | **94.4%** | 3.0% |
| mnv4s, from scratch | 1,757 s | 1.78% | 1.47% | 93.0% | 7.5% |
| tiny, from scratch | 1,218 s | 1.77% | 1.27% | 93.8% | 5.1% |

Figure: `figures/2026-11-24_ablation_h32.png` (train loss, val CER, val
line accuracy).

- **Pretraining pays off.** The same network converges far faster (3.0%
  vs 7.5% CER at step 2k) and ends about 21% lower in CER. ImageNet's
  low-level filters transfer even to synthetic geometric glyphs.
- **The plain from-scratch tiny net beats the from-scratch MobileNetV4**
  (1.27% vs 1.47%) and trains 30% faster in wall time (fewer, wider
  layers). Worth remembering for the S3, where per-layer overhead counts
  too.
- All three plateau around 1.2-1.5%, dominated by dakuten/handakuten
  confusions plus v2's label-noise bugs. The pretrained model's top
  confusions: ぱ→ば 12, ぷ→ぶ 12, ば→は 10, ぶ→ふ 9, だ→た 9, び→ぴ 9… and
  a new one, も→ち 7: the filled-dot も and ち are both a cross with a dot.
  v3 data (tighter crops, fixed binarisation and neighbours) and a 48 px
  input are the next levers.

## 2026-11-24: hxh-infer v0, the lean Rust int8 engine (built by a parallel agent)

**Result.** The engine reproduces ESP-PPQ's int8 simulation (the ESP32-S3
semantics) **bit for bit**: every logit and every greedy CTC decode on all
test vectors, on host and as wasm in node. I re-ran it myself: on the real
pretrained mnv4s_h32 (PTQ) it is bit-exact on **512/512** held-out val lines,
plus an independent naive reference implementation agrees. That run hit
18.6M exact .5 rounding ties in conv outputs, so the rounding really is
exercised. 10 unit and golden tests pass, clippy is clean.

**Pieces.**
- `scripts/espdl_to_hxhm.py` (`.venv-ppq`) reads the official `.espdl` and
  writes `.hxhm`, a minimal format: a 64-byte header, 16-byte tensor records
  (NHWC shape, exponent, precomputed arena offset), 40-byte op records (6
  kinds: Conv, DwConv, Add, Lut, Requant, AvgPool), and 16-byte-aligned
  blobs. Weights use ESP-DL's 16-output-channel blocking. A u8-pixel→int8 input
  table folds in normalisation and input quantisation, so the runtime does
  no float math on the input. Arena offsets come from first-fit planning over
  tensor lifetimes, so there is no heap at runtime. Every unpacked filter is
  checked against ESP-PPQ's own quantised values; unsupported graphs are
  rejected, not guessed at. Documented in `rust/FORMAT.md`.
- `rust/hxh-infer`: `#![no_std]`, no alloc, no dependencies. Scalar kernels
  sit behind a `Kernels` trait so an S3 SIMD backend can replace conv and
  dwconv. Greedy CTC included.
- `rust/hxh-infer-wasm`: raw `extern "C"` exports, no wasm-bindgen.
- `rust/hxh-cli`: run a PGM image or a golden file, `--profile`.

**Sizes and speed.**

| artifact | size |
|---|---|
| real model `.hxhm` (mnv4s, int8) | 338 KB, arena 80 KB |
| toy model `.hxhm` vs `.espdl` | 22.9 KB vs 37.9 KB |
| wasm, opt "z" + LTO + strip | 11.3 KB (toy model ~0.95 GMAC/s in node) |
| wasm, opt 3 | 16.9 KB (1.8 GMAC/s) |
| wasm, opt 3 + simd128 | 33.7 KB (2.3 GMAC/s); the browser build to use |

The host runs single-thread scalar at about 3.5-4.7 GMAC/s, so the real
model takes 5.4-8.7 ms per 32x256 line (measured under load). 1x1 convs are
70% of the time (75% of MACs), 3x3 convs 16%, depthwise 13%.

**ESP-DL semantics that needed care** (read from the ESP-DL C source and
ESP-PPQ):
- **Rounding is half-up:** `(acc + 2^(s−1)) >> s`. Every alternative breaks
  every test vector:

  | rounding mode | logit values changed | decodes changed (of 69) |
  |---|---|---|
  | half-to-even | 250 K | 11 |
  | half-away-from-zero (ESP-NN / TFLite style) | 87 K | 6 |
  | plain shift (floor) | 317 K | 35 |

- **ESP-DL's int8 Add ignores exponents:** it's an int32 add plus clip. That
  only works because ESP-PPQ inserts RequantizeLinear so both inputs share
  the output's exponent, and the converter enforces it.
- **LUT ops** index with `x + 128`. Conv bias sits at exponent `in + w`, and
  ReLU applies after saturation. Padded taps are skipped (same as zero
  padding). AvgPool multiplies by a float reciprocal, and the exact f32
  sequence is reproduced.
- **Input exponent −7:** pixel 127 maps to −1 and 128 to +1, so the int8
  input is never 0.

**Implications for the S3 backend.**
- **ESP-NN is not a drop-in:** its TFLite-style requantisation rounds half
  away from zero (and twice), which is the 87 K-value / 6-decode mismatch
  above. To stay bit-exact, port ESP-DL's MIT-licensed TIE728 kernels (their
  16-channel blocking is already the `.hxhm` layout) or write our own asm.
  Priority: the 1x1 conv.
- **Unverifiable without a board:** the S3's conv accumulates in 20-bit QACC
  lanes, while PPQ simulates in float32. Observed max |acc| is 73,193
  against a limit of 524,287 (worst-case bound 130,028), so there's headroom,
  but re-check every new model. Also unverified: non-power-of-two pooling
  areas (keep them powers of two), TIE728 RequantizeLinear rounding, and
  negative shifts (none occur yet).

## 2026-11-24: int8 on the real model, and why we need QAT

`experiments/esp_ppq_spike/quantize.py` on the pretrained mnv4s_h32 (ONNX
opset 18, static 1x1x32x256), target esp32s3, 64 calibration lines, 512
held-out lines. Lines longer than 256 px are squeezed to the static width
here, which is why fp32 CER is higher than in training eval.

| version | holdout CER | SNR vs fp32 | per-timestep argmax agree | lines decoded as fp32 does | time (CPU, busy) |
|---|---|---|---|---|---|
| fp32 | 3.68% | – | – | – | – |
| int8 PTQ (KL) | 5.88% (+2.2) | 9.8 dB | 98.0% | 85.2% | 665 s |
| int8 PTQ + TQT | 4.87% (+1.2) | 14.3 dB | 98.7% | 90.4% | 310 s |

`.espdl` is 400 KB (fp32 ONNX 1.3 MB). The QDQ ONNX matches ESP-PPQ exactly
when its rounding is set to half-to-even (0 differing timesteps), as the
spike found.

The real 49-conv network loses far more to per-tensor power-of-two int8
than the toy model did (9.8 dB vs 22.6 dB SNR). TQT recovers only about half.
**Next: quantisation-aware training** that simulates the ESP-DL scheme in
the forward pass (BN folded, per-tensor symmetric power-of-two int8 weights
and activations, half-up rounding, shared exponent at residual Adds),
fine-tuned with CTC on real labels. Then ESP-PPQ PTQ/TQT on the QAT weights.
Also needed: a wider static input (e.g. 32x512) or several widths, so long
lines aren't squeezed.

## 2026-11-24: round 2 crashed on memory (Windows commit limit)

Round 2 (v3 data: mnv4s 32 px, mnv4s 48 px, mnv4s3 32 px; 3 runs × 4 loader
workers), with the QAT and web-demo agents also running, died with
`RuntimeError: Couldn't open shared file mapping ... error code 1455`.
That's Windows' "paging file too small": committed memory was **71.7 of
72.2 GB**. Each training main process held 4-6 GB private, and each of the
12 loader workers about 1.7 GB, because on Windows every spawned worker
re-imports torch and receives its own pickled copy of the dataset (three
350k-element Python lists: paths, labels, targets). The desktop itself
(browser, chat apps, etc.) holds about 20 GB.

**Fixes.**
- `LineDataset` stores paths and labels as numpy unicode arrays and targets
  as one flat int16 array plus offsets, so a worker's pickled copy is 107 MB
  of contiguous arrays.
- `prefetch_factor` 4 → 2.
- 2 workers per run: with packed data that's still 4,860 img/s, more than a
  GPU-sharing run consumes.
- `launch_round2.sh` runs at most 2 jobs at once (`MAXPAR`); the 3-stage
  model starts when a slot frees up.

### Round 2 (synth_v3), first result: mnv4s 32 px

30k steps in 1,921 s (2 runs sharing the GPU, 2 workers each).
- Best val CER on the 2k subset: **0.78%** (round 1 on v1+v2: 1.54%).
- Full synth_v3 val (6,998 horizontal lines): **CER 0.84%, exact lines 95.4%**.
- Top confusions are still the small marks: が→か 12, ぽ→ぼ 10, ぴ→び 9,
  ば→は 9, だ→た 9, ぱ→ば 9, ぷ→ぶ 7, ぼ→ぽ 7, ぺ→べ 7, げ→け 7, べ→へ 7…;
  missing spaces 17.
- Worst-32 (`figures/2026-11-24_v3_mnv4s_h32_worst32.png`): the v2
  binarisation blobs and full-size neighbour lines are gone. What's left is
  hard but legitimate: very low contrast or blur, heavy screentone, eroded
  binarised strokes, small marks, single glyphs cropped tight (ら ⅃ vs た ⊥).
- One generator quirk remains: a "shapes" background can draw a blob as
  dark as the ink, hiding a glyph (てほんなうたす). Rare; fix in the next
  generator pass by keeping shape luminance on the background's side of the
  ink.

### Round 2: mnv4s 48 px

30k steps in 1,977 s. Full synth_v3 val: **CER 0.64%, exact lines 96.5%**
(32 px: 0.84% / 95.4%). About a quarter fewer errors, at 2.5x the MACs (77M
vs 30M per line). Marks improve, e.g. が→か 12 → 7, but stay the main error
(ぜ→せ 9, げ→け 8, ぶ→ふ 8…). That's a real trade-off for the S3, where 48 px
is about 2.5x slower.

## 2026-11-24: mobile browser demo (built by a parallel agent)

`web/`: `index.html`, `app.js`, `ocr.js` (preprocessing, engine wrapper,
decode, romaji), `style.css`, `hxh_infer.wasm` (opt 3 + simd128, 33 KB),
`hxh_infer_nosimd.wasm` (17 KB fallback), `models/model.hxhm` +
`model.json`, 6 example lines. `scripts/build_web.ps1` builds everything,
including `web/dist/hxh-ocr.html`, a single 642 KB file with no network
requests.

**Verification (node).**
- JS grayscale and area resize match OpenCV 4.12 exactly (294/294 resize
  cases; OpenCV 4.x uses 15-bit gray weights).
- The JS 32x256 strip matches the Python calibration pipeline pixel for pixel
  on 300/300 lines, so decodes match 300/300.
- The wasm decoder matches the JS decode, and romaji matches `to_romaji` on
  every label.

**Accuracy with the round-1 int8 model (whole-line box, 300 v3 val lines).**
245/300 exact, CER 8.1%. Lines that fit in 256 px: CER 3.0%. Lines squeezed
from wider: 13.8%. So the static 256 width is the main limit.

**Box framing matters most.** CER is 3.0% as cropped, 15.9% with 25% extra
height, 44.8% with 50%. The agent added **auto-fit** (find the text band,
reframe it like the training crops, read both boxes, keep the more confident
one). With auto-fit: 2.3% / 6.0% / 15.8%, and tight hand-drawn boxes on
rendered "sign photos" read 7/8 lines instead of 2/8. Light-on-dark text is
fine (2.5% with auto-fit).

**Speed.** ~12 ms per inference in node with SIMD (15-17 ms without); in the
browser pane, 25-32 ms for the two auto-fit boxes. I checked the page myself
at 375x812: the examples read れおりお (Leorio) and くじらじま (Whale Island)
correctly, with no console errors.

### Demo upgraded to the v3 model at 32x512

The round-2 32 px model was exported at a static 32x512 input (T = 128,
~60M MACs), so about 16 glyphs fit before squeezing instead of 8. It was
quantised for the S3 with ESP-PPQ PTQ + TQT, calibrated and evaluated on v3
data (`HXH_DATA` / `HXH_W` overrides added to `spike_common.py`). Results on
512 held-out lines:

| version | holdout CER | lines decoded as fp32 does |
|---|---|---|
| fp32 | 0.68% | – |
| int8 PTQ + TQT | **1.16%** (+0.48) | 94.1% |

The Rust engine is bit-exact with ESP-PPQ on all 512 lines (max |acc|
75,052, within the S3's 20-bit lanes). `.hxhm` is 331 KB.

In the demo, all 6 example lines read correctly (れおりお, くじらじま,
ほてるねてろ, ぱーていー, ようこそみるき, ぐりーどあいらんど) at about 25 ms
per box (WASM SIMD, desktop). Page fixes: the strip's aspect ratio and the
"glyphs per line" tip now follow the model's input size, and the CSS dark
theme respects the host's `data-theme`.

`scripts/make_artifact_page.py` derives `web/artifact.html` (no
doctype/head/body) for publishing as a claude.ai Artifact with its assets
alongside. **Publishing is pending the owner's OK**: it uploads the code and
model to claude.ai, and the automatic permission check refused it without
explicit approval.

**Published (owner approved):** private claude.ai Artifact "HxH Alphabet
Reader", (a private claude.ai link) (version 1). The
page plus `style.css`, `app.js`, `ocr.js`, both wasm builds, 6 examples and
the model. Artifacts serve only web media types, so a raw `.hxhm` was
rejected. The model ships base64-encoded as `models/model.hxhm.b64.txt`
(451 KB, round-trip verified byte-identical), with a `model.json` variant
pointing at it; `app.js` decodes `*.b64.txt`. `scripts/make_artifact_page.py`
regenerates the page, the base64 model and the JSON variant. Not yet
verified inside the artifact frame (the browser pane isn't signed in to
claude.ai): the open question is whether the frame's CSP allows
WebAssembly.

### Round 2 summary (synth_v3, 30k steps, fp32, full v3 val, 6,998 lines)

| model | params | MACs (input) | S3 estimate* | train time | **val CER** | exact lines |
|---|---|---|---|---|---|---|
| mnv4s 48 px | 323k | 77M (48x384) | ~110 ms | 1,977 s | **0.64%** | 96.5% |
| mnv4s 32 px | 323k | 30M (32x256) | ~43 ms | 1,921 s | 0.84% | 95.4% |
| mnv4s3 32 px (3 stages) | 96k | 15M (32x256) | ~22 ms | 1,342 s | 1.14% | 94.0% |

\*At the ~0.7 GMAC/s Espressif measured for ESP-DL on the S3; our Rust engine
needs ported ESP-DL kernels to get there. Width 512 doubles MACs and time.
Curves: `figures/2026-11-24_round2_v3.png`.

- **Clean data paid off:** the same mnv4s 32 px went from 1.16% (v1+v2
  data, round 1) to 0.84%, with no model change.
- **More input pixels beat more layers** for this task: 48 px cuts errors by
  about 24% at 2.5x MACs. The 3-stage model costs +0.30 points of CER for
  half the MACs and 30% of the parameters (96k, about 100 KB as int8).
- In every model, almost all remaining errors are dakuten/handakuten, which
  depend on resolution. The 3-stage model's top confusions: ば→は 14,
  が→か 12, せ→ぜ 10, ず→す 9, ゔ→う 9; missing spaces 37.
- **S3 candidates:** mnv4s3 at 32 px for speed, mnv4s at 32 px as the
  balance, mnv4s at 48 px if a ~110 ms+ line is acceptable. Final choice
  waits for QAT results and real-image tests.

**First real-image test (owner, 2026-11-24):** the owner opened the
published demo and tested it on a real Hunter-script image: "it works
great". This is the first evidence that training on purely synthetic data
transfers to real text. It also confirms the artifact frame runs the
WebAssembly engine (the open CSP question above). Model: v3 mnv4s 32 px,
32x512, int8 PTQ+TQT. Next: collect such images into a small labelled real
test set, so the transfer can be measured as a number instead of an
impression.

## 2026-11-24: quantisation-aware training (built by a parallel agent)

**Implementation.** `hxh_ocr/qat.py` + `scripts/train_qat.py`. torch.fx traces
the BN-folded model and inserts fake-quant that reproduces ESP-PPQ's rules,
read from its source:
- weights: 99.99th-percentile max, exponent rounded up, recomputed every step;
- bias: int32 at exponent in+w, clamped to 20 bits;
- half-up rounding, ReLU fused before quantising;
- Add inputs requantised to the Add's exponent;
- input exponent fixed at −7;
- activation exponents from a re-implementation of ESP-PPQ's KL calibration,
  recalibrated every 100 steps.

Checkpoints hold the stripped float model (BN folded, `folded: True`), so
export and eval work unchanged. **Verified:** with the same weights and
calibration lines, all 49 conv exponents agree with ESP-PPQ and every
holdout logit is identical.

**Results** (QAT applied to the round-1 mnv4s_h32 for 4k steps on synth_v3, 611 s):

| model | fp32 CER | int8 PTQ | int8 PTQ + TQT |
|---|---|---|---|
| baseline, v1 holdout (512 lines, 32x256) | 3.68% | 5.88% | 4.87% |
| QAT, v1 holdout | 3.92% | **4.06%** | 5.28% |
| baseline, v3 holdout (1,500 lines) | 3.16% | 8.25% | – |
| QAT, v3 holdout | 3.58% | **3.68%** | – |

- **QAT closes the int8 gap:** int8 is within 0.1-0.14 points of its own
  fp32, against 2.2-5 points for plain PTQ. The fp32 drift (+0.24) comes from
  fine-tuning a v1/v2 model on v3 data. A float-only control fine-tune
  drifted far more (3.89 → 7.36%), so QAT itself isn't the cause.
- **ESP-PPQ's KL calibration is unstable:** a 1e-5 relative weight
  perturbation flips 2-10 of 51 activation exponents. The baseline's PTQ int8
  CER ranges 5.6-8.9% across six near-identical calibrations, so the single
  5.88% reading understated the risk. QAT narrows the spread to 4.2-4.5%.
- **TQT after QAT hurts** (4.06% → 5.28%): it pulls blocks back towards fp32
  and undoes what QAT learned. Recipe: QAT, then plain ESP-PPQ PTQ (KL).
- **Didn't help:** training under randomly sampled exponents (4.70% vs
  4.47%); minmax calibration (deterministic, but less accurate and fragile
  when the calibration lines change).
- **Recommended:** QAT from the final model's own checkpoint on its own data,
  with the same calibration lines in QAT and ESP-PPQ (synth_v3 val), plain PTQ,
  and checkpoint selection by int8 val CER.

## 2026-11-24: first real test set (scraped by a cheap agent) and real-image CER

**Collection.** A Haiku agent (cheapest model) scraped Hunterpedia for real
Hunter-script images. It viewed every candidate and kept 48 (40 anime frames,
6 manga, 2 merch; 7.9 MB). `data/real/raw/` + `manifest.csv` record the source
and page URL of every image. Most came from the wiki's "Hunter × Hunter
Alphabet/Image Gallery" through the MediaWiki API; web search found nothing
usable. These are copyrighted images: they stay local for evaluation only and
are never published. Contact sheet: `previews/real_raw_sheet.jpg` (local).

**Free labels.** 25 of the images are the 2011 anime's "ハンター語講座"
(Hunter Language Course) title cards: one word in Hunter script with its
katakana reading printed underneath. Kanji readings were converted by hand
(ゼビル島 = ぜびるとう, shown in furigana; 号泣観光 = ごうきゅうかんこう;
試しの門 = ためしのもん; 待ち時間 = まちじかん). `scripts/real_cards.py` (not in the public repository) crops
26 lines (ep22 has two). How the cropping evolved:
- row-projection bands merged the header, the text and the katakana;
- centroid grouping cut off glyphs whose centre of mass sits low (⊥);
- the final version uses connected components of near-black ink: glyph-sized
  shapes only, rows grown by vertical overlap, small marks (dakuten) added
  inside the band;
- one hand-set box for ep10, where Hisoka's dark hair outlines look like glyphs.

**Results** (`scripts/eval_real.py`; int8 runs through the Rust engine on
the exact demo/S3 input; 26 lines, about 100 characters, so 1 error ≈ 1
point and the ranking is mostly noise):

| model | CER | exact lines |
|---|---|---|
| fp32 mnv4s_h32 (round 1, v1+v2 data) | 6.93% | 76.9% |
| fp32 v3_mnv4s_h32 | 8.91% | 73.1% |
| fp32 v3_mnv4s_h48 | 10.89% | 65.4% |
| fp32 v3_mnv4s3_h32 | 6.93% | 76.9% |
| **int8 demo model (v3_mnv4s_h32, PTQ+TQT, 32x512)** | **5.94%** | **80.8%** |
| int8 QAT v3_mnv4s3_h32 | 8.91% | 76.9% |
| int8 QAT v3_mnv4s_h32 | 7.92% | 73.1% |

**The error types are the useful part** (consistent across all 7 models):
1. **Extra glyphs at the line end** (くらぴか+つ/と/く, ざんねん+く,
   ためしのもん+と/き/く). The crop's right margin catches character art,
   and the model reads it as a glyph. Training never shows non-text shapes
   beside a line.
2. **ま is mirrored in the official anime font:** "Ƹ" instead of the chart's
   Σ, so every model reads な or へ (まちじかん → なち/へち…). It's a genuine
   variant, like the two forms of も.
3. **The anime's せ is the compact "4" form** (as in Omniglot's chart), not
   Hunterpedia's wide double-triangle. It was still read as せ, but its
   tucked-in dakuten was missed (ぜびるとう → せ…). じ → し in まちじかん is
   also a missed dot.
4. **Heavy strokes:** the anime font is bolder than our heaviest style, and
   ろ's ring is read as り's filled dot by some models.
5. ひそか → ひそた: Hisoka's hand covers か's V arms. Legitimate occlusion.

The anime font otherwise uses exactly our traced shapes (れ, お, り, き, ゆ,
こ, か, ん, ぽ, ー…), with a soft grey drop shadow on busy colourful
backgrounds.

**Generator fixes this points to (v4):** non-text distractor shapes (art
fragments, blobs, curves) touching the line ends; glyph variants for ま
(mirrored) and せ (compact); stroke widths up to about 0.3 em; more frequent
soft drop shadows. The other 22 scraped images (signs, letters, TV screens,
manga, merch) still need hand transcription. Several mix in digits and
punctuation, which only Hunter-only segments can be scored on.

### QAT on the round-2 models (`scripts/qat_pipeline.sh`)

Recipe from the QAT entry: QAT from each model's own checkpoint on synth_v3
(lr 1e-4, 4k steps), calibration lines = synth_v3 val at the deployment
width, then ESP-PPQ plain PTQ (KL, no TQT), `.hxhm`, and a Rust bit-exact
check. 512 held-out v3 lines. "fp32" is the QAT weights run without fake
quantisation, which is not what they were trained for:

| model (deploy input) | "fp32" | **int8** | Rust bit-exact | `.espdl` | wall time |
|---|---|---|---|---|---|
| demo model: v3 mnv4s 32 px, PTQ+TQT (32x512) | 0.68% | 1.16% | 512/512 | 400 KB | – |
| QAT mnv4s3 32 px (32x512) | 1.78% | **0.82%** | 512/512 | 141 KB | 8 min |
| QAT mnv4s 32 px (32x512) | 1.69% | **0.96%** | 512/512 | 403 KB | 11 min |
| QAT mnv4s 48 px (48x768) | 1.64% | **0.51%** | 512/512 | 402 KB | 13 min |

On synthetic data QAT beats PTQ+TQT everywhere, and the 96k-parameter
3-stage model beats the 4-stage PTQ+TQT model. **On the 26 real lines it
doesn't show:** QAT 48 px int8 is 10.89% (same as its fp32), QAT 32 px
7.92%, QAT 3-stage 8.91%, against 5.94% for the current demo model. The
real errors are data problems (mirrored ま, art at the line edges, missed
dots, heavy font) that precision can't fix, and with about 100 characters
the spread is within noise. **Decision:** keep the current demo model;
generator v4 comes next, then QAT again, then pick the model on a larger
real set.

**Real CER set aside (owner's call).** The title-card set isn't a good
benchmark: 26 lines (about 100 characters) from a single template, with a
clean bold font, near-uniform backgrounds, and crops that catch character
art. It can't separate models, so it no longer drives decisions. Model
choice goes back to synthetic int8 val CER, with real images used only as
qualitative spot checks. The failure modes it surfaced (mirrored ま, compact
せ, art at the line edges, heavier strokes) are still valid generator
inputs. Transcribing the other 22 scraped images is shelved.

### Demo model swapped to QAT 48 px (published version 2)

Picked on synthetic int8 CER, now that real CER is set aside: QAT mnv4s
48 px, ESP-PPQ PTQ, 48x768 input, 0.51% int8 holdout CER (previous demo
model 1.16%). It is 338 KB int8 and 154M MACs per line. The page now shows
the model's real input size in the "How it works" text, and the
"glyphs per line" tip follows the input's aspect ratio. Checked locally: all
6 examples read correctly, at about 47 ms per box with WASM SIMD on the
desktop (≈95 ms when auto-fit tries two boxes). Republished to the same link
(version 2).

## 2026-11-24: the official alphabet chart (2011 anime)

The owner shared two frames of the 2011 anime's "ハンター語講座" alphabet
chart: all 46 base glyphs in the official font, and the same layout in
hiragana (`data/real/reference/`, local only). A glyph-by-glyph comparison
with our Hunterpedia trace:

- **43 of 46 match:** あ-お, か-こ, さ/し/す/そ, た-と, な-の, は-ほ, み/む/め,
  や/ゆ/よ, ら-ろ, わ/を/ん. This is the first official confirmation of the
  trace.
- **ま is mirrored.** The zigzag's point faces left (Ƹ); Hunterpedia's chart
  points it right (Σ). This explains the まちじかん → なち/へち misreads on
  the title cards.
- **せ is drawn narrow.** Same two triangles on one diagonal, but the
  diagonal is near vertical, giving the "4" look Omniglot also shows. Our
  trace is about 1.6x wider for its height, beyond what per-glyph aspect
  jitter (about ±20%) reaches.
- **も ends in a hollow ring**, like Omniglot's form, which we already
  generate as a variant.

**Change:** `glyphs.py` gains variants for ま (mirrored about its centre) and
せ (x squeezed to 0.6 around its centre), drawn 50/50 with the traced forms,
like も. (The comparison figure is not in the public repository: its top half shows crops of the anime.) These go into generator v4 along
with non-text distractors at the line ends and heavier strokes.

**Second official source: a printed guidebook chart.** The owner shared a
scan of a printed "獵人文字講座" chart, apparently from a Chinese-language
edition of an official HxH guidebook (Tumblr watermark). It also shows the
dakuten (゛ = small filled dot) and handakuten (゜ = small ring). Local copy:
`data/real/reference/guidebook_hunter_chart_zh.webp`.

| glyph | Hunterpedia fan chart (our trace) | printed guidebook | 2011 anime |
|---|---|---|---|
| ま | point faces right (Σ) | **point faces left (Ƹ)** | **point faces left (Ƹ)** |
| せ | wide | wide | narrow ("4") |
| も | filled dot | filled dot | hollow ring |

The fan chart's せ and も are the print forms, and the anime restyled them,
so generating both forms of each is right. ま is different: both official
sources point it left, so the fan chart's Σ is most likely an error (in the
chart or in our reading of it). **Change:** ま's main form is now
left-pointing (85% of samples), with Σ kept as a rare variant (15%) for
fan-made renditions. `Glyph.weights` and `_VARIANT_WEIGHTS` add per-form
sampling weights (verified: 85.3% left-pointing over 2,000 samples). Note
that every model trained so far learned ま only as Σ.

## 2026-11-25: generator v4 and round 3

**Generator v4 changes** (from the title-card failures and the two official
charts; preview `previews/samples_v4.png`):
- **Official glyph forms:** ま points left (85%, Σ 15%), せ narrow and wide
  50/50, も ring and dot 50/50 (`glyphs.py`, previous entries).
- **Non-text "character art" beside the line ends** (25% of lines): blobs
  with dark outlines, spiky hair-like shapes and thick outline strokes.
  They sit only in the margin beyond the text box, after a 0.12-0.4 em gap,
  and the crop is widened on that side to make room (`_art_fragments`).
- **Heavier strokes:** the bold style now covers 18% of lines (was 10%) and
  reaches 0.30 em (was 0.24).
- **Drop shadows** on 18% of lines (was 10%), 60% of them soft and offset
  down-right like the anime cards.
- **Dakuten/handakuten placement** varies from tucked against the glyph to
  well clear (0.55-1.35x the chart offset, small vertical shift). A minimum
  gap of stroke half-width + mark radius + 0.03 em keeps a bold stroke from
  swallowing the mark.
- Speed unchanged: 78 samples/s per core.

**Data:** `data/synth_v4`, 500k train + 10k val, grayscale, seed 3, packed.
**Round 3** (`scripts/launch_round3.sh`): mnv4s 48 px (browser demo) and
mnv4s3 32 px (ESP32-S3 candidate), from ImageNet weights on v4 only. v3
drew ま only as Σ, so mixing it in would teach the wrong orientation.
QAT and int8 via `DATA=data/synth_v4 scripts/qat_pipeline.sh`.

**In parallel:** a page-mode generator (milestone step 1, own module) and
WASM SIMD kernels for hxh-infer, as background agents.

**Decision (owner, 2026-11-25): Rust for everything that doesn't need
Python.** The page-mode generator is being built as a Rust crate
(`rust/hxh-synth`) instead of Python. Python stays for PyTorch training, QAT,
ESP-PPQ quantisation, and exporting Python-owned data: the glyph strokes,
variants, weights, vocab and word lists go to `assets/glyphs.json` via
`scripts/export_glyphs.py` and are embedded in Rust, so there is still one
source of truth. A side benefit: data generation was CPU-bound in Python
(78 lines/s per core).

## 2026-11-25: WASM SIMD kernels for hxh-infer (built by a parallel agent)

`rust/hxh-infer/src/kernels/wasm_simd.rs`: a `Simd128` backend (only in
wasm32 builds with simd128) for conv2d (any kernel, stride and padding,
including the one-channel stem), depthwise conv (windows up to 64 taps),
avgpool, add and requant. The scalar code stays the reference.
`hxh_infer::Fast` picks the SIMD backend in SIMD builds. The wasm export
`hxh_set_simd(0|1)` switches to scalar for A/B checks on a phone.
`rust/KERNELS.md` documents what a backend must implement, which is also the
contract for a future ESP32-S3 TIE728 backend.

**Bit-exact everywhere** (re-run myself with `test_wasm.mjs --all`): 3 toy
sets × 69 vectors and 5 real models × 512 vectors, with 0 differing logit
values and identical CTC ids, in both the SIMD and no-SIMD builds. A
randomised self-test compared 58,757 cases (kernels 1-5, strides 1-3,
asymmetric padding, odd channel counts, shifts −24..31, all-(−128) inputs)
with 0 mismatches; it does catch a deliberately broken kernel (76
mismatches).

**Speed (node 24, Ryzen 7 3700X, one thread, CPU shared):**

| model (MACs) | old demo build | **SIMD** | speedup |
|---|---|---|---|
| 48 px, 48x768 (154M) | 65.5 ms | **9.5 ms** | ~7x |
| 32 px, 32x512 (60.5M) | 25.6 ms | **3.7 ms** | ~7x |
| 3-stage, 32x512 (30.5M) | 14.9 ms | **2.0 ms** | ~7.6x |

About 16 GMAC/s. The 1x1 convs run at 28 GMAC/s, about 90% of Zen 2's limit
of one `pmaddwd` per cycle. SIMD wasm grows from 34 KB to 54 KB (20 KB gzipped).

**What made SIMD tricky in ESP-DL's semantics:**
- two (−128)·(−128) products overflow i16, so pairs go straight into i32
  with `i32x4.dot_i16x8_s`;
- the half-up rounding constant overflows i32 for shifts ≥ 30, which needs
  an equivalent formula;
- negative shifts clamp before shifting;
- the partial last block (75 classes) needs masked loads;
- the stem's odd reduction length (3 terms) pairs the last term with a zero
  weight;
- AvgPool is float in ESP-DL, so the vector version must do the same f32
  operations in the same order.

**Not measured:** ARM phones. V8 lowers the dot instruction to three ops on
arm64; an extmul-based kernel set might double phone speed if phone timings
disappoint.

## 2026-11-25: page-mode generator in Rust (`rust/hxh-synth`), milestone step 1

Built by a parallel agent. It started in Python and switched to Rust on the
owner's call; the Python version was measured first, then deleted.

**One source of truth.** `scripts/export_glyphs.py` exports `glyphs.py` /
`text.py` (73 glyphs and 77 forms in em units, form weights, mark geometry,
normalisation tables, sampler frequencies and 264 words) to
`rust/hxh-synth/assets/glyphs.json` (67 KB), which the binary embeds. The
Rust crate ports the v4 renderer: styles, layout with small kana, spaces,
weighted variants and varied mark gaps, wobble/jitter, brush stamping.

**Pages.** 512-1024 px on the long side. Three layouts:
- manga (60%): bordered panels with per-panel backgrounds (screentone,
  speed lines, hatching, gradients, scenes);
- frame (25%): an anime still or poster;
- doc (15%): paper with paragraphs and ruled tables.

Text groups are speech bubbles (ellipse, cloud, burst, with tails),
captions, signs and cards with soft shadows, paragraph blocks, big
outlined titles/SFX, small fillers and table cells. They are rotated up to
±12°, sometimes in perspective, and about 10% are vertical. Distractor art
is drawn before and after the text, and about 3% of pages are art-only
negatives. Quads come from the stroke geometry, so they are tight; lines
never overlap.

**DBNet targets** (`targets.rs`): shrink D = A(1−r²)/L with r = 0.4,
mitred kernel, threshold map 0.3-0.7, dilated mask, optional
vertical-orientation map. A numpy pixel-centre fill reproduces the kernels
exactly; cv2.fillPoly is about 0.5 px fatter per side, so the trainer should
port the numpy rule.

**Speed:** 92 pages/s on 6 threads against 16.5 for the Python version
(5.6x; about 3.5x per core), with the CPU shared. The binary is 1.5 MB.
Smoke set `data/pages_v1`: 2,000 + 200 pages, 16 lines/page on average (max
49), median line height 19 px (p1 9 px, p99 71 px), 11% vertical. Preview:
`previews/pages_v1.png`.

**Next for the detector:** about 0.35M-param ReLU-only DBNet-lite (stride-2
probability head for small text, FPN with nearest Resize + Add + Concat,
threshold branch used only in training). Loss: BCE with OHEM, L1 and Dice.
No flips, because ま/せ are mirror-sensitive. hxh-infer needs Resize-nearest
and Concat kernels. Measure MACs at 320² for the S3 against espdet_pico.

**Decision (owner, 2026-11-25): the ESP32-S3 leaves the milestones.** It is
now a possible future project, not a goal. The target is the mobile browser.
Nothing done so far is wasted: the int8 models follow ESP-DL's rules (QAT,
power-of-two scales) and still give a 338 KB model that the Rust/WASM engine
runs at 16 GMAC/s, and `rust/KERNELS.md` documents what an S3 backend would
need. What changes:
- S3-only work is parked: firmware, TIE728 kernels, the Xtensa toolchain,
  and picking the 3-stage model as the S3 candidate.
- Model choices now optimise browser accuracy and speed.
- For the line detector, ESP-DL's per-tensor power-of-two rule is no longer
  a hard requirement; keeping the same int8 chain is simply the easiest path
  to the browser engine.

### Round 3 results (synth_v4, 30k steps)

| model | train time | best val CER (v4, 2k subset) |
|---|---|---|
| mnv4s 48 px | 1,963 s | 0.95% |
| mnv4s3 32 px (3-stage) | 1,354 s | 1.61% |

**Cross-evaluation, fp32, 3,000 lines of each val set:**

| 48 px model | on v3 val (clean) | on v4 val (official forms, art, bold) |
|---|---|---|
| trained on v3 | 0.63% | 3.07% |
| trained on v4 | 0.69% | **0.90%** |

v4 costs almost nothing on the old conditions (+0.06 points) and is 3.4x
better on the new ones, where the v3 model trips over the left-pointing ま,
the narrow せ, art beside the line and heavy strokes.

**QAT int8, 3-stage model** (`qat_v4_mnv4s3_h32`, 32x512): 1.75% on 512
held-out v4 lines (QAT weights without fake-quant: 2.06%), bit-exact in
Rust, 141 KB.

**QAT int8, 48 px model** (`qat_v4_mnv4s_h48`, 48x768): **0.84%** on 512
held-out v4 lines (QAT weights without fake-quant: 1.31%), bit-exact in
Rust, 402 KB `.espdl` / 338 KB `.hxhm`.

**Demo published as version 3:** the v4 48 px QAT model plus the WASM SIMD
engine (52 KB). In the browser pane, all 6 examples read correctly at
10-12 ms per box (19-24 ms when auto-fit tries two boxes), against about 47
ms per box with the previous engine, so about 4-5x faster in the browser (7x
in node). Phones are not measured.

## 2026-11-25: detector training pages, the vertical recogniser, romaji macrons

**Detector training set `data/pages_v2`** (`hxh-synth pages`, seed 1, 10
threads): 30,000 train + 1,000 val pages in 3 min (168 pages/s), 3.2 GB.
500,141 lines, 16.7 per page on average (max 56), 11.2% vertical, 905
art-only pages. Median line height 19 px (p1 9 px, p99 70 px). Preview:
`previews/pages_v2.png`.

**Vertical text, in the recogniser data path.** `LineDataset(orient="v")`
reads the vertical columns of a line set and rotates each one 90°
counter-clockwise on load, so the top glyph ends up on the left and the text
runs left to right like a horizontal line, with every glyph lying on its
side. Rotated glyphs collide with upright ones (for example ふ and う), so
vertical text gets its own model instead of sharing the horizontal one.
The orientation flows through the whole chain:
- `train.py --orient v` stores it in the checkpoint config, and
  `train_qat.py` / `eval.py` read it from there;
- `HXH_ORIENT=v` switches ESP-PPQ calibration, the hold-out set and the
  golden vectors (`spike_common.oriented_rows` / `read_gray`) to the
  rotated columns.

synth_v4 already had 150,576 vertical train columns and 3,030 val columns
(30%), the same length distribution as the horizontal lines. Preview of
rotated samples: `previews/vertical_rotated.png`.

**Vertical recogniser** `v4_mnv4s_h48_vert`: same recipe as the horizontal
48 px model (mnv4s, ImageNet weights, 30k steps, batch 128, lr 2e-3), on the
vertical columns only. 1,968 s.

| model | val CER (fp32, 2k subset) | int8 hold-out CER (512) | `.espdl` |
|---|---|---|---|
| horizontal `qat_v4_mnv4s_h48` | 0.95% | 0.84% | 402 KB |
| vertical `qat_v4_mnv4s_h48_vert` | **0.55%** | **0.71%** | 401 KB |

The vertical model came out better than the horizontal one. The reason is not
known yet: the two val sets differ in content (vertical columns use their own
layout and degradation draws), so this is not a like-for-like comparison.
QAT took 4,250 steps (626 s): it started at 1.31% int8 / 0.57% float and
ended at 0.60% int8. The Rust engine is bit-exact with ESP-PPQ on 512/512
held-out columns.

**Romaji: macrons for ー.** The owner tested a real line, ハンターライセンス
in Omniglot's glyphs, and the demo showed "hantaaraisensu". It looked as
if the lengthening mark had been misread as あ. The model was right
(はんたーらいせんす, ー at 100%), but the romaji helper spelled the long
vowel by doubling it. `to_romaji` (Python) and `toRomaji` (JS) now put a
macron on the vowel before ー: hantāraisensu, rāmen, sūpā. A second ー adds
nothing, and ー with no vowel before it stays "-". Demo republished as
version 4. While checking, `web/test/verify.mjs` turned out to have been
broken since the demo moved to the 48 px model, because its reference script
hard-coded 32x256. The script now reads the strip size from
`web/models/model.json`. It passes again: strips are bit-exact with
Python, 1.01% CER on 300 v4 lines, 8.0 ms per line in node.

## 2026-11-25: detector support in the Rust engine, and `hxh-detect` (built by a parallel agent)

**New ops, `.hxhm` format version 2.** Resize (nearest, integer factors) and Concat (2-64
inputs, axis H/W/C), plus a header flag for raw-output models (a detector: `hxh_run` returns
0 and the caller reads the int8 maps). The engine reads v1 and v2, and the converter still
writes v1 when a model needs nothing new, so recogniser files are byte-identical to before.

ESP-DL int8 semantics, read from esp-dl master and esp-ppq 1.3.15:
- Concat is a plain memory copy that ignores exponents.
- Resize computes `round(x * 2^(in_exp - out_exp))`. Scale 2 has its own kernel; other
  factors pick the source pixel with a float32 `int(y * (1/s))`, whatever ONNX's
  `nearest_mode` says.
- ESP-PPQ aligns both ops to their output exponent and inserts a RequantizeLinear in front of
  any input that keeps another one. The toy model shows both cases.
- The converter rejects unequal exponents, and accepts only scale factors where ESP-DL's
  float index equals `y // s` (all powers of two).

**Bit-exact before the real model existed.** A toy "detector-like" network (128x128 input,
[2, 64, 64] output, stride-2 stem, depthwise, Add + ReLU, Resize x2/x4/x8, a 6-input Concat)
was trained for 75 s and quantised by ESP-PPQ.
- Host: 12/12 vectors and 54/54 intermediate tensors bit-exact.
- Wasm: bit-exact in the SIMD, scalar and no-SIMD builds; the 5 real recognisers stay
  bit-exact on 512 vectors each.
- Random kernel self-test: 157,421 cases with 0 mismatches (a deliberately broken copy
  kernel is caught).

**`rust/hxh-detect`** (no_std, no allocator, no dependencies) turns the int8 maps into lines:
1. threshold in the integer domain (`sigmoid(q 2^e) > 0.3` becomes `q > t`);
2. 8-connected components;
3. convex hull of the pixel squares;
4. minimum-area rectangle by rotating calipers;
5. unclip (DBNet `A r / L`, or the exact inverse of the training shrink);
6. filters (score, size, centre on the page, not the letterbox padding);
7. orientation from the summed vertical logit;
8. quads clockwise from the top-left in reading orientation;
9. reading order (union-find blocks; columns right to left);
10. rectified crops (projective map, bilinear; columns turned 90 degrees counter-clockwise)
    and the letterbox itself.

Software `sqrt`/`exp`/`floor` make host and wasm bit-identical. Against OpenCV on random
maps (about 100k components): hulls identical, rectangles within 5e-5 px except equal-area
ties, letterbox within 1 grey level. Costs: post-processing a 320x320 map with 60 lines takes
0.7-2 ms, and letterboxing an A4 scan at 300 dpi takes 5-8 ms.

The JS API is `hxh_det_*` (`rust/DETECT.md`). It needs two instances of the same module, one
per model, because the bump allocator holds one model. The SIMD wasm grows from 54 to 80 KB
(31 KB gzipped).

**The open question it flagged, unclip:** at r = 0.4 a long line's kernel is only 0.16 h
thick. The exact inverse amplifies kernel errors 6x, and DBNet's 1.5 relies on a blurry
map. It had to be measured on the real model (next entry).

## 2026-11-25: the line detector `det_v1` (built by a parallel agent)

**Model.** A DBNet-lite on the ImageNet-pretrained `mobilenetv4_conv_small_050` (stem summed
to one channel), using only ops the int8 engine has:
- **Backbone:** features at strides 4/8/16/32 with 16/32/48/64 channels.
- **FPN-lite:** 1x1 laterals to 48 channels; top-down nearest Resize x2 + Add; per level a
  depthwise 3x3 + 1x1 down to 16 channels; Resize to stride 4; Concat to 64; 1x1.
- **Head:** 3x3 conv 64->32, Resize x2, 3x3 conv 32->16, 1x1 to 2 channels: the text-kernel
  logit and the vertical-orientation logit, at stride 2 (320x320 for a 640x640 letterbox).
  The threshold head is used in training only.
- ReLU everywhere, BatchNorm folded at export.

**Cost.** 350k parameters (344k deployed), 1.57 G MACs at 640x640:
- backbone 442 M;
- FPN 180 M;
- head 947 M, almost all of it in the two full 3x3 convs at high resolution.
A depthwise-separable head would remove about 0.8 G MACs.

**Training** (`scripts/train_det.py`, `runs/det_v1`): 36k steps, batch 16, 512 px crops of
pages_v2 (random scale, crop, rotation of +-5 degrees, no flips), 71 min on the shared GPU.
- **Loss:** BCE with 3:1 OHEM on the kernel map, L1 on the threshold map, Dice on the
  approximate binary map, and BCE on the orientation inside kernels.
- **Training recipe:** AdamW, EMA with BN recalibration.
- **Targets:** the numpy port of the Rust target rule matches `hxh-synth targets` exactly on
  the kernel, masks and orientation.
- **Not converged:** validation F1 was still rising at the end, so longer training would help.

**Full val, fp32** (1,000 pages, 16,748 lines, quad IoU >= 0.5):

| | value |
|---|---|
| precision / recall / F1 | 0.935 / 0.909 / **0.922** |
| orientation accuracy | 98.9% |
| recall: horizontal / vertical | 0.921 / 0.816 |
| recall by line thickness at 640: 6-9 / 9-14 / 14-25 / >= 25 px | 73% / 91% / 95% / 96% |
| end-to-end CER, horizontal (matched lines only) | 9.1% (4.9%) |
| end-to-end CER, vertical (matched lines only) | 15.3% (4.7%) |
| reader alone on ground-truth quads: horizontal / vertical | 3.8% / 3.2% |

Missed lines dominate the error. Most misses are boxes that exist but fall below IoU 0.5
(split lines, wrong sizes). The other half of the gap is the readers: about 3.5% CER on
lines cut from pages, against 0.9% on their own synthetic line set. Page lines are smaller,
tilted and framed differently from the training crops, so fine-tuning the readers on page
crops is the obvious next step. Figure: `previews/det_val.png`.

**int8** (ESP-PPQ, esp32s3, KL calibration on 64 pages; 100 held-out pages):

| | F1 | CER h | CER v |
|---|---|---|---|
| fp32 | 0.916 | 8.6% | 16.4% |
| PTQ | 0.904 | 9.7% | 16.5% |
| **PTQ + TQT** | **0.914** | 9.1% | 18.2% (222 lines, noisy) |

Plain PTQ lost 1.2 points of F1; TQT (28 min on the CPU) recovers almost all of it.

In the engine, every Resize is nearest x2 with equal exponents (a pure copy). The one Concat
has all inputs at the output's exponent. ESP-PPQ adds 5 RequantizeLinear ops. The output
exponent (-3) comes from calibration, so the app reads it from the model.

## 2026-11-25: full-page reading in the demo (interim detector)

`web/ocr.js` gained `HxhDetector`, `readPage` and `readAt`, all on the Rust exports.
`readPage`:
1. letterbox to 640;
2. int8 detector;
3. post-processing;
4. for every line, a crop of its quad grown by margins (`expandQuad`);
5. `prepareLine`, then the horizontal or vertical reader.

The page shows every line numbered in reading order, vertical columns in purple, and lists
the readings. Three gestures:
- **Tap a box** to see the line's details: uncertain glyphs and the strip the reader saw.
- **Tap unmarked text** to look closer: "tap a line" runs the detector again on windows
  around the tap (640, 1280, then 320 px of the original image), so small text is seen at
  native resolution.
- **Drag a box** as the old fallback: the box is read as one horizontal line.

Page lines the reader finds nothing in are dropped (in the example below, the one false
positive was an empty box). The single-file build and the Artifact page now carry three
models. Four generated val pages join the example lines.

**`web/test/page_e2e.mjs`** runs exactly the browser code path in node on generated pages
with ground truth (`web/test/make_pages.py`): IoU >= 0.5 matching, CER over every
ground-truth line (missed lines count as deletions). With the interim detector (step 14k,
int8) on pages_v2 val[900:1000] (1,657 lines), it settled the unclip question:

| post-processing | crop margins (along / across, x line height) | F1 | CER h | CER v | CER of matched lines |
|---|---|---|---|---|---|
| exact inverse, r = 0.4 | 0.3 / 0.2 | 87.7% | 14.0% | 28.7% | 5.6% |
| unclip 1.0 | 0.3 / 0.3 | 89.0% | 11.6% | 20.4% | 4.9% |
| **unclip 1.5** | **0.3 / 0.2** | **89.6%** | **10.3%** | **20.6%** | **4.4%** |
| unclip 1.5 | 0 / 0 | 89.6% | 23.1% | 26.2% | 17.3% |
| unclip 2.0 | 0.3 / 0.2 | 86.6% | 11.7% | 28.2% | 4.9% |

- DBNet's usual 1.5 wins: the predicted map at 0.3 is fatter than the hard kernel.
- Margins matter more than the unclip. The reader was trained on lines with space around
  the glyphs, and tight crops triple its error.
- Most of the remaining CER is missed lines (recall 87.6%, lower on vertical text). The
  reader also errs more on page crops (4.4%) than on its own synthetic lines (0.84%).

**Speed (node, Ryzen 7 3700X, one thread):**
- per page: 200-250 ms;
- detector: 75-100 ms at 1.57 G MACs (letterbox and post-processing included);
- per line: 7.5-10 ms.
The browser pane measured 130-190 ms to find the lines and 10-14 ms per line.

**Tap a line on a big photo.** One generated page was pasted small into a 3000x3000 canvas,
then tapped at the centre of each line:
- the page pass found only 4 of 14 lines;
- taps read 13/14 exactly, at 75-110 ms per tap, including a vertical column the page pass
  misread at its own resolution (はくぶつかん).
The page pass loses accuracy whenever it shrinks a page. Tiling big pages at native
resolution is a candidate next step.

## 2026-11-25: final detector in the demo, live camera mode, milestone done

**det_v1 in the browser.** `.hxhm` v2 (367 KB):
- bit-exact with ESP-PPQ on 16 held-out pages, on the host (the naive reference agrees) and
  in wasm, SIMD and scalar;
- 62 ms per page in node with SIMD (25 GMAC/s); the scalar fallback takes 683 ms.

Re-tuned in the browser pipeline (`page_e2e.mjs --sweep`, pages_v2 val[900:1000], every model
int8):

| post-processing | margins | F1 | CER h | CER v | CER of matched lines |
|---|---|---|---|---|---|
| **unclip 1.5** | **0.3 / 0.2** | **91.0%** | **9.2%** | **18.2%** | **4.4%** |
| unclip 1.0 | 0.5 / 0.3 | 90.3% | 10.7% | 17.7% | 4.4% |
| exact inverse, r = 0.4 | 0.3 / 0.2 | 89.5% | 11.0% | 26.5% | 5.2% |

Same winner as for the interim checkpoint. Against the interim model, F1 rises from 89.6 to
91.0%, CER h falls from 10.3 to 9.2%, and CER v from 20.6 to 18.2%. Artifact republished as
version 6.

**Live camera mode.**
- Each cycle grabs a frame (long side 960), detects, and reads up to 10 lines, the biggest
  first.
- A `LiveTracker` matches lines to tracks across frames by quad IoU. Each track shows the
  reading it got most often over its last 5 frames, and a line is only shown once seen twice.
- A line that has read the same text 3 times running reuses it for up to 2 frames instead of
  being read again, which cuts reading time on a still scene.
- Freeze, or a tap on a line, turns the current frame into the normal page view with every
  line's details.

**Where the camera works.** claude.ai Artifacts cannot open the camera, and this account has
no runtime capability for it. The page then plays a moving example page through
`canvas.captureStream()` as a stand-in camera, so the overlay can still be seen there. The
browser pane shows 2-2.5 frames/s with the CPU shared: detection about 195 ms, reading
145-210 ms for 10 lines. The real camera needs a normal https:// address.
**Hosted on Cloudflare, so the real camera works** (decided with the owner): the demo is
deployed as the assets-only Worker `hxh-ocr` at https://hxh-ocr.kokkakniphon.workers.dev
(public, HTTPS).
- `scripts/build_site.py` assembles `site/dist`: 23 files, 1.65 MB, the raw `.hxhm` models,
  no base64 copies.
- `site/` is a small `cf` project: `cloudflare.config.ts` and `cf@1.0.0-rc.5` pinned.
  - cf 0.15 could not deploy a folder.
  - `cf deploy ./dist` turns the folder into an assets-only Worker.
  - Steps: `cf auth login`, then `npm run deploy` in `site/`.
- The account's other ten Workers were listed first, to be sure the name was new.
- In the browser pane every file loads (HTTP 200) and the first example page reads 14 lines
  (177 ms to find them, 10.4 ms per line). The camera itself is untested here: the pane
  blocks it.

**Milestone status (full-page reading): done.**
1. page generator;
2. detector;
3. engine ops and post-processing;
4. demo with tap a line, full page and reading order.
Vertical text and a live mode came on top.

**Next, in rough order of value:**
- **Fine-tune the readers on lines cut from pages.** They score about 3.5% CER on page crops
  against 0.9% on their own lines.
- **Train the detector longer.** F1 was still rising at 36k steps.
- **Recover the misses.** They are mostly splits or wrong sizes below IoU 0.5; vertical
  recall is 0.82.
- **Tile big pages at native resolution.** The page pass found 4 of 14 lines on a 3000 px
  photo; taps read 13 of 14.
- **Make the detector head lighter.** Depthwise-separable 3x3 convs would remove about
  0.8 G MACs, for phones and live mode.
- **Real-image evaluation**, once a varied, labelled test set exists.

## 2026-11-25: the detector boxed every kind of writing; det_v2 learns to ignore it

**The problem.** The owner pointed the demo at the Omniglot page about the script. det_v1
boxed everything: English paragraphs, Japanese, even icons. The reader then turned each of
those boxes into confident-looking garbage kana. det_v1 had never seen any writing that was
not Hunter script, so to it, writing meant Hunter script.

**Stopgap: a reading-confidence filter** (commit a518147).
- `readingConfidence` is the mean character probability of a reading.
- Page lines under the threshold are hidden behind a "show them" toggle and drawn dashed
  grey when shown.
- Live mode shows only confident lines.
- It trims the noise but cannot fix it. At 0.85 on pages_foreign, precision only rises from
  8.0% to 36.5%.

**Negatives in the generator** (`rust/hxh-synth`, commit b499eac, `FOREIGN.md`).
- Pages now carry "foreign" text runs (Latin and Japanese) rendered with system fonts
  through `ab_glyph`, plus icon runs.
- A web and document layout was added, and some pages carry no Hunter script at all.
- Their quads are saved as `foreign` and kept out of the detector's targets: to the
  detector they are background.
- New sets:
  - **pages_v3**: 30,000 training and 1,000 validation pages.
  - **pages_foreign**: 300 pages, built to be hard and used only for evaluation.

**det_v2** (commit ce856b9).
- Fine-tuned from det_v1 on pages_v3 and pages_v2 together: 60,000 pages, 16k steps,
  learning rate 5e-4, 27 min (`train_det.py --init`).
- Quantised with PTQ + TQT: int8 F1 0.836 against 0.847 in fp32 on 100 held-out pages_v3
  pages.
- Bit-exact in the engine on 16 held-out pages.

The browser pipeline (unclip 1.5, margins 0.3 / 0.2) counts a false line as a shown line
that matches no Hunter line:

| set | detector | filter | P | R | F1 | false lines on text per page |
|---|---|---|---|---|---|---|
| pages_foreign (150) | det_v1 | none | 7.98% | 56.78% | 13.99% | 25.00 |
| | det_v1 | 0.85 | 36.45% | 55.63% | 44.04% | 2.39 |
| | det_v2 | none | 84.49% | 91.73% | 87.96% | 0.43 |
| | **det_v2** | **0.8** | **92.53%** | **90.16%** | **91.33%** | **0.15** |
| pages_v3 val (100) | det_v1 | none | 37.72% | 86.96% | 52.62% | 13.59 |
| | det_v2 | none | 84.57% | 81.88% | 83.20% | 0.80 |
| | **det_v2** | **0.8** | **91.42%** | **80.07%** | **85.37%** | **0.29** |
| pages_v2 val (100, all Hunter) | det_v1 | none | 92.09% | 89.92% | 90.99% | n/r |
| | det_v2 | none | 92.30% | 87.51% | 89.84% | 0.00 |

What the table shows:
- Recall on pages_foreign rises as well (56.8% to 91.7%): web and document layouts are no
  longer unfamiliar to the detector.
- The cost is a small regression on all-Hunter pages, F1 91.0% to 89.8%.
- With det_v2 the best threshold drops to 0.8. That is the deployed setting
  (`web/models/det.json`, `min_conf`).

## 2026-11-25: the blog, in two versions and three languages

The owner asked for a blog inside the demo site:
- styled after the datalab.to Indra OCR post;
- full of visuals;
- one version for readers who just want to know how it works, one research deep dive;
- then Thai, Japanese and a light/dark theme;
- then the explainer was aimed at middle-school readers, to get them interested in
  computer vision.

Hosted under `/blog/` on the same Worker.

**Stack.**
- Plain HTML, CSS and ES modules with no framework:
  - `blog.js` holds the shared charts, glyph drawing, the CTC figure, figure numbering and
    the translator;
  - `explain.js` and `nn.js` serve the explainer;
  - `deep.js` serves the deep dive.
- Fonts: Geist, Manrope and Geist Mono.
- Colours are tokens with light and dark values. `web/theme.js` cycles auto, light and dark
  and stores the choice.
- Every number in a figure comes from a file the project wrote:
  - `data/glyphs.json` holds the generator's own strokes;
  - `data/charts.json` holds the training curves;
  - `data/ctc.json` is a real int8 reading;
  - the pipeline images come from a real run.

**Explainer (`/blog/`).** Written for a 12-year-old. It is a tour from pixels to a reading:
- pixels as numbers you can toggle;
- a 3x3 filter sliding over a symbol, with Sobel and blur on the real pixels;
- layers;
- how fake training pictures are made;
- the guess, check, adjust loop, with the real validation curve;
- reading a line;
- finding lines;
- why the model is small and private;
- vertical text and live mode;
- what it cannot do;
- a word list.

Section ids must differ from figure ids: two clashes made figures render into headings.

**The neural network figures** (this commit, bb93a9b), asked for together with a better
"How it reads a line" figure.
- **A network you can poke.**
  - `scripts/blog/tiny_net.py` trains a real 256-12-8-6 ReLU network on 24,000 wobbly
    16x16 renders of あ い く ぐ と ど. It uses the glyph strokes and rasterises exactly as
    the page does: a 64 px canvas, round caps, area-averaged.
  - Held-out accuracy is 95.2% on 3,600 renders.
  - The page runs it live. Choosing a symbol sends a delayed wave through the edges; edge
    brightness is |weight × activation|.
  - Pixels can be painted and erased.
  - Focusing a first-layer neuron paints its 256 weights on the grid, blue for positive
    and purple for negative.
  - Erasing the dot of ぐ makes it say く, the same confusion the real reader's runner-up
    guesses show.
  - Under 520 px wide the network is drawn top to bottom.
- **Inside the real reader.**
  - `scripts/blog/nn_activations.py` runs the float reader (`runs/v4_mnv4s_h48/best.pt`)
    on the exact strip the int8 reader saw.
  - It exports, after the stem and each of the four block groups, the 4 channels with the
    most spatial variation. The shapes are:

    | stage | channels | height × width |
    |---|---|---|
    | stem | 32 | 24 × 384 |
    | 1 | 16 | 12 × 192 |
    | 2 | 32 | 6 × 192 |
    | 3 | 48 | 3 × 192 |
    | 4 | 64 | 2 × 192 |

  - It also exports the 64 × 192 per-step features and the softmax scores of the 14
    strongest classes.
  - For every step it records the span of input columns that holds the central 80% of
    |d score / d input|. The median span is 55 px, about one symbol.
  - The float argmax differs from the int8 one at 6 of 192 steps. Both decode to
    ぐりーどあいらんど.
- **CTC, step by step.** One cursor walks the strip and three rows fill in:
  1. what each step says, with one colour per symbol that is the same in every row;
  2. repeats glued together;
  3. blanks dropped. At the end the symbols close up into the reading.
- The CTC figure also has:
  - a 2x close-up with nine big step tiles and the attention span;
  - the top three guesses;
  - step, restart and slow controls.
- The 73 steps over the padding all say blank, so they are cut off: the figure is 1.6 times
  bigger on phones.

**Deep dive (`/blog/deep-dive`).**
- About 30 figures and 15 tables: architectures, the generator versions, training
  curves, the QAT and int8 engine internals, detector post-processing and the det_v1 vs
  det_v2 study.
- MathML for the CTC decode and the confidence formula.
- Written by an agent from the journal and the code, then checked.

**Thai and Japanese.**
- Each language has its own page, plus an `i18n-xx.js` dictionary of strings and patterns.
  A MutationObserver over `<article>` applies them to the text the figures draw.
- Two agents translated the pages:
  - Thai: 306 strings and 74 patterns.
  - Japanese: 350 strings and 56 patterns, です・ます in the explainer and である in the deep
    dive.
- `build_site.py` leaves the translated pages out unless `--translations` is passed.
  Language links to pages not published yet show "soon".
- Japanese and Thai text in captions fell back to a system monospace font. `--mono` now
  falls back to Noto Sans JP / Thai.

**Deploys.** Commits a3818c0 through bb93a9b; the English pages are live. The translated
pages wait for the new figure strings.

## 2026-11-25: first real photos, and tiled detection

The owner tried the Reader on a few of their own photos and screenshots of Hunter script.
They are other people's images and stay private: the pictures, their names, the text in them
and the per-photo results are not published. In general terms they showed four failures: text
only a few pixels tall after the 640 px letterbox of a 2000 px photo (the reader was right
whenever a line was found), lines on saturated colour panels, Japanese captions and round
icons next to the Hunter text read as Hunter script, and a narrow form of ー that the
generator did not draw.

**Tiled detection** (`web/ocr.js` `readPage({ tiles: true })`):
- **Tiles.** Besides the full-page pass, the detector looks at overlapping square tiles of about
  0.58 of the long side (640–1280 px, a quarter overlap).
- **Second look.** If the confidently read lines sit in less than half the frame, it then
  detects once more inside that region.
- **Merging.**
  - Lines overlapping by more than half of the smaller one are duplicates.
  - The closer look wins, unless it covers less than 60% of a same-thickness full-pass line;
    then the full line replaces the pieces.
  - Lines touching an inner tile edge come last.
  - Reading order is rebuilt.

(Per-photo results are not published.)

On the synthetic sets it helps too (browser pipeline, min_conf 0.8):

| set | F1 before | F1 with tiles |
|---|---|---|
| pages_v2 val | 89.9% | 90.9% |
| pages_v3 val | 85.4% | 86.4% |
| pages_foreign | 91.3% | 91.1% |

The extra passes cost 20–30 ms per page in node.

**Showing lines.**
- The detector's own score (the mean text-kernel probability of a line) separated most of the
  false hits from the real lines.
- A minimum score of 0.6 for shown lines cost recall on synthetic vertical lines (CER v
  20 → 28%).
- 0.55 is neutral on all three sets, so `det.json` now has `min_show_score` 0.55. Lines below
  it are hidden with the low-confidence ones.
- A stricter score for one-symbol readings cost 0.9–1.4 F1 on synthetic pages, so it was
  dropped.

**Next** (in progress, a background agent): photo- and anime-style training data and a model
family. The data adds:
- light-on-dark text;
- outlined, glowing and wavy glyphs;
- wide letter spacing;
- the narrow ー;
- screen-photo effects;
- colour panels;
- katakana captions and UI icons as negatives;
- possibly digits and punctuation.

The family is "tiny" (today's architectures, retrained) and "base" (bigger, for still photos).

## 2026-11-25: a model family (tiny, base) for real photos, and generator v5

Built by a background agent from the owner's brief after the first real photos (previous
entry): "This project can have many model size, and capability." Two members, each a detector,
a horizontal reader and a vertical reader, all int8 `.hxhm` checked bit for bit in the Rust
engine (host, wasm SIMD, wasm scalar), with jsons in the `web/models` format:
- **tiny** (`experiments/family/tiny/`): today's architectures retrained on the new data; the
  live-camera, phone and ESP32-class option.
- **base** (`experiments/family/base/`): bigger and more accurate, for still photos in the
  browser.

`web/models` is untouched; the owner integrates. Everything about the real photos stays local
(`data/real/user_photos`, git-ignored; scores only).

**Starting point** (deployed models: det_v2, v4 readers, `min_conf` 0.8, `min_show_score` 0.55).
On the new photo-style synthetic set (pages_v5, below) the browser
pipeline reached F1 75.5% with CER 29.8% (horizontal) and 67.2% (vertical).

### The family

| | tiny | base |
|---|---|---|
| detector | DBNet-lite, MobileNetV4-small 0.5, 640 letterbox (det_v2's architecture) | MobileNetV4-small **1.0** (32/64/96/128 ch), FPN 64, levels 24, fuse 96, depthwise-separable head, **960** letterbox |
| detector size | 350k params, 1.57 G MACs | 1.18M params, 4.44 G MACs |
| readers (h, v) | MobileNetV4-small 0.5, 48x768 (today's) | MobileNetV4-small **1.0**, 48x768 |
| reader size | 324k params, 154 M MACs / line | 1.18M params, 555 M MACs / line |
| vocab | 90 tokens (kana, space, 0-9 ! ? : / . -) | same |

**Training.**
- *tiny detector* `det_tiny_v3`: det_v2 fine-tuned 8k steps (lr 5e-4, batch 16, 512 crops) on
  pages_v2 / pages_v3o / pages_v5 sampled 25 / 25 / 50% (`--data-weights`), selected on pages_v5
  val.
- *base detector*: ImageNet -> pages_v2 + pages_v3 at 960 (24k steps; F1 0.946 on pages_v2 val
  at step 22k, against det_v1's 0.922 at 640), then `det_base_v1`, 7k steps on the same 25/25/50
  mix (lr 1e-3, batch 12, 640 crops of pages scaled to 576-1920 px).
- *readers*: lines_v5b (the fixed cut, 14k pages, ~100k lines) + synth_v4 (350k h / 150k v
  lines, the old distribution). tiny: from the v4 readers, 6k steps, lr 1e-3, the classifier grown
  from 75 to 91 classes (old rows kept). base: ImageNet -> synth_v4 (14k of 20k steps, val CER
  0.99%), then 9k steps on the mix; the vertical base reader starts from the horizontal
  pre-train.
- *int8*: detector QAT (2k steps, below) and reader QAT (4,250 steps on lines_v5b + synth_v4),
  then ESP-PPQ PTQ (esp32s3, KL) -> `.hxhm` -> bit-exact checks.

**Files.** `experiments/family/<tiny|base>/`: `det.hxhm`, `model.hxhm` (horizontal),
`model_v.hxhm` (vertical) and `det.json`, `model.json`, `model_v.json` in the `web/models`
format (vocab in the model json, `min_conf` 0.8, `min_show_score` 0.55), written by
`scripts/family_json.py` from `notes.json`. Every model is bit-exact in the host engine (512
golden lines per reader, 16 pages per detector, plus the naive reference) and in wasm SIMD
and scalar (`test_wasm.mjs`). The whole run is `scripts/launch_family.sh`, stage by stage.

### Generator v5 (`rust/hxh-synth`, all options off by default)

Built from the failure list of the first real photos. One implementation serves both the
detector pages and the reader lines (below), so the reader sees exactly the styles the detector
is trained on.

**Scenes** (`scenes.rs`, new `--mode-p` weights 5-7):
- **card**: "Hunter Vocabulary" name cards. Panels in the grey values of saturated colours
  (purple 0.42, pink 0.52, blue 0.36, yellow 0.86 ...), huge outlined Latin letters behind
  ("HUNTER", clipped by the card), a bold Hunter name, its **katakana right under it**, the romaji under that, a header ("Hunter Vocabulary" + ハンター語講座),
  character art on one side.
- **lyric**: a lyric video. Near-black frame, often a portrait column with dark side bars;
  1-3 white lines with **letters 0.6-3 em apart** (75%), sometimes a glow or shadow; a kanji / kana
  caption right under 85% of them; app UI around (round buttons with counts, an avatar with a
  "+" badge, @user, a hashtag line, tabs, a progress bar).
- **tv**: an anime frame with screens in the scene: white glyphs with a dark outline on a blue
  panel, bubble-font titles (thick hollow glyphs, light fill, darker outline), a news ticker,
  white glyphs on a dark shirt blob, and times or numbers beside a title ("3/11 15:00", "225").

**Photographed off a screen** (`photo.rs`, `--photo-prob`): the page becomes the screen content
of a phone photo. Black level 0.02-0.2 and white 0.7-1.0; the screen fills 35-95% of a
portrait / landscape frame (long side up to 2000 px), with keystone (±12%), corner jitter and
±3° rotation, a dark bezel, wall and desk blobs around it; moiré bands (50%), glare (50%), a
reflection gradient (25%), vignetting (65%), uneven exposure, defocus (70%) or shake (12%),
gain and gamma (a colour cast is only a brightness change in grayscale), sensor noise and JPEG
35-92. Every quad goes through the same homography.

**Hard negatives** (`--ui-neg-prob`, `--caption-prob`): avatars (discs with blobs and eyes,
rings, badges), round icon buttons (an icon-font glyph or a drawn +, ×, ≡, ▶, ⋯ inside: ⊕ is
ほ), button rows, a browser-chrome or toolbar strip with an address and icons, and a Japanese
or Latin caption placed 0.31-0.9 line heights under or above free Hunter text. All recorded in
`foreign` (icons and avatars as script "icon") and kept clear of labelled lines.

**Text styles** (`render.rs`, `sample_style_v5`): wide spacing up to ~3 glyph widths (7%, and
75% of lyric lines), square ends with mitred joins (15%), serif ticks at stroke ends (7%), a
wavy warp that bends whole glyphs (6%), weights as before (0.04-0.30 em). For free-standing
text (`v5_effects`): glow (light ink on dark), a soft dark outer stroke, bevel, and hollow
glyphs (the outline carries the contrast, the strokes take the background's value).

**The narrow ー** (`hxh_ocr/glyphs.py`, exported): two new forms after a fan font, about 0.55 em tall with a short bar (20%) or almost no bar (10%), next to the
chart's >—< (70%).

**Inline digits and punctuation** (`--inline-prob`): `0-9 ! ? : / . -` set in an open font
inside horizontal Hunter lines (end 60%, start 20%, middle 20%; "!", "?!", "...", "225",
"3/11", "15:00", "99.9", "10-2"), cap height matched to the glyphs. Labels keep them.

**Reader lines from pages** (`hxh-synth lines`): each labelled line of a generated page is cut
exactly the way the browser cuts it: the quad jittered like a detector box (corners ±4% of the
line height, a little fatter and longer), grown by 0.15-0.45 / 0.1-0.3 line heights like
`expandQuad`, rectified with `hxh_detect::crop`. Vertical columns are stored upright. A quarter
of the pages also give one **negative crop** (non-Hunter writing or an icon) with an empty
label, so the reader learns to say nothing there. This is the "move line generation into Rust"
item: the Python line generator stays for synth_v4, and the Rust one comes with every page
style, photo effect and neighbour for free. Speed: about 25 pages (170 lines) a second on 6
threads with the CPU shared.

**Fonts: openly licensed only** (the coordinator's constraint from the open-source audit: the
datasets will be published under CC BY 4.0, and pages_v3 / pages_foreign drew their distractors
from Windows fonts whose terms for rendered output are unclear). Since v5 every non-Hunter run
comes from `fonts/open` (git-ignored), 41 faces:
- OFL-1.1: Noto Sans, Noto Sans JP / KR / TC (4 weights) / CJK SC, Noto Sans Symbols 2, Noto
  Emoji (monochrome outlines), Go Noto Kurrent (2), Lato (5), Inter, Inter Tight, Font Awesome
  4.3 (icons);
- Apache-2.0: Roboto (8 styles), Roboto Slab (2), Roboto Mono, Open Sans (4), GenEi Gothic
  Pro (5 weights, a Source Han Sans derivative, Apache-2.0 as stated in the font), Droid Sans
  Fallback.
The copies came from other software installed on this machine that bundles these fonts
(games launchers, a 3D package, developer docs), WOFF2 decompressed with fontTools. Nothing was
downloaded. Each file was checked by its own licence fields (name IDs 13 / 14), except seven
that carry no licence text of their own: six Roboto files from 2011 (Roboto is Apache-2.0
upstream) and Font Awesome 4.3, whose field only links to its licence page (the font files are
OFL-1.1); for those the upstream licence was checked instead (corrected 2026-11-25 by the
release tooling's font audit). Three candidates whose own fields gave other terms were left out
(Droid Sans Mono: an Ascender EULA, Brutal Type, Motiva Sans). `rust/hxh-synth/fonts.lock.json` pins name,
version, licence string and sha256 of every file used, and `scripts/fetch_fonts.py` can fetch
the official files from google/fonts, notofonts/noto-cjk, go-noto-universal and Font Awesome
(they may be newer versions, so their hashes can differ). **Font Awesome 4.3 stands in for
Material Symbols** (Apache-2.0), which is not on this machine. No open handwriting or mincho
face is available yet, so those two looks are missing from v5. `pages_v3o` (30k + 1k) and
`pages_foreign_open` (300) re-render the pages_v3 and pages_foreign recipes with these fonts;
they can replace the old sets in the Hugging Face release. The list is also in
`rust/hxh-synth/FOREIGN.md`.

**Robustness.** pages and lines now write their labels in chunks of 2000 pages and can
`--resume 1` (below: what went wrong). A v5 test checks reproducibility, disjoint quads inside
the image and the keep-clear rule on 80 pages with every option on.

### Decision: the readers' vocab grows to digits and ! ? : / . -

Both family readers are trained with 90 tokens (91 classes with the CTC blank): the 74 kana and
space first, with the same ids as before, then `0123456789!?:/.-`. Real frames mix them in
("3/11 15:00", "225", "!?" on the news ticker); with the old vocab a "!" right after the last
glyph is inside the detector's box and comes out as a kana.

The trade-off:
- **Confusable shapes.** そ is two stacked rings (8), お a ring (0), い a big dot (.), ん a dot
  and a slash (/), the narrow ー has a short bar (-). The reader has to tell them apart by
  stroke style and size, so pure Hunter lines can lose a little. Measured below on synth_v4 val
  (no digits at all): CER and "digits or punctuation read where there are none".
- **ESP32 path.** The classifier grows from 75 to 91 outputs, padded to 96 instead of 80 in
  ESP-DL's 16-channel blocks: 64 x 96 instead of 64 x 80 MACs per timestep, under 0.1% of the
  154 M MACs of a line. The int8 chain needs `--num-classes 91` (`qat_pipeline.sh NC=91`).
- **Browser.** The app reads the vocab from the model json (`vocab`) and the class count from
  the model, so nothing else changes; `toRomaji` passes digits and punctuation through.
- **Labels.** `normalize()` still maps ‐ － − ～ 〜 to ー; the ASCII "-" stays "-".

### Detector int8: plain PTQ broke the base detector; detector QAT

(See "what went wrong" for the diagnosis.) `hxh_ocr/qat.py` now covers the detector graph:
nearest Resize is transparent (a copy at one exponent) and Concat is an "Align to Output" site
like Add. `scripts/train_det_qat.py` fine-tunes under fake quantization (kernel BCE-OHEM, Dice,
vertical BCE; exponents recalibrated every 250 steps until 75% of the run; selection on int8
F1), exports the folded model, and plain ESP-PPQ PTQ (KL) follows, the recogniser recipe.

| detector | float F1 | int8 F1 at the start (fake quant) | after 2k QAT steps | ESP-PPQ int8, pages_v5 hold-out |
|---|---|---|---|---|
| tiny (det_tiny_v3) | 0.770 | 0.716 | 0.749 | 0.743 |
| base (det_base_v1) | 0.845 | 0.520 | 0.822 | 0.802 |

The fake-quant estimate and ESP-PPQ's own simulation agree within 1-2 points (different page
subsets), so the QAT model is faithful for these graphs too. For the base detector, PTQ with
cross-layer equalization reached 0.73 and plain PTQ 0.50 (on the pre-trained checkpoint, same
hold-out); QAT closes most of the gap.

### Readers

fp32 checkpoints, `scripts/eval.py`: lines_v5b val (7,444 lines cut from 1,000 v5 pages; 6,589
horizontal Hunter lines, 650 vertical, 205 negatives) and the first 3,000 lines of synth_v4 val
(the old distribution, no digits). "Negatives read" = share of non-Hunter crops (empty label)
read as text.

| reader | lines_v5b CER | negatives read | synth_v4 CER | digits / punct. on plain lines |
|---|---|---|---|---|
| current h (v4, 75 classes) | 18.2% | 92% | 0.90% | - |
| tiny h | 9.6% | 27% | 1.38% | 0.05 per 1000 chars |
| base h | **8.3%** | **20%** | 1.11% | 0.18 / 0.05 per 1000 chars |
| current v | 12.2% | 100% (8) | 0.56% | - |
| tiny v | 8.5% | 13% (8) | 0.75% | 0 |
| base v | **8.1%** | 25% (8) | 0.76% | 0 |

- The new data halves the error on photo-style lines and cuts the "reads anything as kana" rate
  from 92% to 20-27%. The old v4 reader cannot write digits or punctuation, which is part of its
  18%.
- The price is on the old distribution: +0.2 to +0.5 points of CER on synth_v4 (int8 below).
  synth_v4 is three quarters of the training lines, so this is the cost of sharing capacity,
  not of missing data.
- The bigger vocab does not leak: on 3,000 lines with no digits or punctuation, 0-1 such
  characters were read.
- The base vertical reader is only a little better than the tiny one: it started from the
  horizontal pre-training and had 9k steps.

**int8** (QAT 4,250 steps on lines_v5b + synth_v4 from each fp32 checkpoint, ESP-PPQ PTQ KL;
CER on ESP-PPQ's 512 synth_v4 hold-out lines; every model bit-exact 512/512 in the Rust engine
and against the naive reference):

| reader | int8 CER, synth_v4 hold-out | greedy identical to fp32 | .espdl |
|---|---|---|---|
| current h (`qat_v4_mnv4s_h48`) | 0.84% | 93.9% | 402 KB |
| tiny h, QAT | 1.61% | 91.8% | 404 KB |
| tiny h, equalized + QAT | 1.59% | 93.6% | 404 KB |
| base h, equalized + QAT | 1.11% | 93.4% | 1.27 MB |
| current v (`qat_v4_mnv4s_h48_vert`) | 0.71% | 90.4% | 401 KB |
| tiny v, QAT | 1.64% | 87.3% | 404 KB |
| tiny v, equalized + QAT | 3.64% | 85.0% | 404 KB |
| base v, equalized + QAT | **0.82%** | 95.3% | 1.27 MB |

- **The base readers needed equalization.** Plain QAT of the width-1.0 reader started at 100%
  int8 CER (fake quant, mixed val), was at 6.4% after 3,500 steps and gave 2.95% on the
  hold-out after PTQ: the same flattened depthwise channels as the base detector.
  `hxh_ocr/qat.py equalize()` (Nagel et al. 2019, on the BN-folded model: each conv -> ReLU ->
  conv pair is rescaled per channel so both ranges meet, exact in fp32, max |diff| 4.7e-5 on 34
  pairs) brought the int8 start to 5.9% and the end to the table.
  The first version found only 12 pairs because torch.fx puts Identity / Dropout nodes between
  the convs; it now looks through them.
- For the tiny readers equalization does not help: 1.61 -> 1.59% horizontal (one character in
  the hold-out), 1.64 -> 3.64% vertical. The family keeps the plain QAT pair.
- On the old synthetic distribution the family readers are behind the deployed ones in int8
  (tiny +0.8-0.9 points, base +0.3 / +0.1); on photo-style lines they halve the error (fp32
  table above, and the browser numbers below).

### Tiny vs base vs the deployed models

All numbers below are the browser pipeline (`web/ocr.js readPage` in node, int8, WASM SIMD),
reading confidence 0.8 and detector score 0.55 as in the app (`scripts/family_eval.sh`,
`scripts/family_table.py`; raw output in `experiments/family/eval/`). CER h / v count missed
lines in full; "matched" is the CER of the lines that were found. False lines are shown lines
on no Hunter line, split by what they sit on.

| set | models | F1 | CER h | CER v | CER matched | false lines / page (text, icons, other) |
|---|---|---|---|---|---|---|
| pages_v2 val | deployed (det_v2, v4 readers) | 89.5% | 12.7% | 22.8% | 3.0% | 0.00, 0.00, 0.67 |
| pages_v2 val | **tiny** | 85.9% | 16.6% | 35.8% | 3.0% | 0.00, 0.00, 0.71 |
| pages_v2 val | det_v2 + tiny readers | 89.5% | 12.4% | 22.6% | 2.7% | 0.00, 0.00, 0.67 |
| pages_v2 val | **base** | 91.7% | 8.3% | 16.7% | 2.3% | 0.00, 0.00, 0.56 |
| pages_v3 val | deployed (det_v2, v4 readers) | 85.6% | 18.9% | 24.2% | 3.4% | 0.19, 0.06, 0.40 |
| pages_v3 val | **tiny** | 83.4% | 22.9% | 25.6% | 3.1% | 0.10, 0.01, 0.40 |
| pages_v3 val | det_v2 + tiny readers | 86.2% | 18.3% | 24.4% | 3.3% | 0.14, 0.08, 0.37 |
| pages_v3 val | **base** | 91.1% | 10.5% | 14.4% | 2.7% | 0.10, 0.05, 0.32 |
| pages_v3o val (open fonts) | deployed (det_v2, v4 readers) | 85.8% | 18.9% | 21.9% | 3.8% | 0.38, 0.02, 0.28 |
| pages_v3o val (open fonts) | **tiny** | 83.6% | 22.4% | 28.1% | 3.7% | 0.10, 0.00, 0.35 |
| pages_v3o val (open fonts) | det_v2 + tiny readers | 86.7% | 17.9% | 22.4% | 3.9% | 0.26, 0.01, 0.31 |
| pages_v3o val (open fonts) | **base** | 91.3% | 10.9% | 14.8% | 2.9% | 0.05, 0.00, 0.30 |
| pages_foreign | deployed (det_v2, v4 readers) | 91.5% | 7.9% | 17.1% | 1.9% | 0.13, 0.09, 0.07 |
| pages_foreign | **tiny** | 91.9% | 10.4% | 24.6% | 1.7% | 0.04, 0.01, 0.07 |
| pages_foreign | det_v2 + tiny readers | 93.0% | 7.5% | 18.7% | 1.8% | 0.06, 0.05, 0.07 |
| pages_foreign | **base** | 95.6% | 4.0% | 3.2% | 1.3% | 0.07, 0.06, 0.06 |
| pages_foreign_open | deployed (det_v2, v4 readers) | 92.5% | 10.7% | 15.5% | 2.9% | 0.10, 0.11, 0.06 |
| pages_foreign_open | **tiny** | 92.1% | 13.1% | 21.8% | 2.5% | 0.05, 0.02, 0.07 |
| pages_foreign_open | det_v2 + tiny readers | 93.8% | 10.1% | 16.1% | 3.1% | 0.07, 0.05, 0.05 |
| pages_foreign_open | **base** | 96.0% | 4.7% | 7.5% | 1.9% | 0.09, 0.05, 0.03 |
| pages_v5 val (photo-style) | deployed (det_v2, v4 readers) | 75.5% | 29.8% | 67.2% | 5.2% | 0.14, 0.13, 0.34 |
| pages_v5 val (photo-style) | **tiny** | 76.2% | 30.1% | 68.6% | 4.0% | 0.04, 0.01, 0.15 |
| pages_v5 val (photo-style) | det_v2 + tiny readers | 76.3% | 28.8% | 66.7% | 4.7% | 0.09, 0.10, 0.34 |
| pages_v5 val (photo-style) | **base** | 83.2% | 20.9% | 57.8% | 3.4% | 0.06, 0.03, 0.23 |

- **base is the better model on every set**: +2 to +7.7 points of F1 over the deployed models
  (the most on the photo-style pages), CER of the found lines down by a fifth to a third, and
  fewer false lines on non-Hunter writing.
- **The tiny readers are a drop-in gain.** Behind det_v2 (row "det_v2 + tiny readers") they
  match the deployed readers on pages_v2 and gain 0.6-1.5 points of F1 everywhere else, with a
  lower CER of the found lines on 4 of 6 sets.
- **The tiny detector is a trade-off.** det_tiny_v3 halves the false lines on pages_v5 (on
  writing, icons and elsewhere) but loses recall on the old manga and frame pages (pages_v2
  -3.6 F1 against det_v2, pages_v3 -2.8).
- Vertical text on photo-style pages stays weak for every model (CER v 58-69% on pages_v5,
  mostly missed columns: the int8 detectors find 31% (tiny) and 43% (base) of the 61 columns
  in the pages_v5 hold-out).

Results on the owner's private photos are not published.

### Speed (node 24, WASM SIMD, one thread, Ryzen 7 3700X, machine otherwise idle)

| | deployed | tiny | base |
|---|---|---|---|
| detector pass | 60 ms (1.57 G MACs) | 61 ms (1.57 G MACs) | 190 ms (4.44 G MACs, 23 GMAC/s) |
| reader, per line | 6.0 ms (154 M MACs) | 6.0 ms (154 M MACs) | 19.7 ms (555 M MACs, 28 GMAC/s) |

Model sizes: tiny 0.37 + 0.34 + 0.34 MB, base
1.21 + 1.21 + 1.21 MB of `.hxhm`.

**Decision for the owner: tiles with base.** At 960 px the base detector's full-page pass
already sees text that needed tiles at 640, and tiles cost it about 2.5x the time, so base
reads without tiles (recommended; checked on the private photos); tiles stay useful for the
640 detectors.

**Decision for the owner: which 640 detector.** The tiny readers improve on the deployed ones
behind either detector. For the detector:
- (a) keep det_v2 and swap only the readers (`model.hxhm`, `model_v.hxhm` and their jsons from
  `experiments/family/tiny/`): never worse on the synthetic sets, recommended for the live camera, where text fills the frame.
- (b) det_tiny_v3 (`experiments/family/tiny/det.hxhm`): fewer false lines on photo-style
  frames, but -3.6 / -2.8 F1 on the old manga and frame pages.

### What went wrong on the way

- **The machine hit its commit limit, three times.** Two generators, two trainers with loader
  workers, a node benchmark and the owner's own sessions together reached the Windows commit
  limit (60 GB): the Rust generators died with "memory allocation failed", the detector's
  loader workers with error 1455 and the reader with a CUDA out-of-memory. A 30k-page pages_v5
  run died after its last page but before writing labels.jsonl (13 min of work), and the line
  cutter lost 27k pages. Fixes: labels are now written in chunks of 2000 pages with
  `--resume 1`; at most two trainers at a time, 2-3 loader workers each; the base reader's
  ImageNet pre-training was stopped at step 14k (val CER 0.99%) and its best checkpoint used.
- **An inverted rectangle panicked a v5 scene** (`Gray::fill` with x1 < x0, from a screen or
  ticker rect clipped by the page edge). `fill` and `mark` now ignore empty rects.
- **Plain PTQ broke the base detector.** fp32 F1 0.81 -> int8 0.50 on pages_v5 (recall 0.34,
  vertical lines all lost; output SNR 19.6 dB against det_v2's 26.5 dB). Not the new
  depthwise head: per-layer weight statistics of the BN-folded ONNX show the width-1.0
  ImageNet backbone's depthwise convs with almost every channel under 4 quantisation levels
  (stages.2.2: 190 of 192 channels, median 0.5 levels; stages.2.3: 188 of 192), against at
  worst 212 of 256 (median 1.8) in det_v2. Per-tensor power-of-two weights flatten them.
  Fix: ESP-PPQ's layerwise (cross-layer) equalization, which rescales each conv -> ReLU -> conv
  pair so the channel ranges balance (exact in fp32): `quantize_det.py --equalize`.
- **Editing a shell script while bash runs it** corrupted the running copy (bash reads scripts
  as it goes); the bit-exact step of that run had to be repeated by hand.
- **Invisible labelled lines.** The first reader runs on lines_v5 had a floor of hopeless lines:
  the worst 32 of the validation set were blank crops with a label. The hollow style drew a
  light outline on a light background when the ink was light (the strokes take the
  background's value, so nothing was left). Fixed in `v5_effects` (the outline takes the old
  ink, the fill moves to the other side when it is not the background), and two guards now keep
  such lines out whatever the cause: `hxh-synth lines` drops a crop whose 5th-95th percentile
  spread is under 28 grey levels, and the detector loader puts such lines in the ignore mask.
  lines_v5 was regenerated as lines_v5b, and the pages_v5 test set was rebuilt with
  `make_pages.py --drop-illegible` (13 lines dropped).
- **The tiny detector's fine-tune cost recall on the old pages.** Already in fp32 (pages_v2 val
  F1 0.896 against det_v2's 0.911), so it is the mix, not int8: half of the batches are v5
  pages now. In the browser pipeline, with the same readers behind both, it loses 3.6 points of
  F1 on pages_v2 and 2.8 on pages_v3 against det_v2 and is level on pages_v5 (with half the
  false lines). TQT instead of QAT gave the same int8 numbers.
- **Equalization is not free for small models.** For the tiny vertical reader the equalized QAT
  ended at 3.64% instead of 1.64%; the family keeps the plain QAT readers.
- **A CUDA "illegal memory access"** killed the first tiny detector run at the moment another
  CUDA process was stopped; rerun with 8,000 steps.
- **The base horizontal reader's QAT died at step 2,700 of 4,250** (commit limit again, error
  1455 in the loader's shared memory) and was finished from its best checkpoint (step 2,000,
  fake-quant val 4.42%) by `scripts/qat_finish.sh`.

### Open

- No real-image benchmark is published; the private photos are anecdotes, not a benchmark.
- Vertical columns in photo-style frames: the detectors find a third to under half of them.
  More vertical v5 scenes (signs, title cards) and a vertical-heavy mix would be the next step.
- The tiny readers trail the deployed ones on synth_v4 in int8 (1.6% against 0.7-0.8%);
  equalization did not help. A longer QAT or a QAT mix weighted towards synth_v4 might.
- The base detector loses about 4 points of F1 to float in int8 on pages_v5 (0.845 -> 0.802);
  `train_det_qat.py --equalize` (QAT from an equalized model) is untried.
- pages_v5 should mark illegible lines at generation time (the loader does it now).
- A handwriting and a mincho open font, and Material Symbols, would round out the font set;
  `scripts/fetch_fonts.py` (official sources, pinned) has not been run: nothing was downloaded.
- The Hugging Face card of `hxh-ocr-pages` still describes v3 / foreign rendered with Windows
  fonts; pages_v3o and pages_foreign_open are the open-font replacements if the release
  should switch.

## 2026-11-25: Fast and Accurate in the Reader

The model family went into the site as two sets, following the family agent's recommendations
(above):

| set | folder | detector | readers | size | still pictures |
|---|---|---|---|---|---|
| Fast | `web/models` | det_v2 (640) | qat_fam_tiny_h48, _vert | 1.05 MB | page pass plus tiles and the region second look |
| Accurate | `web/models/accurate` | det_base_v1_qat (960) | qat_fam_base_h48, _vert | 3.63 MB | page pass only (tiles cost ~2.5x the time) |

- **Why det_v2 stays in Fast and det_tiny_v3 does not ship.** det_v2 with the tiny readers is
  never worse than the old deployed set on the synthetic page sets; det_tiny_v3 lost 3.6 and 2.8 points of F1 on pages_v2 and pages_v3.
- **The live camera always uses Fast**; the switch (Fast | Accurate, under the picture) only
  picks the models for pictures. The choice is remembered in localStorage (`hxh-quality`);
  Accurate is the default. Fast loads first, so the live camera and a Fast read are ready as
  before; Accurate downloads right after it, and a page read on Accurate waits for it ("Loading
  the accurate models…"). If it fails to load, the switch disables Accurate and Fast is used.
  A tapped or boxed line is read by the set that is chosen, and each line remembers which set
  read it (vocabulary, confidence cut, the vertical reader in the details).
- **Check that the copied files are the benchmarked ones.** page_e2e on pages_v2 val with the
  files in `web/models` and `web/models/accurate`: F1 89.45% (Fast) and 91.65% (Accurate) at
  conf 0.8 / score 0.55, the same as experiments/family/eval/v2tiny_pages_score.txt and
  base_pages_score.txt. `verify.mjs` passes with the new reader.
- **In headless Chrome** (scripts from the anatomy skill's drive.mjs; the app's browser pane was
  hidden and does not decode images then): example page 1 (14 lines) gives 14 lines on Accurate
  (573 ms to find, 502 ms to read) and 11 on Fast (93 + 112 ms). The Thai and Japanese pages show
  the switch, the busy text and the per-set model lines translated; no horizontal scroll at 390 px.
- **Text that changed.** The Reader's lede (models now 4.7 MB in all), the "How it works" text
  (960 px with Accurate; digits and `! ? : / . -`; a "Fast or Accurate" paragraph), the
  unsure-line note (no longer says "numbers"), the footer (one line per set), in English, Thai
  and Japanese. The blog articles describe the first models (75 classes, det_v2), so each of the
  six pages got a dated update note under its standfirst instead of a rewrite.
- `build_site.py` ships `models/accurate/`; `make_artifact_page.py` was re-run, so the base64
  copies for the claude.ai page are the new Fast models (that page has no Accurate set and shows
  Accurate disabled). The single-file build also carries only Fast.

## 2026-11-25: more private test photos

The owner added four more of their own photos to the private test set. Like the others they
are not published: no pictures, file names, text read from them or per-photo results. In
general terms they showed three things to work on: very wide letter spacing makes the 640 px
detector box single glyphs instead of the row (joining boxes on one baseline with the same
height and a small gap, then reading the joined line, would fix most of it in
post-processing); browser and app icons can still read as kana with high confidence; and
small text in a big photo needs tiles, which the Accurate set does not use.

## 2026-11-25: release tooling for the model family

A parallel agent rebuilt the open-source release around the family (d10d091, 3cae7ad); nothing
was pushed or uploaded.

- **Hugging Face model repo** (`release/hf-model`, 60 files, 79 MB): `int8|espdl|onnx|pytorch`
  each split into `v4/` (the first models, 75 classes), `tiny/` and `base/` (91 classes), plus
  `presets.json` (Fast = det_v2 + tiny readers, with tiles; Accurate = base, no tiles) and both
  vocabularies. Checks: every int8 file equals its experiment output and `web/models` /
  `web/models/accurate` match the presets byte for byte; each reader `.espdl` reconverts to the
  shipped `.hxhm`; stripped checkpoints load with `weights_only` and give identical outputs; the
  QAT models and det_v2 match their ONNX within 8e-5. Of the 79 MB, 44 MB is float PyTorch
  (before and after QAT) and 22 MB float ONNX; the int8 files are 5.6 MB.
- **Datasets, open fonts only.** pages_v2 uses no fonts (generated before hxh-synth had font
  support) and synth_v4 draws only traced strokes; the v5 sets, pages_v3o and pages_foreign_open
  use only the 41 faces in `fonts/open`. The packager refuses pages_v3 / pages_foreign and any
  page naming another font. Pages configs: `v3o` (default), `v2`, `foreign_open`, `v5` (with a
  `photo` column); lines configs: `synth_v4` (default), `v5`. v5 ships train + validation, and
  the card says validation[900:1000] are the benchmark pages. Every split was checked (row
  counts, labels, ~200 images byte-equal to the source, `datasets` decodes them). Total
  14.44 GB: lines 3.52 GB, pages 10.92 GB, of which the v5 data is about 4.65 GB.
- **Font licence fields.** The audit found that seven of the 41 files carry no licence text of
  their own (six Roboto files from 2011, Font Awesome 4.3); the font note above and
  `rust/hxh-synth/FOREIGN.md` now say so, and no longer name the applications the copies came
  from.
- **Public export** (`release/hxh-ocr`, 748 files, 120 MB): the scripts that read the private
  photos are left out, every journal passage about them is rewritten to a neutral line (the
  export fails if a rewrite rule stops matching), and a new audit blocker rejects photo file
  names, label text, copies of `data/real` files and photo-result tables. Against the raw journal
  it raises 16 blockers; against the export, none.
- **Open questions for the owner:** the 14.44 GB total (approved: about 10 GB); qualitative
  "reads real photos well" wording in `base/det.json` and the blog notes; the v5 card scene's
  header text, which follows a fan-made card series.
- **Owner's answers (same day):** publish all 14.44 GB; Hugging Face namespace `KokkakNiphon`
  (filled into the cards, README and citation links with `--namespace`; the export audit still
  passes); keep the "reads real photos well" wording. Publishing itself waits for the owner's
  go-ahead.
- **Credit for the weights (owner's choice).** The owner wants users of the weights to credit
  them. Apache-2.0 stays (no standard open licence can require credit for private or server-side
  use; only a custom licence could, and that would stop being open source). `oss/NOTICE` now has a
  required credit line, "HxH OCR by KokkakNiphon" with the repo link, and says that section 4(d)
  makes every redistribution and derived work (fine-tuned or converted models, apps that bundle
  them) keep it and show it with their third-party notices. Since 4(d) only binds when the NOTICE
  ships with the work, `hf_package_models.py` now copies it into the model package (and fails
  without it); the model card and the README got a Credit section.

## 2026-11-25: a YouTube explainer (not in this repository)

The owner is making a YouTube video about the project. Its plan, script, storyboard and motion
graphics are not part of the public repository.

## 2026-11-25: the Reader redesigned: picture beside its text

The owner asked for the Reader to be easier to use. What was wrong, from screenshots at 1440 and
390 px: the picture and its reading were far apart (selecting a line showed its details below
the whole list, a long scroll from the picture; on desktop everything sat in one 760 px column);
there was no single place to read the result, only one card per line; and every control had the
same weight (the intro and a "Start with a picture" card pushed the result below the fold on a
phone, and "Light text on dark" and "Auto-fit boxes" sat beside the model switch and Read page).
The owner picked from three layouts (side by side; text first; picture-first with a sheet):
**side by side**, decoding first with the model details one tap away, and the example page still
read on first visit.

What changed (the site's look stays: logo, colours, Geist and Manrope, the nav, shared with the
blog):
- **Source bar** under a shorter header: Take a photo or pick an image, Paste, Live camera; Models
  (Fast | Accurate) and an Options menu (Light text on dark, Auto-fit boxes, Read page) on the
  right; the example thumbnails on a second row. The eyebrow line above the title and the mono
  uppercase labels are gone.
- **Workspace**: from 960 px the picture is a sticky left column sized to the window's height and
  the text a 340-400 px right column; below 960 px the picture sticks under the nav (at most 42%
  of the screen) while the text scrolls under it.
- **Text column**: the hint, then "Text" with Copy all lines and the timing, a transcript of the
  whole reading (selectable; a click on a line selects it), then the lines in reading order as
  compact rows. The selected line opens in place: its badge beside the large reading with the
  uncertain characters, Copy line, the strip the reader saw and the notices.
- **Linked highlighting**: pointing at a box on the picture lights up its row and its transcript
  line, and pointing at a row or transcript line lights up its box (the cursor turns into a
  pointer over boxes).
- **States**: a dashed drop area before any picture (a click opens the file picker), placeholder
  rows while a page is read, "Zoom to line" / "Show all" as a small button on the picture that
  hides when there is nothing to zoom to.
- **Live camera** uses the same two columns: the camera on the left, Reading now with Stop and
  Freeze on the right.
- The Thai and Japanese pages were rebuilt from the new English markup with their existing header
  and "How it works" text; the new labels (Models, Options, Examples, Text, the drop area) were
  translated. No runtime strings changed, so the dictionaries did not.

Checked in headless Chrome (the app's browser pane was hidden): desktop 1440 x 900 in light and
dark, a 390 x 844 phone, Thai at 1440, Japanese at 390; no horizontal scroll; the sticky picture
stays at the nav's edge on a phone (top 56 px after scrolling 1,100 px); hover linking both ways,
a transcript click opening its row, zoom, the Options menu closing on Escape with focus back on
its button, the model switch re-reading the page and being remembered, live mode on the stand-in
camera and Freeze back into the picture. Fixed on the way: the opened row repeated the reading
twice (now the badge sits beside the large reading), "ライブカメラ" wrapped on a phone next to the
Ctrl+V hint (labels no longer wrap; the hint hides under 480 px), and the live speed line pushed
Freeze onto a second line.

## 2026-11-25: HxH Translator (not in this repository)

The owner built a separate fan web app, HxH Translator, on the Reader's models and the traced
glyphs. It is not part of the public repository, so its design notes are left out here.

## 2026-11-25: readers v6: numbers in the lettering

Built by a background agent from the owner's brief: "train a new version that supports numbers",
and read a picture the owner keeps as a private test image (never trained on, copied or
reproduced; its results are in the marked paragraph below). The family
readers already have the 90-token vocab (74 kana, space, `0-9 ! ? : / . -`), but they had seen
digits only in plain text fonts, a few per cent of the time.

This entry: generator v6, new line sets, both horizontal readers fine-tuned (tiny and base, same
91 classes and ids), int8 and bit-exact like the family runs, delivered as two sets in the
`web/models` layout:
- `experiments/family/tiny_v6/`: the Fast set, det_v2 (deployed, unchanged) + the new tiny reader
  `qat_v6c_tiny_h48` + the family tiny vertical reader (unchanged);
- `experiments/family/base_v6/`: the Accurate set, det_base_v1_qat (unchanged) + the new base
  reader `qat_v6b_base_h48` + the family base vertical reader (unchanged).
Only `model.hxhm` / `model.json` differ from what is deployed. The vertical readers were not
retrained: numbers inside vertical Hunter text are rare and the generator writes them only in
horizontal lines. `web/` is untouched; the owner integrates. The run is `scripts/launch_v6.sh`
(data | readers | int8 | json | eval).

### Generator v6 (`rust/hxh-synth`, `--v6 1`, all off by default)

v5 already wrote `0-9 ! ? : / . -` into 15% of horizontal Hunter lines, always in a text font
in that font's own weight. v6 makes numbers part of the lettering:

- **Digits drawn like the glyphs** (`digits.rs`, `render::layout_ex`). Every digit and mark is a
  monoline skeleton (polylines and dots in em, centred on the line axis) that goes through the
  same per-glyph transform, hand jitter, stroke width, round / square ends, brush, wave warp and
  effects as the Hunter glyphs, so a `07` comes out in the line's own ink and weight. Per line a
  digit style: height 0.78-1.1 em (a quarter of the lines up to 1.3 em: tall display digits),
  width 0.36-0.72 of the height (condensed and very condensed), digit pairs touching or
  overlapping (35%), tight (40%) or spaced; 0 as an ellipse or a stadium, 1 with or without flag
  and base, open or closed 4, flat or round 3, straight or curved 7; italic digits (20%), italic
  `!` and `?` (55%), `!` as a tapered wedge (40%); counters always open (width >= 2.3 strokes,
  height >= 3.6 strokes). 55% of the digit lines.
- **Display-font digits matched to the lettering** (the other 45%): bold, black and condensed
  faces from `fonts/open` first (Roboto Black / Bold Condensed, Lato Black, Noto Sans TC Black,
  GenEi Heavy ...); the digit height set to the line's; the stroke thickness measured on the
  face's `0` (2 x area / perimeter) and the outline dilated up to the glyphs' stroke width;
  squeezed now and then; tracking down to touching; italic `!` and `?`.
- **Real content** (`digits::inline_parts`): model numbers (07, 001), single digits, counts,
  years, dates (11/25, 11.25, 2026/11/25), times (10:30), round prices, ranges, decimals,
  fractions, and `! !! !? ?! ? !!! ?? ...`; numbers at the start, in the middle or at the end,
  punctuation mostly at the end, spaces around a number 25% of the time. Only vocab characters:
  no Latin letters (`No.12` cannot be labelled), no commas or currency signs. `.` only inside
  numbers and in `...`, never right after a kana (a dot after a glyph is its dakuten).
- **Reading guides (furigana)** (`--ruby-prob`, `page::ruby_line`): small kana (hiragana, or
  katakana for a whole line; the digits over a number) centred over each glyph on one baseline
  0.04-0.32 em above the line's ink box, 0.24-0.46 em high, on horizontal free / sign groups
  (half as often on captions and fillers, a quarter on bubbles and blocks). Unlabelled foreign
  runs (script kana, kind `ruby`); the line's quad stays the glyphs' own box. They share the
  line's frame and effects (a sticker outline goes around them too) and are exempt from the
  keep-clear rule only against their own line.
- **Sticker lettering** (`--sticker-prob`): free-standing Hunter text with a thick outline
  (0.1-0.34 em: light around dark ink, or the inverse) and, 75% of the time, the outline's own
  soft shadow (`Fx::shadow_grow`). v5's outlines were 0.05-0.14 em and mostly on busy grounds.
- **Promo scene** (8th `--mode-p` weight, `scenes::promo_page`): a merchandise picture: light or
  dark grey ground with a marble / paper texture, a glossy product on one side (below the
  lettering on portrait pages), one to three left-aligned lines of sticker lettering with numbers
  and `!!` and reading guides, a Latin logo, fine print, sometimes a price tag (all unlabelled).
- **Small dakuten** (`--small-mark-prob`, added after the first int8 readers, see "what went
  wrong"): the dakuten as a small dot, 0.55-1 stroke widths across, close to the glyph's
  bottom-right corner on the baseline. Before, the generator never drew one smaller than a
  stroke (radius 0.065 em at the reference stroke, scaled up with the weight: typically 1.3-1.7
  strokes across); bold printed lettering often does.
- **Lines**: with `--v6 1`, `hxh-synth lines` no longer cuts negatives (empty label) from foreign
  runs that contain digits, `!` or `?`: digits are read wherever they are written.
- Scenes that place lines directly (lyric, card, tv, promo) refuse a line that would overlap a
  labelled line already there (v6 only: taller digits and longer numbers outgrew a tv screen's
  spacing in the new test).
- Tests: a v6 page test (80 pages, every option on: reproducible, labelled quads disjoint and
  inside the page, labels in the 90-token vocab, foreign runs keep_margin away except the reading
  guides, which must not touch any labelled quad); unit tests for every skeleton and for the
  inline parts. v5 output is unchanged byte for byte (281 lines of lines_v5b val regenerated:
  labels and JPEGs identical), and lines_v6 regenerates exactly after the small-dakuten option
  was added (169 lines checked).

### Data

All with `--font-dir fonts/open --v5 1 --v6 1`, mode weights manga 0.2, frame 0.1, doc 0.06,
web 0.08, card 0.1, lyric 0.1, tv 0.12, promo 0.24; photos 30%, captions 25%, UI negatives 40%,
reading guides 35%, sticker lettering 35%; inline numbers 50% (`scripts/launch_v6.sh data`):

| set | seed | pages | lines (horizontal / vertical / negatives) | horizontal with numbers or `! ? : / . -` | use |
|---|---|---|---|---|---|
| lines_v6 train | 16 | 30,000 | 181,408 (161,820 / 14,774 / 4,814) | 55,158 (34%) | training |
| lines_v6 val | 16 | 1,000 | 5,866 (5,228 / 468 / 170) | 1,863 | eval |
| lines_v6m train (small dakuten in 60% of the styles) | 17 | 12,000 | 72,129 (64,425 / 5,768 / 1,936) | 21,605 | training, stage 2 |
| lines_v6_digits val (`--inline-prob 0.9`) | 61 | 1,500 | 8,902 (7,859 / 817 / 226) | 4,425 | held-out digit test |
| pages_v6 val | 62 | 200 | 1,271 labelled lines (1,269 legible), 182 reading guides | | detector check, browser pipeline |

lines_v6 train is 498 MB packed; it took 11.7 min at 43 pages/s on 6 threads, the digit test
68 s, the supplement 7.1 min on 4 threads. `previews/pages_v6.png` shows the test pages.

### Training

Both readers start from their family float checkpoints (`fam_tiny_h48b`, `fam_base_h48`),
batch 128, AdamW, cosine schedule, EMA, as in the family runs.

- **Stage 1**: lines_v6 twice + lines_v5b + synth_v4 (v6 43% of the lines, v5b 12%, synth_v4
  45%); tiny 8,000 steps, lr 8e-4, warmup 300 (9.6 min); base 10,000 steps planned, lr 8e-4: it
  died at step 8,000 (commit limit, below) and its best checkpoint (step 7,000, lr already at 11%
  of peak) was used.
- **Stage 2** (rebalancing): stage 1 doubled the tiny reader's int8 error on the old synth_v4
  hold-out and read stray `!` at line ends (below). 3,000 steps from stage 1, lr 3e-4, warmup
  100, synth_v4 twice so it is about 70% of the lines again (the family mix had 79%): `v6b` on
  lines_v6 + lines_v5b + 2 x synth_v4 (v6 17%, v5b 10%, synth_v4 73%), `v6c` with the
  small-dakuten supplement added (v6 + v6m 22%, v5b 9%, synth_v4 69%). 3.2-3.6 min each, one
  loader worker.
- **Choice**: tiny **v6c** (better than v6b on every v5 / v6 set at the same false-digit rate in
  int8; v6b's int8 build reads runs of `!` after short lines, below); base **v6b** (v6c's best
  checkpoint was picked at step 500 and its last one was level with v6b, with more stray digits).
- **int8**: the family recipe (`qat_pipeline.sh`, `NC=91`): QAT 4,250 steps on the stage-2 mix,
  lr 1e-4, one loader worker (`QAT_WORKERS=1`, new); the base reader with cross-layer
  equalization first (`EQ=1`); ESP-PPQ PTQ (esp32s3, KL) calibrated on 64 synth_v4 val lines;
  `.hxhm`; bit-exact checks. QAT took 9.7 min (tiny) and 11.1 min (base); the whole pipeline 16
  and 27 min (the base golden check with the naive reference takes about 13 min).

**Bit-exact everywhere**: both readers (and the alternative `qat_v6b_tiny_h48`) 512/512 in the
host engine and against the naive reference (`cargo test ... external`), and 512/512 logits and
CTC ids in wasm SIMD and wasm scalar (`test_wasm.mjs`). Speed in wasm SIMD: tiny 6.2 ms a line
(24.9 GMAC/s), base 29.8 ms median, 21.4 ms min (the machine was busy with other evals; the
architecture is the family's, 19.7 ms idle).

ESP-PPQ's synth_v4 hold-out (512 lines, offset 2000; the family table's numbers):

| reader | fp32 CER | int8 CER | int8 greedy identical to fp32 | .espdl |
|---|---|---|---|---|
| family tiny h (`qat_fam_tiny_h48`, deployed Fast) | 1.42% | 1.61% | 91.8% | 404 KB |
| tiny stage 1 (`qat_v6_tiny_h48`) | 2.76% | 3.06% | 79.9% | 404 KB |
| tiny v6b (`qat_v6b_tiny_h48`) | 1.73% | 2.03% | 90.2% | 402 KB |
| **tiny_v6** (`qat_v6c_tiny_h48`) | 1.98% | **1.73%** | 89.1% | 403 KB |
| family base h (`qat_fam_base_h48`, deployed Accurate) | 1.50% | 1.11% | 93.4% | 1.27 MB |
| **base_v6** (`qat_v6b_base_h48`) | 1.20% | **1.00%** | 95.3% | 1.27 MB |

### Results

**Line sets, int8** (the app's own `readCrop` in node, WASM SIMD: `scripts/eval_lines_int8.py`,
horizontal lines; synth_v4 = first 3,000 val lines, no digits at all). "Stray" = digits or
punctuation read on lines that have none.

| set | reader | CER | lines exact | CER of the lines with numbers | their digit / punct. chars wrong | stray on plain lines | negatives read |
|---|---|---|---|---|---|---|---|
| digit test | Fast (deployed) | 24.84% | 50.3% | 32.07% | 64.07% | 6/3434 | 37% |
| digit test | **tiny_v6** | **10.56%** | 71.8% | 12.84% | **19.84%** | 6/3434 | 18% |
| digit test | Accurate (deployed) | 23.41% | 52.7% | 30.25% | 60.88% | 1/3434 | 34% |
| digit test | **base_v6** | **8.54%** | 76.6% | 10.18% | **14.55%** | 6/3434 | 15% |
| lines_v6 val | Fast | 20.10% | 62.4% | 32.34% | 65.11% | 8/3365 | 40% |
| lines_v6 val | **tiny_v6** | **9.61%** | 76.9% | 13.32% | 20.99% | 8/3365 | 15% |
| lines_v6 val | Accurate | 19.02% | 64.4% | 30.35% | 61.83% | 3/3365 | 32% |
| lines_v6 val | **base_v6** | **7.91%** | 80.6% | 10.51% | 15.99% | 10/3365 | 17% |
| lines_v5b val | Fast | 9.22% | 80.1% | 18.57% | 32.58% | 21/5924 | 48% |
| lines_v5b val | **tiny_v6** | **7.68%** | 82.5% | 13.52% | 18.60% | 23/5924 | 27% |
| lines_v5b val | Accurate | 8.85% | 81.2% | 17.88% | 30.96% | 5/5924 | 41% |
| lines_v5b val | **base_v6** | **6.88%** | 84.6% | 11.43% | 15.17% | 17/5924 | 26% |
| synth_v4 val | Fast | 1.50% | 91.6% | - | - | 8/3000 | - |
| synth_v4 val | **tiny_v6** | 1.57% | 91.7% | - | - | 4/3000 | - |
| synth_v4 val | Accurate | 1.16% | 93.7% | - | - | 2/3000 | - |
| synth_v4 val | **base_v6** | **0.96%** | 94.8% | - | - | 4/3000 | - |
| synth_v4 val | tiny v6b (not delivered) | 1.73% | 91.7% | - | - | 20/3000 | - |

- Numbers: the share of digit and punctuation characters read wrong drops from 61-65% to 15-21%
  on the digit test and lines_v6, and the lines that carry them are read exactly 58-66% of the
  time instead of 20-24%; whole-line CER on the digit test falls by more than half.
- Old distributions: lines_v5b improves for both (-1.5 / -2.0 points); synth_v4 is level for
  tiny (+0.07 points, within noise for 3,000 lines) and better for base (-0.2).
- Negatives (non-Hunter crops) read as text: about halved, partly because v5b's digit-only
  negatives are now read as digits on purpose.
- Stray digits / punctuation: tiny_v6 is at the deployed Fast reader's level (better on
  synth_v4). base_v6 reads them on 17 of 5,924 lines_v5b plain lines (0.29%) against 5 for the
  deployed Accurate reader (10 against 3 and 6 against 1 on the plain lines of the v6 sets, 4
  against 2 of 3,000 on synth_v4): the one regression by the measure below. Most are a blurred
  last glyph read as `!` on photo-style lines (き, し and に are bar-and-dot glyphs).

fp32 checkpoints (`scripts/eval.py`, which decodes only the frames over the text):

| reader | digit test CER | digit / punct. chars wrong | lines_v6 val | lines_v5b val | synth_v4 | stray (v5b; synth_v4) |
|---|---|---|---|---|---|---|
| tiny family (`fam_tiny_h48b`) | 27.57% | 73.54% | 22.02% | 9.70% | 1.38% | 2/5924; 0/3000 |
| tiny stage 1 | 9.37% | 16.87% | 8.51% | 7.19% | 1.74% | 32/5924; 5/3000 |
| tiny stage 2 v6b | 10.05% | 18.92% | 9.26% | 7.56% | 1.42% | 22/5924; 3/3000 |
| **tiny stage 2 v6c** | 9.58% | 17.44% | 8.72% | 7.23% | 1.51% | 22/5924; 4/3000 |
| base family (`fam_base_h48`) | 24.32% | 64.52% | 19.31% | 8.36% | 1.13% | 3/5924; 1/3000 |
| base stage 1 | 7.92% | 13.61% | 7.30% | 6.24% | 1.64% | 26/5924; 8/3000 |
| **base stage 2 v6b** | 8.41% | 15.31% | 7.73% | 6.53% | 1.31% | 14/5924; 3/3000 |
| base stage 2 v6c (best.pt, step 500) | 8.34% | 14.79% | 7.67% | 6.57% | 1.65% | 22/5924; 6/3000 |

**Browser pipeline** (`web/ocr.js readPage` in node, int8, reading confidence 0.8; the
detectors are the deployed ones, so P / R move only through the confidence filter;
`experiments/family/eval/<models>_<set>.txt`). Fast = det_v2 + readers, full-page pass:

| set | models | P | R | F1 | CER h | CER of matched | exact h | false lines / page (text, icons, other) | F1 at score 0.55 |
|---|---|---|---|---|---|---|---|---|---|
| pages_v2 val | Fast (deployed) | 94.74% | 85.82% | 90.06% | 11.66% | 2.79% | 80.84% | 0.00, 0.00, 0.79 | 89.45% |
| pages_v2 val | **tiny_v6** | 94.87% | 85.94% | **90.18%** | 11.35% | 2.82% | 80.98% | 0.00, 0.00, 0.77 | 89.58% |
| pages_v2 val | Accurate (deployed) | 96.06% | 88.23% | 91.98% | 7.95% | 2.45% | 84.25% | 0.00, 0.00, 0.60 | 91.65% |
| pages_v2 val | **base_v6** | 96.25% | 88.23% | **92.07%** | 7.48% | 2.00% | 85.02% | 0.00, 0.00, 0.57 | 91.74% |
| pages_v3o val | Fast | 92.73% | 81.56% | 86.79% | 17.24% | 3.88% | 73.34% | 0.29, 0.02, 0.37 | 86.65% |
| pages_v3o val | **tiny_v6** | 93.81% | 81.28% | **87.10%** | 17.21% | 3.62% | 73.76% | 0.20, 0.02, 0.35 | 86.86% |
| pages_v3o val | Accurate | 95.98% | 87.58% | 91.59% | 10.62% | 2.87% | 80.40% | 0.06, 0.01, 0.32 | 91.34% |
| pages_v3o val | **base_v6** | 95.91% | 88.24% | **91.92%** | 9.75% | 2.71% | 81.66% | 0.07, 0.01, 0.32 | 91.67% |
| pages_foreign_open | Fast | 94.55% | 92.11% | 93.31% | 9.56% | 3.05% | 86.83% | 0.11, 0.07, 0.07 | 93.75% |
| pages_foreign_open | **tiny_v6** | 95.68% | 92.11% | **93.86%** | 8.76% | 2.36% | 88.51% | 0.04, 0.07, 0.08 | 93.96% |
| pages_foreign_open | Accurate | 96.24% | 95.55% | 95.90% | 4.65% | 1.92% | 91.58% | 0.10, 0.05, 0.03 | 96.03% |
| pages_foreign_open | **base_v6** | 97.10% | 95.98% | **96.54%** | 3.53% | 1.16% | 93.87% | 0.07, 0.03, 0.03 | 96.75% |
| pages_v5 val | Fast | 87.00% | 68.01% | 76.34% | 27.90% | 4.76% | 61.15% | 0.15, 0.11, 0.41 | 76.30% |
| pages_v5 val | **tiny_v6** | 88.57% | 68.31% | **77.13%** | 27.55% | 4.48% | 62.95% | 0.12, 0.09, 0.36 | 77.25% |
| pages_v5 val | Accurate | 92.80% | 75.19% | 83.07% | 20.54% | 3.55% | 69.18% | 0.08, 0.04, 0.27 | 83.17% |
| pages_v5 val | **base_v6** | 93.84% | 75.19% | **83.49%** | 19.71% | 2.63% | 71.48% | 0.03, 0.03, 0.27 | 83.58% |
| pages_v6 val (new) | Fast | 86.16% | 65.25% | 74.26% | 44.01% | 14.62% | 46.25% | 0.10, 0.10, 0.46 | 73.72% |
| pages_v6 val (new) | **tiny_v6** | 86.34% | 65.72% | **74.63%** | 38.95% | **8.04%** | 53.14% | 0.08, 0.08, 0.50 | 74.35% |
| pages_v6 val (new) | Accurate | 93.93% | 78.09% | 85.28% | 32.65% | 14.33% | 55.64% | 0.04, 0.01, 0.26 | 85.12% |
| pages_v6 val (new) | **base_v6** | 93.58% | 79.20% | **85.79%** | 25.57% | **7.51%** | 66.06% | 0.04, 0.03, 0.28 | 85.47% |

(The deployed rows on the older sets are the family run's files, `v2tiny_*` and `base_*`; the
int8 models are deterministic. CER v is unchanged: same vertical readers.)

- No regression on any page set: F1 +0.1 to +0.8 points, CER of the found lines down on 9 of 10
  rows, false lines level (pages_v6 "other": 0.46 -> 0.50 for Fast, 0.26 -> 0.28 for Accurate).
- On the v6 pages the CER of the found lines halves (14.6 -> 8.0%, 14.3 -> 7.5%).
- **What counts as a meaningful regression** (decided before reading the results): more than 0.5
  points of F1 or of CER h on a page set, more than 0.2 points of int8 CER on synth_v4, or stray
  digits on more than twice the deployed reader's share of plain lines. Only base_v6's stray
  digits cross that line (17 against 5 of 5,924 lines_v5b lines; also on the v6 sets).

**Detector check** (`scripts/det_check_v6.py`: the deployed detectors on pages_v6, all lines
found, IoU >= 0.5): the reading guides are left alone (159-166 of 182; 5-6 boxed as their own
line and none of those shown at the app's thresholds; 11-17 partly inside a matched Hunter box).
But lines with numbers are found less often than plain lines: det_v2 63.8% against 71.7%,
det_base_v1_qat 78.6% against 84.3% (398 and 763 lines). The misses are mostly not found at all
rather than cut short (partly boxed: 18 of 144 and 24 of 85 misses on number lines, 32 of 216
and 37 of 120 on plain lines), and the detectors have only ever seen digits as negatives
(pages_v5 had no inline digits; foreign digit runs are unlabelled). Not changed here, see
"decisions".

Results on the owner's private test photos are not published.

### What went wrong on the way

- **The commit limit again.** The tiny and base stage-1 fine-tunes ran side by side with two
  loader workers each (the family limit), and the tiny reader's first QAT started while the base
  fine-tune was still running: at 23:49 both died with Windows error 1455 in the loader workers
  ("Couldn't open shared file mapping"), and their main processes hung holding about 10 GB of GPU
  memory until they were killed. The base fine-tune had reached step 8,000 of 10,000; its best
  checkpoint (step 7,000, val CER 3.81%; 3.87% at 8,000) was used as is. Since then: one heavy
  job at a time, one loader worker (`QAT_WORKERS` in `qat_pipeline.sh`, default 2 as before).
- **Stage 1 alone was a regression on the old distribution.** With lines_v6 at 43% of the
  lines, the int8 tiny reader doubled its synth_v4 hold-out error (1.61% -> 3.06%, greedy
  identical 92% -> 80%), mostly dropped word spaces, and read stray `!` at line ends (`こう` ->
  `こ!` on blurred photo lines). Stage 2 put synth_v4 back to about 70% of the lines.
- **Runs of `!` after short lines in int8.** The app pads every line to 768 px by repeating its
  last pixel column; training pads batches the same way but leaves the padded frames out of the
  CTC loss, so nothing teaches the model to stay blank there. In fp32 it does anyway (no stray
  marks from the padding on synth_v4), but int8 readers emit `!` in the streaks: the deployed Fast
  reader on 8 of 3,000 synth_v4 lines, tiny v6b on 20 (51 characters; 9 lines with the padding
  filled with the background grey instead). tiny v6c happens to be clean (4 lines). The fix would
  be to pad training lines to the static width and supervise the padded frames as blank (QAT and
  fine-tuning); not done here (open).
- **Small dakuten.** After stage 2 the dakuten on the test picture sat on a knife edge in fp32.
  The generator never drew a dakuten smaller than a stroke, while bold printed lettering often
  does: `--small-mark-prob` and the lines_v6m supplement (above). It helped one of them.
- **The first promo pages had no Hunter text**: the lines were longer than the text column on
  portrait pages and every placement was refused; the scene now puts the lettering above the
  product on portrait pages and shrinks the glyph size until a line fits. The logo came out as a
  black blob: a dilation passed in pixels where `font_line` takes em.
- **A v6 test page had overlapping labelled quads** (a tv screen whose second line grew taller
  with v6 digits); scenes now refuse a line that would overlap one already placed (v6 only).
- `scripts/eval.py` decodes only the frames over the text, the app all 192: line-set numbers for
  int8 readers come from `scripts/eval_lines_int8.py` (the app's `readCrop`) instead.

### Decisions for the owner

- **Integrate?** base_v6 is better than the deployed Accurate reader on every set and reads the
  numbers; its cost is more stray digits / punctuation on plain photo-style lines (0.29% of the
  plain lines_v5b lines against 0.08%). tiny_v6 is better on every set at the deployed false-digit rate.
  Both are drop-in: copy `model.hxhm` and `model.json` of each folder (the detector and vertical
  files are the deployed ones).
- **Detector fine-tune with inline digits.** The detectors find lines with numbers 6-8 points
  less often than plain lines on pages_v6. A short fine-tune of det_v2 / det_base_v1_qat on v6
  pages (inline digits labelled as part of the line, reading guides as negatives) plus detector
  QAT would address it; not done (the brief keeps the detectors).
- **Fast on the test picture**: keep tiny_v6, or take
  `experiments/rust_engine/qat_v6b_tiny_h48_ptq/model.hxhm`, which reads one more character of
  it right but reads runs of `!` after 0.7% of short synthetic lines. Recommended: tiny_v6, and
  the padding fix below before another tiny round.

### Open

- Supervise the padded frames (static-width padding in fine-tuning and QAT) and re-check the
  trailing `!` of every int8 reader, the deployed ones included.
- The small dakuten of bold lettering is still the main miss on the test picture.
- Vertical readers: unchanged; v6 writes numbers only in horizontal lines.

### Review, the owner's decision and the blog (2026-11-26)

- **Checked again before reporting to the owner**: the private test picture, re-read with all
  four sets through `web/test/read_photo.mjs` (`scripts/eval_photos.py`'s `run`), gives the same
  readings as the agent's run; the int8 line-set numbers match `experiments/family/eval/
  lines_int8_v6.json`; the page scores match the `*_score.txt` files (pages_v5 Accurate 83.07 ->
  83.49, pages_v2 CER of matched 2.45 -> 2.00).
- Pushed 2a766af to the private repo, together with the two video commits under it (a0a0345,
  ea52cf1; their `docs/video/refs/` images are all captures of this project's own figures and
  generated pages).
- **Decision: the Reader stays on the family readers for now.** The v6 sets wait in
  `experiments/family/tiny_v6` and `base_v6` (drop-in: `model.hxhm` and `model.json`).
- **Blog**: the deep dive (EN / TH / JA) has a new Appendix E, "Numbers in the lettering (readers
  v6)": why, generator v6, the two training stages, a results table (the line sets and the page
  sets above, family against v6), the costs (stray digits, the detectors' misses on number lines,
  `!` runs in the padding, small dakuten) and that the Reader keeps the family readers. Two new
  figures from `scripts/blog/v6_images.py`: two pages_v6 val pages from `previews/pages_v6.png`,
  and ten digit-test lines with numbers drawn at random (seed 7) as the Accurate reader and its
  v6 version read them in int8 (3 of 10 exact before, 6 after; the image per language, readings
  of all four readers in `web/blog/img/deep/v6_lines.json`). Both posts got a November 26 update
  note under the November 25 one. Nothing from the private photos is on the blog: its
  evaluation stays "generated images only".

## 2026-11-24: plan addition: full-page reading (text-line detection)

Decided with the owner. The recogniser reads one horizontal line per crop;
reading a full page needs a detector in front: detect line boxes → crop and
deskew → recognise each line → order the lines.

**Milestone (after QAT and the real-image test):**
1. **Generator "page mode":** many lines per manga-style page (panels,
   screentone, speech bubbles, mixed sizes and angles, later vertical lines),
   saving line boxes and a segmentation mask. Labels come for free.
2. **Detector:** small DBNet-style segmenter (text mask → connected
   components → line boxes), budget ~0.3-0.5M params. Espressif's
   espdet_pico (0.36M params, ~120 ms at 224² on the S3) shows the size class
   is realistic.
3. **Deployment:** same chain as the recogniser (PyTorch → ONNX → ESP-PPQ
   int8 → `.hxhm`); add upsampling/concat ops to hxh-infer if needed.
4. **Demo:** a "tap a line" mode first (detect around the tap), then full
   page with reading order.

Expected cost: a fraction of a second per page in the browser; seconds per
page on the S3 (single-line camera mode stays the fast path). Open issues:
vertical text in manga, reading order across columns and panels, and real
scan/photo quality.
