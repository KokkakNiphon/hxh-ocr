# ESP-PPQ / ESP-DL feasibility spike (ESP32-S3, int8)

Host-side only, no hardware. Question: can the planned CRNN/CTC line recogniser
(1x1x32x256 in, [1,75,1,64] logits out) be quantized by ESP-PPQ to an `.espdl`
for the ESP32-S3, and how much does int8 hurt?

**Short answer.** Yes. Every planned op quantized and exported for `esp32s3`
with zero warnings, at opset 13, 17 and 18 and with both torch exporters. The
output is per-tensor, symmetric, power-of-two int8 (int32 biases limited to
20 bits), rounded half up. On a briefly trained toy model, plain PTQ raised
greedy-CTC CER from 20.4 % to 22.5 %. TQT ran on CPU in about 23 s and brought
it back to 19.9-21.4 %, which is within noise of fp32. ESP-PPQ can also export
a QDQ ONNX that onnxruntime runs, and a JSON of every scale. The QDQ model is
not bit-exact with the S3, because ONNX rounds half to even while the S3
rounds half up.

## Setup and versions

```powershell
python -m venv --system-site-packages .venv-ppq
.venv-ppq\Scripts\python.exe -m pip install esp-ppq
```

The pins were checked first, using PyPI metadata and `pip install --dry-run`.
esp-ppq 1.3.15 has no torch pin (torch is only in the `cpu`/`cu124` extras). It
requires `onnx<1.20,>=1.18`, `onnxsim==0.8.1`, `onnxscript>=0.6` and
`onnxruntime>=1.19`. The largest download was onnx at 16.5 MB, and the global
torch was left untouched.

| package | version |
|---|---|
| esp-ppq | 1.3.15 |
| torch | 2.11.0+cpu (global, inherited) |
| onnx | 1.19.1 (installed into the venv) |
| onnxruntime | 1.25.1 (global) |
| onnxsim / onnxscript / onnx-ir | 0.8.1 / 0.7.2 / 1.1.0 |
| numpy / opencv / flatbuffers | 2.2.6 / 4.12.0 / 25.9.23 |

## Files

| file | what |
|---|---|
| `toy_model.py` | Toy CRNN with every planned op, BN folding, ONNX export (opset 13/17/18, TorchScript and dynamo exporters, AvgPool vs ReduceMean). `--train-seconds N` adds a short CPU CTC training run. |
| `quantize.py` | Official flow: `espdl_quantize_onnx(..., target="esp32s3", quant_type="w8a8", setting=espdl_setting())`, optional TQT and mixed precision. Compares fp32 onnxruntime against ESP-PPQ's simulated int8 (`TorchExecutor`), exports a QDQ ONNX and checks it in onnxruntime. Writes `out/<tag>/metrics.json`. |
| `inspect_espdl.py` | Parses an `.espdl` (header + FlatBuffers) without ESP-DL. Prints nodes, dtypes and exponents. |
| `diag_tensors.py` | Per-tensor SNR of the simulation against the same graph with quantization off. |
| `summarize.py` | Builds the results tables from `out/*/metrics.json`. |
| `run_all.ps1` | Reproduces everything (~15 min, 9 of them for the optional training). |
| `out/<tag>/` | `model.espdl`, `model.json` (scales), `model.info` (text dump with integer weights), `model_qdq.onnx`, `metrics.json` |
| `out_superseded/` | int16 runs made with the wrong calibration setting, kept as evidence for gotcha 4 |

Data: calibration uses the first 64 horizontal lines of `data/synth_v1/val`
(batch 8, so 8 calib steps). Error is measured on those 64 lines and on 256
held-out val lines (offset 2000, 1,866 characters). Preprocessing: resize to
32 px height, squash to 256 if wider, otherwise right-pad with the edge
column, then `(x/255-0.5)/0.5`.

## Toy model

```
stem   Conv 3x3 s2, Cin=1 -> 16            blk1  dw 3x3 + 1x1, Add, ReLU
down1  Conv 3x3 s2 16 -> 32 (regular)      blk2  dw 5x5 + 1x1, Add, ReLU
down2  dw 3x3 stride (2,1) + 1x1 32 -> 64  blk3  dw (1,5) + 1x1, Add, ReLU
pool   AveragePool (4,1)  or  ReduceMean(H, keepdim)       head  1x1 Conv 64 -> 75
```

19k parameters, about 6.4M MACs. Every conv has BN, folded with
`fuse_conv_bn_eval` before export (max abs diff ≤ 8e-6). The ONNX graph is
11 Conv, 10 Relu, 3 Add, 1 AveragePool or ReduceMean, and no BatchNormalization.

Two weight sets were tested:
- **random**: random init, with randomized BN stats so the folding is not trivial.
- **trained**: 7 min of CPU CTC training (16k horizontal train lines, 2,123
  steps, loss 0.50), giving fp32 greedy CER of 20.4 % on the held-out lines.
  Random weights give near-tied logits (median top-1 margin 0.014), so their
  argmax agreement says nothing about CTC. The trained model (median margin
  10.7) is the one that counts.

## Results

### 1. Op support and export (esp32s3, w8a8)

All runs exported. The logs contain no unsupported-op errors and no warnings.

| check | result |
|---|---|
| Cin=1 first conv, stride 2 | OK. Weight `[3,3,1,16]`, layout `(N/16)HWC16`. |
| depthwise 3x3 / 5x5 / (1,5) | OK. `Conv group=C`, `kernel_shape=[1,5]`, `pads=[0,2,0,2]` kept. |
| depthwise stride (2,1) | OK. `strides=[2,1]` kept. |
| AveragePool kernel (4,1) | OK. `kernel_shape=[4,1] strides=[4,1]`. |
| ReduceMean over H | OK. Axes are rewritten for NHWC (axis 1), at opset 13 (attribute) and 18 (input). |
| Add, ReLU | OK. Conv+ReLU fuses into `Conv(activation='Relu')`. A ReLU after an Add becomes its own `Relu` op backed by a 256-entry int8 LUT. |
| residual skip | When the skip tensor also feeds another op at a different exponent, the exporter inserts a `RequantizeLinear` (2 of them in the trained model). |
| opset 13 vs 17 vs 18, TorchScript vs dynamo exporter | All accepted. The quantized outputs were identical. |
| AvgPool vs ReduceMean | Identical numbers in every metric. |

### 2. Quantization scheme in the output

Checked with `inspect_espdl.py`, `model.json` and the quantizer source
(`esp_ppq/quantization/quantizer/EspdlQuantizer.py`):

- **Per-tensor everywhere.** All 50 exponent vectors in the `.espdl` have
  length 1. Per-channel weights are only enabled for P4/S31 targets.
- **Power of two.** Scales are stored as integer exponents (`exponents: [-7]`),
  and every scale in `model.json` is 2^k.
- **Symmetric**, zero point 0. Int8 range is −128..127 for both weights and
  activations, so a post-ReLU tensor uses only 0..127.
- **int8 vs int16.** With `w8a8`, every weight and activation is INT8.
  Biases are stored as INT32 with a 20-bit range (±524,287) and exponent
  `in_exp + w_exp`. `w16a16`, `w8a16` and per-op mixing via
  `setting.dispatching_table` all work (see below).
- **Rounding.** `ROUND_HALF_UP` (`ESPDL_S3_INT8`), matching ESP-DL's
  operator table for the S3. P4/S31 use half-to-even.
- **Calibration.** Activations default to KL (`espdl_setting()`).
- **Layout.** NHWC: input `[1,32,256,1]`, output `[1,1,64,75]`, weights in
  16-channel blocks. The 75-channel head is tagged `(N/16)HWC16_UNALIGNED`.
- **Input.** Exponent −7, so on the device int8 input = round(x·128). With
  this normalisation that is roughly `pixel − 128`.
- **Trained logits.** Exponent −2 (step 0.25, range ±32); fp32 logits span about −57 to +28.

### 3. Quantization error (trained model)

Held-out set: 256 lines, 16,384 timesteps. "argmax" is per-timestep agreement
over the 75 classes. "seq" is the share of lines whose greedy CTC decode is
identical to fp32. CER is against the ground-truth labels.

| run | SNR dB | cos | argmax agree | seq identical | CER fp32 → int8 | .espdl B | time |
|---|---|---|---|---|---|---|---|
| PTQ, KL (default) | 22.6 | 0.9972 | 98.42 % | 65.6 % | 20.4 → **22.5** | 37,904 | 2 s |
| PTQ, percentile | 22.6 | 0.9979 | 98.52 % | 65.2 % | 20.4 → 22.7 | 37,552 | 2 s |
| PTQ, minmax | 21.7 | 0.9973 | 98.41 % | 63.3 % | 20.4 → 22.5 | 38,224 | 2 s |
| PTQ + head int16 (w8a16) | 22.7 | 0.9972 | 98.51 % | 66.0 % | 20.4 → 22.7 | 38,864 | 7 s |
| PTQ + stem & blk1 int16 | 22.2 | 0.9967 | 98.41 % | 62.5 % | 20.4 → 23.8 | 38,912 | 4 s |
| **TQT** (500 steps, lr 1e-5, block 4, int_lambda 0.25) | 24.6 | 0.9979 | 98.85 % | 69.5 % | 20.4 → **19.9** | 37,936 | 23 s |
| TQT, lr 1e-4 | 25.0 | 0.9981 | 98.86 % | 67.2 % | 20.4 → 20.5 | 37,920 | 25 s |
| TQT, lr 1e-4, 512 calib lines | 24.3 | 0.9980 | 98.61 % | 66.8 % | 20.4 → 21.4 | 37,952 | 34 s |
| w16a16 (all int16) | 69.9 | 1.0000 | 100.00 % | 100.0 % | 20.4 → 20.4 | 57,680 | 7 s |

The calibration set (64 lines) gives the same picture: PTQ 23.0 dB and
98.17 % argmax, TQT 25.0 dB and 98.68 %. With 1,866 characters, CER moves of
about ±1 point are noise, so the TQT rows are statistically equal to fp32.
`summarize.py` prints the full table, random-weight runs included. Random
weights give 34.4 dB with PTQ and 38.9 dB with TQT, but their logits are
near-tied, so those numbers say little.

Where the error comes from (`diag_tensors.py`, w8a8):
- SNR is 34 dB after the stem conv, 23.5 dB after the stem ReLU, and
  11-15 dB through the middle of the network (KL clips blk1's depthwise
  output at ±4 while fp32 reaches 16). It recovers to 18.8 dB at the logits
  after pooling.
- The error is spread over all the layers. Making the first three convs int16
  did not help, and neither did an int16 head.
- At the int8 logits, 79 of 16,384 timesteps had a top-2 tie (step 0.25), and
  48 of those disagreed with fp32. The median fp32 margin at a disagreement
  was 0.55, about 2 LSB.
- Per-channel weight ranges in the trained model differ 3-14x across channels
  (worst: the (1,5) depthwise, 13.7x). Together with power-of-two scales, the
  median channel uses only 23-80 of the 127 levels.

### 4. TQT

- It runs on CPU without any change: `collecting_device="cpu"`, Adam,
  block-wise MSE against fp32 block outputs, no labels.
- Time: 7 blocks × 500 steps took about 23 s with 64 lines and 31 s with 512
  (the 34 s in the table includes PTQ and the error report). Expect it to
  scale with model size.
- It improves the error: +2.0-2.4 dB SNR, argmax agreement 98.42 → 98.85 %,
  and it removes the PTQ CER loss.
- In this run TQT did **not** change any exponent. After rounding
  (int_lambda 0.25), every scale came back to its PTQ value. All of the gain
  came from fine-tuning weights and biases, which TQT also trains.
- Each block's tuning is undone if its loss goes up. That never happened here.
- ESP-PPQ's docstring recommends 256-2048 samples. Going from 64 to 512 lines
  did not help this toy model.

### 5. Formats a non-ESP runtime can use

- **QDQ ONNX.** `export_ppq_graph(graph, platform=TargetPlatform.ONNXRUNTIME, graph_save_to=...)`
  on the ESP-quantized graph writes an opset-13 QDQ model: per-tensor
  power-of-two scales, zero points 0, biases as int32 DequantizeLinear.
  onnxruntime 1.25.1 runs it, although ESP-PPQ warns "Exported Onnx Model is
  not executable ... (bitwidth != 8)" because of the 20-bit biases. Size is
  37.7 KB.
  - **Not bit-exact with the S3.** 86 % of the logits differ from ESP-PPQ's S3
    simulation, by a median of 2 LSB (max about 11). Argmax agreement is
    99.1 %, and held-out CER is 23.1 % vs 22.5 % for the simulation.
  - Rounding mode explains it. With the simulation switched to half-to-even,
    QDQ and simulation agree on 16,382 of 16,384 argmaxes and most lines are
    bit-identical (`qdq_ort_vs_ppq_half_even_holdout` in `metrics.json`).
    Graph optimisation level in onnxruntime makes no difference.
- **JSON of scales.** `model.json` sits next to every `.espdl`. For every tensor
  it gives bit width, quant_min/max, policy flags, state and scale/zero point,
  plus the dispatch platform per op. `model.info` is a text dump of the
  `.espdl` with all integer weights and exponents.
- **`.espdl` format.** A 16-byte header (`"EDL2"`, u32 encrypted flag, u32
  payload length, 4 padding bytes), then a FlatBuffers `espdl.Model`.
  - It is ONNX-like: Graph, Node, Attribute, Tensor, with an `exponents` field
    on tensors and value infos, and `raw_data` stored as 16-byte-aligned blocks.
  - Schema: `esp-dl/fbs_loader/espdl.fbs` in github.com/espressif/esp-dl.
    Python bindings ship in `esp_ppq/parser/espdl/FlatBuffers/Dl`, and
    `inspect_espdl.py` uses them.
  - Optional AES-128 encryption via `encrypt_data=True`.
  - Test inputs and outputs can be embedded with `export_test_values=True`.

### 6. Sizes

| file | bytes |
|---|---|
| fp32 ONNX | 80,348 |
| `.espdl` (int8) | 37,904 (21,136 of it is weights, biases and LUTs; the rest is FlatBuffers graph metadata, names and value_info) |
| `.espdl` with embedded test input/output | 51,088 |
| `.espdl` w16a16 | 57,680 |
| QDQ ONNX | 37,748 |

The metadata overhead is roughly fixed at about 17 KB. A 0.3M-parameter model
would come to about 0.3 MB, plus 4 bytes of int32 bias per output channel and
that overhead.

## Gotchas

1. **`espdl_quantize_onnx` overwrites its input file** with the onnxsim-simplified
   model (`onnx.save(model_sim, onnx_import_file)`). Pass it a copy.
   `quantize.py` does this.
2. **QDQ export plus ReduceMean at opset 18 gives an invalid model.** The exporter
   declares opset 13 but keeps the axes input. onnxruntime reports
   "ReduceMean:13 has input size 2". Fixes: use AveragePool, export at opset 13,
   or raise the QDQ model's opset to 18 (tested: it runs).
3. **QDQ ONNX is not bit-exact with the device** because of rounding (see 5).
4. **For `quant_type="w16a16"`, build the setting with `espdl_setting(num_of_bits=16)`.**
   `espdl_quantize_onnx` only picks the 16-bit defaults (minmax calibration)
   when `setting=None`. The default 8-bit setting keeps KL calibration, which
   clips badly at 16 bits. That gave 17.7 dB and 28 % CER instead of 70 dB and
   lossless (`out_superseded/`). The same applies to 16-bit ops in a mixed
   setup: `quantize.py --mixed` sets `calib_algorithm_per_layer[op]="minmax"`.
5. **Mixed precision: dispatch a fused Conv and its ReLU together.** Otherwise
   ESP-PPQ warns "Op ... should be send to a same platform" and the ReLU stays
   int8.
6. `export_test_values=True` adds the test tensors to the `.espdl` (+13 KB here).
   Leave it off for size measurements.
7. The calibration DataLoader may use batch > 1, but `input_shape` and `inputs`
   must have batch 1. TQT trains on the cached calibration batches, so their
   batch size is the TQT batch size.

## Notes from ESP-DL source (not tested on hardware)

- int8 SIMD kernel selection on the S3 is keyed on kernel shape (1x1, 3x3 or
  "other") and on channel alignment: `c_align = Cin%16==0 && Cout%16==0`
  (`esp-dl/dl/base/dl_base_conv_select.hpp`). In this model:
  - the Cin=1 stem and the 75-class head take the unaligned path;
  - the 5x5 and (1,5) depthwise kernels fall in the "other" class.
- AveragePool has a TIE728 path when `C%16==0`. It requantizes with a float
  multiply, because 1/(kh·kw) is not a power of two. ReduceMean uses generic
  code with float requantization.
- Neither the Conv nor the AveragePool module restricts asymmetric strides or
  kernels. `operator_support_state.md` lists Conv (groups 1 or C_in),
  AveragePool (1d/2d, no dilation), ReduceMean, Add and Relu as int8-supported
  on the S3, and recommends opset 18.

## Recommendations for the real model

1. **Ops:** keep the planned set. Nothing failed. Use **opset 18** (Espressif's
   recommendation; 13 also works). Prefer **AveragePool (h,1)** over ReduceMean
   for the height collapse: same numbers, an aligned SIMD path, and the QDQ
   export works as is.
2. **Channel counts:** multiples of 16 everywhere except the network input.
   Consider **padding the classifier to 80 outputs** (5 dummy classes ignored at
   decode) to avoid the unaligned head. Verify the speed gain on hardware.
3. **Kernels:** 1x1 and 3x3 have their own kernel class. 5x5 and (1,5)
   depthwise go to the generic class. Where speed matters, prefer stacked 3x3
   depthwise, and benchmark (1,5) against 3x3 on the device before committing.
4. **Residual blocks:** a ReLU after an Add is a separate LUT op, and a skip
   tensor reused at a different exponent costs a RequantizeLinear. A
   MobileNetV2-style linear bottleneck (no activation after the Add) avoids
   the first. Both costs are small, about one extra pass over the activation.
5. **Accuracy plan:**
   - Plain PTQ cost about 2 CER points on the toy model. Expect the same order
     of loss on the real one.
   - Budget for **TQT**: it is cheap (seconds to minutes on CPU, no labels)
     and recovered the full loss here.
   - For robustness, train with **QAT that mimics the target** (per-tensor,
     symmetric, power-of-two, signed int8, half-up rounding). Keep plain ReLU:
     the `.espdl` exporter fuses only `Relu` into a Conv
     (`ESPDL_CONV_LIKE_FUSED_ACTIVATIONS = {'Relu'}` in `esp_ppq/core/common.py`),
     so ReLU6/Clip would become a separate op. If ranges need bounding, do it
     in training (QAT or regularisation) rather than with Clip.
   - Per-tensor weight scales across depthwise channels were not the dominant
     error in this toy (3-14x channel spread), but they will be in a trained
     model with a large BN-folded spread. Watch the per-layer table in
     `quantize.py`'s log.
6. **int16 fallback:**
   - w16a16 was lossless in simulation, with a 1.5x larger file and 2x
     activation memory. The largest tensor is 32 KB in int8 and 64 KB in int16,
     both well inside SRAM. Kernels are slower (8 lanes instead of 16).
   - Mixing int16 into only a few layers did not help here, because the error
     is spread across all layers.
7. **Browser:**
   - The QDQ ONNX from ESP-PPQ is a quick onnxruntime-web demo, at 99 % argmax
     agreement with the device simulation.
   - For outputs identical to the S3, use the planned custom integer engine.
     Read weights and exponents from the `.espdl` (or `model.json` plus
     `model.info`), shift with round-half-up, and check it against
     `TorchExecutor`, or against the test values embedded with
     `export_test_values=True`.
8. **Static shapes, batch 1:** keep `1x1x32x256`. ESP-PPQ requires batch 1, and
   the vertical-text path needs its own static shape (or a rotate-to-horizontal
   preprocess).

## Caveats

- No hardware. "int8 output" means ESP-PPQ's `TorchExecutor` simulation, which
  Espressif documents as matching the board. Latency was not measured.
- The toy model is tiny (19k params) and only weakly trained (20 % CER). A
  well-trained model has larger margins, which helps argmax stability, but the
  real model will also be deeper. Re-run `quantize.py` on the real checkpoint.
- Runs that overlapped with CPU training show inflated `quant s` (the 34 s
  and 51 s random-weight runs). The TQT timings in the table were taken after
  that training had finished. Other processes on the PC may still have used
  CPU at the time.
