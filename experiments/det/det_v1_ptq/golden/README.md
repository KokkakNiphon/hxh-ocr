# Detector golden vectors (det_v1_ptq)

Model: `../model.espdl` (ESP-PPQ, target esp32s3, int8 per-tensor power-of-two,
PTQ, calibration kl on val[0:64]).
Source ONNX: `det_v1.onnx`.

- `inputs_u8.npy`: uint8 [64, 640, 640], pages val[900:964] of
  `data/pages_v2`, letterboxed (long side scaled to 640 with cv2 INTER_AREA when shrinking /
  INTER_LINEAR when enlarging, placed top-left, right/bottom padded with 255).
- `input_lut_int8.npy`: int8 [256], u8 pixel -> int8 model input. The float input is
  (x / 255 - 0.5) / 0.5, quantised with exponent -7 (round half up, clip to [-128, 127]).
- `outputs_int8.npy`: int8 [64, 2, 320, 320] (NCHW), ESP-PPQ TorchExecutor simulated output
  (the .espdl semantics), stored as integers; float value = int * 2^-3.
  Channel 0 = text-kernel logit, channel 1 = vertical-orientation logit.
- Output exponent -3: the prob threshold logit > -0.8473 (prob 0.3) becomes
  int > -6.778, i.e. int >= -6.
