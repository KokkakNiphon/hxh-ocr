# Reproduce the ESP-PPQ spike. Run from anywhere; uses .venv-ppq.
# Total time on this PC: ~15 min (about 9 of them are the optional CPU training incl. image loading).
$ErrorActionPreference = 'Stop'
$py = Join-Path $PSScriptRoot '..\..\.venv-ppq\Scripts\python.exe'
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force logs | Out-Null

# 1. toy model: random weights + 7 min of CPU CTC training; ONNX opset 13/17/18 (+ dynamo exporter), AvgPool / ReduceMean
& $py toy_model.py --train-seconds 420 *> logs/toy_model.log

# 2. PTQ for every export variant (random weights) + TQT on random weights
foreach ($t in 'random_avg_op13','random_avg_op17','random_avg_op18','random_avg_op18_dynamo',
               'random_mean_op13','random_mean_op17','random_mean_op18','random_mean_op18_dynamo') {
    & $py quantize.py --onnx "models/toy_$t.onnx" --tag $t *> "logs/$t.log"
}
& $py quantize.py --onnx models/toy_random_avg_op18.onnx --tag random_avg_op18_tqt --tqt *> logs/random_avg_op18_tqt.log

# 3. trained weights: PTQ calibration variants, mixed precision, TQT, int16 ceiling, test values
$m = 'models/toy_trained_avg_op18.onnx'
$runs = @(
    @('trained_avg_op18'),
    @('trained_mean_op18', '--onnx', 'models/toy_trained_mean_op18.onnx'),
    @('trained_avg_op18_percentile', '--calib-algo', 'percentile'),
    @('trained_avg_op18_minmax', '--calib-algo', 'minmax'),
    @('trained_avg_op18_head16', '--mixed', '/head/Conv=w8a16', '--no-qdq'),
    @('trained_avg_op18_early16', '--no-qdq',
      '--mixed', '/stem/stem.0/Conv=w16a16', '--mixed', '/stem/stem.2/Relu=w16a16',
      '--mixed', '/blk1/dw/dw.0/Conv=w16a16', '--mixed', '/blk1/dw/dw.2/Relu=w16a16',
      '--mixed', '/blk1/pw/pw.0/Conv=w16a16', '--mixed', '/blk1/Add=w16a16', '--mixed', '/blk1/act/Relu=w16a16'),
    @('trained_avg_op18_tqt', '--tqt'),
    @('trained_avg_op18_tqt_lr1e-4', '--tqt', '--tqt-lr', '1e-4'),
    @('trained_avg_op18_tqt_lr1e-4_n512', '--tqt', '--tqt-lr', '1e-4', '--calib-n', '512', '--calib-bs', '16'),
    @('trained_avg_op18_w16a16', '--quant-type', 'w16a16'),
    @('trained_avg_op18_testvals', '--test-values', '--no-qdq')
)
foreach ($r in $runs) {
    $tag = $r[0]; $extra = @($r | Select-Object -Skip 1)
    if ($extra -notcontains '--onnx') { $extra = @('--onnx', $m) + $extra }
    & $py quantize.py --tag $tag @extra *> "logs/$tag.log"
}

# 4. tables + file inspection + per-tensor error of the simulation
& $py summarize.py
& $py inspect_espdl.py out/trained_avg_op18/model.espdl
& $py diag_tensors.py --onnx models/toy_trained_avg_op18.onnx --quant-type w8a8
