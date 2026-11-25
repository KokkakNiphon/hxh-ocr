<#
.SYNOPSIS
  Build the browser demo in web/: the wasm engine (release, opt-level 3; one build with
  wasm SIMD, one without as a fallback), the models, and web/dist/hxh-ocr.html, a single
  self-contained page (wasm, models, examples, CSS and JS inlined; no external requests).

  Models (web\models): model.json + model.hxhm (horizontal line reader, required),
  model_v.json + model_v.hxhm (vertical reader, optional), det.json + det.hxhm (line
  detector, optional, .hxhm format v2). Each .json names its file and carries its metadata.

.EXAMPLE
  # from the repo root (PowerShell 7 or Windows PowerShell 5.1)
  .\scripts\build_web.ps1
  .\scripts\build_web.ps1 -Model experiments\rust_engine\mnv4s_h32_ptq\model.hxhm -Verify
  # a new model: copy it in, then edit web\models\model.json (name, input size, vocab)
  .\scripts\build_web.ps1 -Model path\to\new.hxhm

.PARAMETER Model
  .hxhm to copy to web\models\model.hxhm. Default: keep the one already there (or, if there
  is none, take experiments\rust_engine\mnv4s_h32_ptq\model.hxhm).
.PARAMETER ModelV
  .hxhm to copy to web\models\model_v.hxhm (vertical reader). Default: keep.
.PARAMETER Detector
  .hxhm to copy to web\models\det.hxhm (line detector). Default: keep.
.PARAMETER SkipWasm
  Don't run cargo; use the wasm files already in web\.
.PARAMETER Verify
  Run web\test\verify.mjs afterwards (needs node and web\test\out from make_reference.py).
#>
param(
  [string]$Model = "",
  [string]$ModelV = "",
  [string]$Detector = "",
  [switch]$SkipWasm,
  [switch]$Verify
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$web = Join-Path $root "web"
$rust = Join-Path $root "rust"
$utf8 = New-Object System.Text.UTF8Encoding $false
function ReadText($p) { [System.IO.File]::ReadAllText($p, [System.Text.Encoding]::UTF8) }
function B64($p) { [System.Convert]::ToBase64String([System.IO.File]::ReadAllBytes($p)) }
function Kb($n) { "{0:N1} KB" -f ($n / 1024) }

# ---------------------------------------------------------------- wasm
if (-not $SkipWasm) {
  $saved = @{
    opt = $env:CARGO_PROFILE_WASM_OPT_LEVEL
    flags = $env:CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS
  }
  Push-Location $rust
  try {
    # The `wasm` profile (rust\Cargo.toml) is size-optimised (opt "z"); the browser build
    # wants speed. Separate target dirs keep the two flag sets from rebuilding each other.
    $env:CARGO_PROFILE_WASM_OPT_LEVEL = "3"
    # Panic messages embed source paths; map the checkout, rustup and cargo homes to neutral
    # names so the published wasm carries no local paths (user name, drive layout).
    $rustup = if ($env:RUSTUP_HOME) { $env:RUSTUP_HOME } else { Join-Path $HOME ".rustup" }
    $cargo = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $HOME ".cargo" }
    $remap = "--remap-path-prefix=$root=/hxh-ocr --remap-path-prefix=$rustup=/rustup --remap-path-prefix=$cargo=/cargo"
    foreach ($b in @(
        @{ name = "hxh_infer.wasm"; dir = "target\web-simd"; flags = "-C target-feature=+simd128" },
        @{ name = "hxh_infer_nosimd.wasm"; dir = "target\web-nosimd"; flags = "" })) {
      $env:CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS = ("$($b.flags) $remap").Trim()
      Write-Host "cargo build hxh-infer-wasm -> $($b.name) ($(if ($b.flags) { $b.flags } else { 'no SIMD' }))"
      cargo build --jobs 2 -p hxh-infer-wasm --target wasm32-unknown-unknown --profile wasm --target-dir $b.dir
      if ($LASTEXITCODE -ne 0) { throw "cargo build failed ($($b.name))" }
      Copy-Item (Join-Path $b.dir "wasm32-unknown-unknown\wasm\hxh_infer_wasm.wasm") (Join-Path $web $b.name) -Force
    }
  }
  finally {
    Pop-Location
    $env:CARGO_PROFILE_WASM_OPT_LEVEL = $saved.opt
    $env:CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUSTFLAGS = $saved.flags
  }
}
foreach ($f in "hxh_infer.wasm", "hxh_infer_nosimd.wasm") {
  if (-not (Test-Path (Join-Path $web $f))) { throw "web\$f is missing (run without -SkipWasm)" }
}

# ---------------------------------------------------------------- model
$models = Join-Path $web "models"
New-Item -ItemType Directory -Force $models | Out-Null
$modelDst = Join-Path $models "model.hxhm"
if (-not $Model -and -not (Test-Path $modelDst)) { $Model = Join-Path $root "experiments\rust_engine\mnv4s_h32_ptq\model.hxhm" }
if ($Model) {
  if (-not [System.IO.Path]::IsPathRooted($Model)) { $Model = Join-Path $root $Model }
  Copy-Item $Model $modelDst -Force
  Write-Host "model: $Model -> web\models\model.hxhm"
}
foreach ($pair in @(@($ModelV, "model_v.hxhm"), @($Detector, "det.hxhm"))) {
  if ($pair[0]) {
    $src = $pair[0]
    if (-not [System.IO.Path]::IsPathRooted($src)) { $src = Join-Path $root $src }
    Copy-Item $src (Join-Path $models $pair[1]) -Force
    Write-Host "model: $src -> web\models\$($pair[1])"
  }
}
# key in window.HXH_INLINE.models -> metadata file; model.json is required, the others optional
$modelSpecs = [ordered]@{ h = "model.json"; v = "model_v.json"; det = "det.json" }
$inlineModels = @()
$modelLines = @()
foreach ($key in $modelSpecs.Keys) {
  $jsonPath = Join-Path $models $modelSpecs[$key]
  if (-not (Test-Path $jsonPath)) {
    if ($key -eq "h") { throw "web\models\model.json is missing" }
    continue
  }
  $jsonText = ReadText $jsonPath
  $meta = $jsonText | ConvertFrom-Json
  $bin = Join-Path $models $meta.file
  if (-not (Test-Path $bin)) { throw "web\models\$($modelSpecs[$key]) names $($meta.file), which is missing" }
  $magic = [System.Text.Encoding]::ASCII.GetString([System.IO.File]::ReadAllBytes($bin)[0..3])
  if ($magic -ne "HXHM") { throw "web\models\$($meta.file) is not a .hxhm file" }
  $inlineModels += "    ${key}: { json: " + $jsonText.Trim() + ", b64: `"" + (B64 $bin) + "`" }"
  $modelLines += ("web\models\{0,-17} {1}  ({2})" -f $meta.file, (Kb (Get-Item $bin).Length), $meta.name)
  if ($key -eq "h") { $mj = $meta }
}

# ---------------------------------------------------------------- single-file page
$exDir = Join-Path $web "examples"
$examples = @()
if (Test-Path (Join-Path $exDir "examples.json")) {
  foreach ($e in (ReadText (Join-Path $exDir "examples.json") | ConvertFrom-Json)) {
    $e | Add-Member -NotePropertyName src -NotePropertyValue ("data:image/png;base64," + (B64 (Join-Path $exDir $e.file))) -Force
    $examples += $e
  }
}
$examplesJson = ConvertTo-Json -InputObject @($examples) -Depth 5 -Compress

$ocr = ReadText (Join-Path $web "ocr.js")
$app = ReadText (Join-Path $web "app.js")
$ocr = [regex]::Replace($ocr, "(?m)^export\s+", "")
$app2 = [regex]::Replace($app, '(?s)import\s*\{[^}]*\}\s*from\s*"\./ocr\.js";', "// (ocr.js is inlined above)")
if ($app2 -eq $app) { throw "app.js: could not find the ocr.js import to strip" }
if ($ocr -match '(?m)^\s*import\s' -or $app2 -match '(?m)^\s*import\s') { throw "unexpected import left in the inlined JS" }
$js = $ocr + "`n" + $app2
if ($js -match '(?i)</script') { throw "the JS contains '</script', cannot inline it" }

$inline = "window.HXH_INLINE = {`n" +
  "  wasm: `"" + (B64 (Join-Path $web "hxh_infer.wasm")) + "`",`n" +
  "  wasmNoSimd: `"" + (B64 (Join-Path $web "hxh_infer_nosimd.wasm")) + "`",`n" +
  "  models: {`n" + ($inlineModels -join ",`n") + "`n  },`n" +
  "  examples: " + $examplesJson + "`n};"

$html = ReadText (Join-Path $web "index.html")
$css = ReadText (Join-Path $web "style.css")
$cssTag = '<link rel="stylesheet" href="style.css">'
$jsTag = '<script type="module" src="app.js"></script>'
if (-not $html.Contains($cssTag) -or -not $html.Contains($jsTag)) { throw "index.html: stylesheet or script tag not found" }
$stamp = (Get-Date).ToString("yyyy-MM-dd HH:mm")
# the hosted site's head block (canonical, social cards, icons, language menu) has no use in a
# single file; the shared site styles (logo) are inlined after the page's own
$html = [regex]::Replace($html, '(?s)<!-- site:begin.*?<!-- site:end -->?
?', '')
$siteCss = Join-Path $web "site.css"
if (Test-Path $siteCss) { $css = $css + "`n" + (ReadText $siteCss) }
$html = $html.Replace($cssTag, "<style>`n" + $css + "</style>")
# the theme script (web/theme.js) runs in <head>, inlined too
$themeTag = '<script src="theme.js"></script>'
if ($html.Contains($themeTag)) { $html = $html.Replace($themeTag, "<script>`n" + (ReadText (Join-Path $web "theme.js")) + "</script>") }
$html = $html.Replace($jsTag, "<script>`n" + $inline + "`n</script>`n<script type=`"module`">`n" + $js + "</script>")
$html = $html.Replace("<html lang=`"en`">", "<html lang=`"en`">`n<!-- single-file build of web/ ($stamp, model $($mj.name)); built by scripts/build_web.ps1 -->")
# nothing may point outside the page
foreach ($m in [regex]::Matches($html, '(?i)(?:src|href)\s*=\s*"([^"]*)"')) {
  $u = $m.Groups[1].Value
  # allowed: inline data, in-page anchors, Google Fonts, and plain links (the blog, the page itself, the author's GitHub)
  if ($u -and -not $u.StartsWith("data:") -and -not $u.StartsWith("#") -and -not ($u -match '^(https://fonts\.(googleapis|gstatic)\.com|https://github\.com/|blog/|index\.html$)')) { throw "single-file page still references '$u'" }
}
$dist = Join-Path $web "dist"
New-Item -ItemType Directory -Force $dist | Out-Null
$out = Join-Path $dist "hxh-ocr.html"
[System.IO.File]::WriteAllText($out, $html, $utf8)

$size = (Get-Item $out).Length
Write-Host ""
Write-Host ("web\hxh_infer.wasm        " + (Kb (Get-Item (Join-Path $web "hxh_infer.wasm")).Length))
Write-Host ("web\hxh_infer_nosimd.wasm " + (Kb (Get-Item (Join-Path $web "hxh_infer_nosimd.wasm")).Length))
foreach ($l in $modelLines) { Write-Host $l }
Write-Host ("web\dist\hxh-ocr.html     " + (Kb $size))
if ($size -gt 3MB) { Write-Warning "single-file page is over 3 MB" }

if ($Verify) {
  node (Join-Path $web "test\verify.mjs")
  if ($LASTEXITCODE -ne 0) { throw "verify.mjs failed" }
}
