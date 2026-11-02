param([string] $Python = "", [string] $Output = "", [string] $Version = "1.0.0", [string] $AsrModelDir = "")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $ProjectRoot "code/python/venv/Scripts/python.exe" }
if (-not $Output) { $Output = Join-Path $ProjectRoot "tmp_release/$Version" }
if (-not $AsrModelDir) { $AsrModelDir = Join-Path $ProjectRoot "models/asr" }
if (-not (Test-Path -LiteralPath (Join-Path $AsrModelDir "model.int8.onnx")) -or -not (Test-Path -LiteralPath (Join-Path $AsrModelDir "tokens.txt"))) { throw "指定的 ASR 模型文件不存在: $AsrModelDir" }
if (Test-Path -LiteralPath $Output) { throw "Use a new output directory: $Output" }
New-Item -ItemType Directory -Path $Output | Out-Null
$OriginalPath = $env:PATH
$OriginalAsrDir = $env:VOICE_LOCK_BUILD_ASR_DIR
try {
    $env:VOICE_LOCK_BUILD_ASR_DIR = (Resolve-Path -LiteralPath $AsrModelDir).Path
    $env:PATH = "$(Split-Path -Parent $Python);$env:WINDIR\System32;$env:WINDIR"
    & $Python -m PyInstaller --noconfirm --distpath (Join-Path $Output "dist") --workpath (Join-Path $Output "build") (Join-Path $PSScriptRoot "voice_lock.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
} finally {
    $env:PATH = $OriginalPath
    $env:VOICE_LOCK_BUILD_ASR_DIR = $OriginalAsrDir
}
$AppDirectory = Join-Path $Output "dist/SRDPVoiceLock"
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "Windows启动说明.md") -Destination (Join-Path $AppDirectory "Windows启动说明.md")
& $Python (Join-Path $PSScriptRoot "make_release.py") $AppDirectory $Version
if ($LASTEXITCODE -ne 0) { throw "Release audit or ZIP creation failed" }
Write-Output "Built: $AppDirectory"
