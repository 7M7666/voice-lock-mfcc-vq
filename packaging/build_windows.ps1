param([string] $Python = "", [string] $Output = "", [string] $Version = "1.0.0")
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) { $Python = Join-Path $ProjectRoot "code/python/venv/Scripts/python.exe" }
if (-not $Output) { $Output = Join-Path $ProjectRoot "tmp_release/$Version" }
if (Test-Path -LiteralPath $Output) { throw "Use a new output directory: $Output" }
New-Item -ItemType Directory -Path $Output | Out-Null
$OriginalPath = $env:PATH
try {
    $env:PATH = "$(Split-Path -Parent $Python);$env:WINDIR\System32;$env:WINDIR"
    & $Python -m PyInstaller --noconfirm --distpath (Join-Path $Output "dist") --workpath (Join-Path $Output "build") (Join-Path $PSScriptRoot "voice_lock.spec")
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed" }
} finally {
    $env:PATH = $OriginalPath
}
$AppDirectory = Join-Path $Output "dist/SRDPVoiceLock"
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "Windows启动说明.md") -Destination (Join-Path $AppDirectory "Windows启动说明.md")
& $Python (Join-Path $PSScriptRoot "make_release.py") $AppDirectory $Version
if ($LASTEXITCODE -ne 0) { throw "Release audit or ZIP creation failed" }
Write-Output "Built: $AppDirectory"
