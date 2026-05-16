param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot "code\python\venv\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "code\python\srdp.py"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python venv not found: $Python"
    exit 1
}

& $Python $Main @Args
exit $LASTEXITCODE
