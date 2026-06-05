param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $Args
)

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot "code\python\venv\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "code\python\srdp.py"
$Gui = Join-Path $ProjectRoot "code\python\gui.py"

if (-not (Test-Path -LiteralPath $Python)) {
    Write-Error "Python venv not found: $Python"
    exit 1
}

if ($Args.Count -gt 0 -and $Args[0] -eq "gui") {
    $GuiArgs = @()
    if ($Args.Count -gt 1) {
        $GuiArgs = $Args[1..($Args.Count - 1)]
    }
    & $Python $Gui @GuiArgs
    exit $LASTEXITCODE
}

& $Python $Main @Args
exit $LASTEXITCODE
