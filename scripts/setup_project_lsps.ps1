param(
    [Parameter(Mandatory = $true)]
    [string]$Path,
    [switch]$Install,
    [switch]$Check,
    [switch]$Yes,
    [switch]$Json
)

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $RepoRoot "venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$Args = @("-m", "engine", "lsp-setup", $Path)
if ($Install) { $Args += "--install" }
if ($Check) { $Args += "--check" }
if ($Yes) { $Args += "--yes" }
if ($Json) { $Args += "--json-output" }

& $Python @Args
exit $LASTEXITCODE
