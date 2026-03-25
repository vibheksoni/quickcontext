param(
    [ValidateSet("local", "cloud")]
    [string]$Profile = "local",

    [ValidateSet("debug", "release", "skip")]
    [string]$ServiceBuild = "debug",

    [string]$Config = "quickcontext.json",
    [string]$Venv = ".venv",
    [switch]$SkipDocker,
    [switch]$SkipInit,
    [switch]$DryRun
)

$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $python = @("py", "-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = @("python")
} else {
    throw "Python 3 was not found on PATH."
}

$scriptPath = Join-Path $PSScriptRoot "bootstrap_quickcontext.py"
$args = @(
    $scriptPath,
    "--profile", $Profile,
    "--service-build", $ServiceBuild,
    "--config", $Config,
    "--venv", $Venv
)

if ($SkipDocker) {
    $args += "--skip-docker"
}
if ($SkipInit) {
    $args += "--skip-init"
}
if ($DryRun) {
    $args += "--dry-run"
}

if ($python.Length -gt 1) {
    & $python[0] $python[1] @args
} else {
    & $python[0] @args
}
exit $LASTEXITCODE
