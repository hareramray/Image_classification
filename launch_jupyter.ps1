$ErrorActionPreference = 'Stop'
$courseRoot = $PSScriptRoot
$coursePython = Join-Path $courseRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $coursePython)) {
    throw 'Course environment missing. Follow the README setup instructions first.'
}
Set-Location -LiteralPath $courseRoot
& $coursePython -m jupyterlab
