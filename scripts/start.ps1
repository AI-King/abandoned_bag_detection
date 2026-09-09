$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
uv sync --locked
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
uv run vision prepare
if ($LASTEXITCODE -ne 0) { throw 'Asset preparation failed' }
uv run streamlit run app.py
