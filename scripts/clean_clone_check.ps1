# Clean clone check (human runs; agent never executes)

Clones this repo path into $env:TEMP, copies .env, runs sync/tests/mock smoke, prints PASS/FAIL.

```powershell
# Usage (from repo root):
#   powershell -File scripts/clean_clone_check.ps1
```

$ErrorActionPreference = "Continue"
$src = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dest = Join-Path $env:TEMP ("2care-clean-" + [guid]::NewGuid().ToString("N").Substring(0, 8))

function Step($name, $scriptBlock) {
  Write-Host "=== $name ==="
  try {
    & $scriptBlock
    if ($LASTEXITCODE -ne $null -and $LASTEXITCODE -ne 0) {
      Write-Host "FAIL $name (exit $LASTEXITCODE)"
      return $false
    }
    Write-Host "PASS $name"
    return $true
  } catch {
    Write-Host "FAIL $name : $_"
    return $false
  }
}

Write-Host "Temp path: $dest"
New-Item -ItemType Directory -Path $dest | Out-Null

$ok = $true
$ok = (Step "clone" { git clone -- "$src" "$dest"; if (-not $?) { exit 1 } }) -and $ok

if (Test-Path (Join-Path $src ".env")) {
  Copy-Item (Join-Path $src ".env") (Join-Path $dest ".env")
  Write-Host "Copied .env into temp clone"
} else {
  Write-Host "WARN: no .env in source; mock steps may still run"
}

Push-Location $dest
try {
  $ok = (Step "uv sync" { uv sync --all-extras; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }) -and $ok
  $ok = (Step "pytest" { uv run pytest -q -rs; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }) -and $ok
  $env:MOCK_LLM = "1"
  $ok = (Step "mock agent help/import" {
    uv run python -c "import clinic_agent; print('agent import ok')"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }) -and $ok
  $ok = (Step "mock eval k=1" {
    uv run python -m clinic_agent.evals policy/policy_v1.yaml 1
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
  }) -and $ok
} finally {
  Pop-Location
}

Write-Host "Temp path: $dest"
if ($ok) { Write-Host "OVERALL PASS"; exit 0 } else { Write-Host "OVERALL FAIL"; exit 1 }
