# ImpulseCalc3. Solver lives in WSL. Windows browser hits 127.0.0.1 via a local bridge.
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if (-not $wsl) {
  Write-Host "No WSL. Outline only (SCOPING). FIELD_CFD needs Ubuntu WSL + OpenFOAM."
  if (-not (Test-Path ".venv\Scripts\python.exe")) { python -m venv .venv }
  $py = ".venv\Scripts\python.exe"
  & $py -m pip install -q -U pip
  & $py -m pip install -q -r requirements.txt
  & $py -m impulsecalc3.serve @args
  exit $LASTEXITCODE
}
$distro = "Ubuntu"
Write-Host "ImpulseCalc3 via WSL $distro (OpenFOAM). Close this window to stop."
Get-NetTCPConnection -LocalPort 8766 -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object {
    $p = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
    if ($p -and ($p.ProcessName -match 'python|wslrelay')) {
      Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    }
  }
wsl -d $distro -- bash -lc "fuser -k 8766/tcp >/dev/null 2>&1 || true" | Out-Null
$drive = $Root.Substring(0,1).ToLower()
$unix = "/mnt/" + $drive + ($Root.Substring(2) -replace "\\", "/")
$wslip = ((wsl -d $distro -- hostname -I) | Out-String).Trim().Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)[0]
Write-Host "WSL path $unix"
Write-Host "WSL IPv4 $wslip"
$foam = Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $distro, "--", "bash", "$unix/start-impulse-wsl.sh") -PassThru -WindowStyle Hidden
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
  try {
    $r = Invoke-WebRequest -Uri ("http://{0}:8766/" -f $wslip) -UseBasicParsing -TimeoutSec 2
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch { Start-Sleep -Milliseconds 400 }
}
if (-not $ok) {
  Write-Host "WSL serve did not come up on ${wslip}:8766"
  exit 1
}
Write-Host "Open http://127.0.0.1:8766/  (this machine only). Fallback http://${wslip}:8766/"
try {
  python "$Root\windows_localhost_bridge.py" $wslip
} finally {
  if ($foam -and -not $foam.HasExited) { Stop-Process -Id $foam.Id -Force -ErrorAction SilentlyContinue }
  wsl -d $distro -- bash -lc "fuser -k 8766/tcp >/dev/null 2>&1 || true" | Out-Null
}
