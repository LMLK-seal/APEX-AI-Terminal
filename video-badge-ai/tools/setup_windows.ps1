# Prepara il PC Windows per la pipeline del cartone (equivalente di setup_sessione.sh). Idempotente: si puo' rilanciare.
#   FFmpeg (winget Gyan.FFmpeg), Python 3.12 + .venv (faster-whisper, numpy, soundfile), Node + HyperFrames 0.9.9 in .tools/,
#   un Chromium (Edge o Chrome gia' installati) e il modello faster-whisper "small" in cache.
# Uso (dalla cartella video-badge-ai):  powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1
#   poi, in ogni sessione:               . .\.tools\env.ps1
# Opzioni: -Browser auto|edge|chrome (ordine di preferenza del browser)   -SkipModel (non scarica il modello Whisper)
param([ValidateSet('auto', 'edge', 'chrome')][string]$Browser = 'auto', [switch]$SkipModel)
$ErrorActionPreference = 'Stop'
$P = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $P
function Step($m) { Write-Host "== $m" -ForegroundColor Cyan }
function Check($what) { if ($LASTEXITCODE -ne 0) { throw "$what fallito (exit $LASTEXITCODE)" } }

# ---------------------------------------------------------------- .tools/env.ps1 (le variabili sono calcolate a ogni dot-source)
$order = switch ($Browser) { 'chrome' { "'chrome','edge'" } default { "'edge','chrome'" } }
New-Item -ItemType Directory -Force .tools | Out-Null
$envTemplate = @'
# Generato da tools/setup_windows.ps1. Uso: . .\.tools\env.ps1   (dalla cartella video-badge-ai, una volta per sessione PowerShell)
$P = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$env:HYPERFRAMES_NO_TELEMETRY = '1'; $env:HYPERFRAMES_NO_UPDATE_CHECK = '1'; $env:HYPERFRAMES_SKIP_SKILLS = '1'; $env:HYPERFRAMES_NO_OPTIONAL_INSTALL = '1'
$findExe = {
  param($name)
  $c = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($c) { return $c.Source }
  Get-ChildItem (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages') -Directory -Filter 'Gyan.FFmpeg*' -ErrorAction SilentlyContinue |
    ForEach-Object { Get-ChildItem $_.FullName -Recurse -Filter "$name.exe" -ErrorAction SilentlyContinue } | Select-Object -First 1 -ExpandProperty FullName
}
$ff = & $findExe 'ffmpeg'; $fp = & $findExe 'ffprobe'
if ($ff -and $fp) { $env:HYPERFRAMES_FFMPEG_PATH = $ff; $env:HYPERFRAMES_FFPROBE_PATH = $fp; $env:FFMPEG = $ff; $env:PATH = (Split-Path $ff) + ';' + $env:PATH }
else { Write-Warning 'FFmpeg non trovato: lancia tools\setup_windows.ps1' }
$pf86 = ${env:ProgramFiles(x86)}
$browsers = @{
  edge   = @("$pf86\Microsoft\Edge\Application\msedge.exe", "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe")
  chrome = @("$env:ProgramFiles\Google\Chrome\Application\chrome.exe", "$pf86\Google\Chrome\Application\chrome.exe", "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe")
}
$br = @(@@ORDER@@) | ForEach-Object { $browsers[$_] } | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($br) { $env:HYPERFRAMES_BROWSER_PATH = $br } else { Write-Warning 'Nessun Edge/Chrome trovato: installa Chrome (winget install Google.Chrome)' }
$env:PATH = "$P\.tools\node_modules\.bin;$P\.venv\Scripts;" + $env:PATH
Remove-Variable findExe, ff, fp, pf86, browsers, br -ErrorAction SilentlyContinue
'@
Set-Content -Path .tools\env.ps1 -Value $envTemplate.Replace('@@ORDER@@', $order) -Encoding ascii
. .\.tools\env.ps1

# ---------------------------------------------------------------- FFmpeg
Step 'FFmpeg'
if (-not $env:HYPERFRAMES_FFMPEG_PATH) {
  winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements; Check 'winget FFmpeg'
  . .\.tools\env.ps1
}
if (-not $env:HYPERFRAMES_FFMPEG_PATH) { throw 'FFmpeg non trovato nemmeno dopo winget' }

# ---------------------------------------------------------------- Python 3.12 (o 3.11) + .venv
Step 'Python (.venv)'
function Find-Python {
  foreach ($v in '3.12', '3.11') {
    $e = & py "-$v" -c 'import sys; print(sys.executable)' 2>$null
    if ($LASTEXITCODE -eq 0 -and $e) { return $e }
  }
}
$py = Find-Python   # il "python" nel PATH puo' essere lo stub del Microsoft Store: si usa il launcher py
if (-not $py) {
  winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements; Check 'winget Python'
  $py = Find-Python
}
if (-not $py) { throw 'Python 3.11/3.12 non trovato (riapri PowerShell dopo winget e rilancia)' }
if (-not (Test-Path .venv\Scripts\python.exe)) { & $py -m venv .venv; Check 'venv' }
$vpy = "$P\.venv\Scripts\python.exe"
& $vpy -m pip install -q --upgrade pip; Check 'pip'
# PyAV 19 ha tolto un argomento che faster-whisper 1.2.1 usa ancora: si resta su av 18 (piper-tts, solo per tools/tts.py, non serve)
& $vpy -m pip install -q "faster-whisper==1.2.1" "av<19" numpy soundfile; Check 'pip install'

# ---------------------------------------------------------------- Node + HyperFrames
Step 'Node (.tools)'
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw 'Node non trovato: winget install --id OpenJS.NodeJS.LTS -e' }
@'
{ "name": "cartone-tools", "private": true,
  "dependencies": { "hyperframes": "0.9.9", "@fontsource/inter": "^5.3.0", "@fontsource/eb-garamond": "^5.3.0", "@fontsource/ibm-plex-mono": "^5.3.0" } }
'@ | Set-Content -Path .tools\package.json -Encoding ascii
Push-Location .tools; npm install --no-audit --no-fund --loglevel=error; Check 'npm install'; Pop-Location

# ---------------------------------------------------------------- modello faster-whisper "small" (una volta sola, da Hugging Face)
if (-not $SkipModel) {
  Step 'Modello faster-whisper small'
  & $vpy -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8'); print('modello small ok')"; Check 'modello Whisper'
}

# ---------------------------------------------------------------- controllo finale
Step 'Controllo'
. .\.tools\env.ps1
& $env:HYPERFRAMES_FFMPEG_PATH -version | Select-Object -First 1
& $env:HYPERFRAMES_FFPROBE_PATH -version | Select-Object -First 1
Write-Host "browser: $env:HYPERFRAMES_BROWSER_PATH"
& $vpy -c "import faster_whisper, numpy, soundfile; from faster_whisper.vad import get_speech_timestamps; print('python', __import__('sys').version.split()[0], '- faster-whisper', faster_whisper.__version__, '- numpy', numpy.__version__, '- soundfile', soundfile.__version__)"; Check 'import python'
hyperframes --version; Check 'hyperframes'
Write-Host 'Pronto. In ogni sessione: . .\.tools\env.ps1' -ForegroundColor Green
