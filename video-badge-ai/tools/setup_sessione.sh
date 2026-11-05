#!/usr/bin/env bash
# Prepara una sessione nuova (container pulito) per montare la voce ElevenLabs e renderizzare il cartone.
# Installa solo cio' che serve alla pipeline del cartone 2D (niente Blender, Remotion, ComfyUI, Chatterbox):
#   - Python: faster-whisper (VAD Silero incluso nel pacchetto, nessun download da Hugging Face), numpy, soundfile -> .venv/
#   - Node:   hyperframes 0.9.9 + font (Inter, EB Garamond, IBM Plex Mono) -> .tools/
#   - FFmpeg: quello di sistema (apt se manca)
# Uso (dalla cartella video-shadow-ai):  bash tools/setup_sessione.sh  &&  source .tools/env.sh
set -euo pipefail
cd "$(dirname "$0")/.."
P=$PWD

if ! command -v ffmpeg >/dev/null; then
  echo "== FFmpeg (apt)"; SUDO=; [ "$(id -u)" = 0 ] || SUDO=sudo
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq ffmpeg
fi

echo "== Python (.venv)"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
# PyAV 19 ha tolto un argomento che faster-whisper 1.2.1 usa ancora: si resta su av 18
.venv/bin/pip install -q "faster-whisper==1.2.1" "av<19" numpy soundfile

echo "== Node (.tools)"
mkdir -p .tools
cat > .tools/package.json <<'JSON'
{ "name": "cartone-tools", "private": true,
  "dependencies": { "hyperframes": "0.9.9", "@fontsource/inter": "^5.3.0", "@fontsource/eb-garamond": "^5.3.0", "@fontsource/ibm-plex-mono": "^5.3.0" } }
JSON
(cd .tools && npm install --no-audit --no-fund --loglevel=error)

# browser: quello preinstallato nel container (Playwright), altrimenti Chrome di sistema
BROWSER=$(ls -d /opt/pw-browsers/chromium-*/chrome-linux*/chrome 2>/dev/null | head -1 || true)
[ -n "$BROWSER" ] || BROWSER=$(command -v google-chrome || command -v chromium || true)
SHELL_BIN=$(ls -d /opt/pw-browsers/chromium_headless_shell-*/*/chrome-headless-shell 2>/dev/null | head -1 || true)
FF=$(command -v ffmpeg); FP=$(command -v ffprobe)
cat > .tools/env.sh <<ENV
export HYPERFRAMES_NO_TELEMETRY=1 HYPERFRAMES_NO_UPDATE_CHECK=1 HYPERFRAMES_SKIP_SKILLS=1 HYPERFRAMES_NO_OPTIONAL_INSTALL=1
export HYPERFRAMES_BROWSER_PATH=$BROWSER
${SHELL_BIN:+export PRODUCER_HEADLESS_SHELL_PATH=$SHELL_BIN}
export HYPERFRAMES_FFMPEG_PATH=$FF HYPERFRAMES_FFPROBE_PATH=$FP FFMPEG=$FF
export PATH=$P/.tools/node_modules/.bin:$P/.venv/bin:\$PATH
ENV

echo "== Controllo"
.venv/bin/python -c "from faster_whisper.vad import get_speech_timestamps; import numpy, soundfile; print('python ok')"
.tools/node_modules/.bin/hyperframes --version
echo "Pronto. Ora: source .tools/env.sh"
