# Ambiente Windows per la pipeline del cartone

Questo PC (Windows 11) è pronto per rifare le voci, il lip-sync e il render come nell'episodio 1
(`video-shadow-ai/PROSSIMA-SESSIONE.md`). L'equivalente Windows di `tools/setup_sessione.sh` è
[`tools/setup_windows.ps1`](../tools/setup_windows.ps1).

## Come si parte (ogni sessione)
Dalla cartella `video-badge-ai`, in PowerShell:
```powershell
. .\.tools\env.ps1          # il punto iniziale e' obbligatorio: le variabili restano nella sessione
```
Imposta `PYTHONUTF8=1`, `HYPERFRAMES_BROWSER_PATH`, `HYPERFRAMES_FFMPEG_PATH`, `HYPERFRAMES_FFPROBE_PATH`, `FFMPEG`
(lo leggono `tools/eleven_tts.py`, `timeline.py`, `make_audio_cartoon.py`) e mette nel `PATH` FFmpeg, `.venv\Scripts`
e `.tools\node_modules\.bin`. Da lì `python` è quello del venv e `hyperframes` è la versione fissata (0.9.9).
I percorsi sono **calcolati** a ogni avvio (non scritti a mano), quindi sopravvivono a un aggiornamento di FFmpeg o del browser.

Per installare o riparare tutto (si può rilanciare quante volte si vuole, non rifà ciò che c'è già):
```powershell
powershell -ExecutionPolicy Bypass -File tools\setup_windows.ps1            # opzioni: -Browser chrome   -SkipModel
```

## Cosa c'è, versioni e percorsi
| Cosa | Versione | Dove |
|---|---|---|
| FFmpeg + ffprobe | 9.0.2 (full build Gyan, `winget install Gyan.FFmpeg`) | `%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_...\ffmpeg-9.0.2-full_build\bin` |
| Browser per HyperFrames | **Edge 156** (predefinito); Chrome 157 come alternativa | `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe` · `C:\Program Files\Google\Chrome\Application\chrome.exe` |
| HyperFrames CLI | 0.9.9 (come su Linux; richiede Node ≥ 22, qui Node 26.13) | `.tools\node_modules\.bin\hyperframes` |
| Font (Inter, EB Garamond, IBM Plex Mono) | `@fontsource/*` ^5.3 | `.tools\node_modules` |
| Python | 3.12.10 (Microsoft Store, lanciato con `py -3.12`) | venv in `.venv` |
| faster-whisper | 1.2.1 (con `av<19`, come nello script Linux), numpy 2.5.4, soundfile 0.14.0 | `.venv\Lib\site-packages` |
| Modello Whisper `small` (int8, CPU) | Systran/faster-whisper-small, 464 MB | `C:\Users\franc\.cache\huggingface\hub\models--Systran--faster-whisper-small` |

Non installato di proposito: `piper-tts` (serve solo a `tools/tts.py`, la voce guida provvisoria, con un modello
in `/opt/models` che qui non c'è; la voce vera è ElevenLabs) e Blender/Remotion/ComfyUI/Chatterbox (prima versione 3D, non più usata).

Il file `.env` con `ELEVENLABS_API_KEY` non è stato toccato: va creato a mano in `video-badge-ai\.env` solo quando serve `eleven_tts.py`.

## Comandi (stessa sequenza di PROSSIMA-SESSIONE.md, in PowerShell)
```powershell
. .\.tools\env.ps1
python tools\timeline.py ...            # e gli altri script in tools\
python cartoon\build.py
Set-Location cartoon\hf
hyperframes check
hyperframes render --quality delivery -o ..\..\design\cartone-vN-master.mp4
ffmpeg -y -i ..\..\design\cartone-vN-master.mp4 -c:v libx264 -preset slow -tune animation -crf 18 -pix_fmt yuv420p -c:a copy -movflags +faststart ..\..\design\cartone-vN.mp4
```
In Windows PowerShell 5.1 `&&` non esiste: usare `;` o righe separate (in PowerShell 7 funziona).
Utile per prove veloci: `hyperframes render --quality draft --start 9 --end 12 -o prova.mp4`.

## Prova eseguita (smoke test)
Copia di `video-shadow-ai\cartoon\hf` in `C:\Users\franc\AppData\Local\Temp\claude\hf-smoke` (l'originale non è stato toccato):
- `hyperframes check`: **passa** con Edge e con Chrome. L'unico avviso è quello già noto, contrasto di `#rk-st` nella scena "rischio" (1,98:1), non bloccante.
- `hyperframes snapshot --at 10`: fotogramma 1920x1080 corretto (orologio, faldoni, sottotitolo "Ore nove e quindici.", font Inter ok), 5,6 s.
- `hyperframes render --quality draft --start 9 --end 12`: MP4 h264 + AAC 1920x1080, 3 s di video in circa 15 s con Edge (13 s con Chrome).
- Dal venv: `import faster_whisper, numpy, soundfile` e `faster_whisper.vad` ok; il modello `small` è in cache e si carica.

## Problemi incontrati e cose da sapere
- **`python` nel PATH è lo stub del Microsoft Store.** Lo script usa il launcher `py -3.12` (c'era già Python 3.12.10) e crea il venv; dopo `. .\.tools\env.ps1` `python` è quello del venv.
- **FFmpeg era già installato da winget** ma non nel `PATH` della sessione: `winget install` risponde "già installato" (exit 43). `env.ps1` lo cerca nella cartella dei pacchetti winget.
- **L'audio non è nel repo dell'episodio 1** (`assets\audio\*.wav` sono generati da `make_audio_cartoon.py`): sulla copia grezza `hyperframes check` si ferma con `audio_src_not_found` *prima* di aprire il browser. Per la prova ho generato tre WAV di silenzio nella copia temporanea; in un progetto vero basta lanciare prima `make_audio_cartoon.py` e `build.py`.
- **Usare `hyperframes`, non `npx hyperframes`.** Fuori da `.tools`, `npx hyperframes` ha risolto una copia in cache dell'ultima versione (0.9.16) invece della 0.9.9 fissata. Con `env.ps1` attivo il comando `hyperframes` usa sempre la 0.9.9; `npx hyperframes` dà 0.9.9 solo se lanciato da dentro `.tools`.
- Edge è stato accettato da HyperFrames (check, snapshot, render): non c'era bisogno di installare altro. Chrome è già presente come ripiego (`-Browser chrome`).
- Hugging Face è raggiungibile (scaricato il modello). L'avviso "unauthenticated requests to the HF Hub" è innocuo: non serve alcun token.
- **Git:** `.venv/` e `.tools/` (con `node_modules`) **non sono nel `.gitignore`**: vanno aggiunti prima di un `git add .`.
- Il render completo (circa 83 s di video) non è stato lanciato: qui è stato provato solo uno spezzone. Il tempo reale dipende dai thread della CPU (worker paralleli automatici).
