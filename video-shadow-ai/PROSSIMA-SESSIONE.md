# Voce ElevenLabs: fatto (v3). Come rifarla o ritoccarla

`design/cartone-v3.mp4` ha la **voce definitiva di ElevenLabs**. Tutto è generato da `timeline.json`: animazione,
lip-sync, sottotitoli, musica ed effetti. Cambiando una battuta si rigenera e si riallinea tutto.

## La voce usata
| | |
|---|---|
| Voce | "Contratto - Sono un contratto" (`voice_id` Iva39E3GwRoV8qBIVcgR), creata con **Voice Design** (`eleven_ttv_v3`) |
| Descrizione | donna italiana sui 35-40 anni, accento neutro, "impiegato modello" orgoglioso e puntiglioso, ironia asciutta, sorriso nella voce (testo completo in `audio/provini/design.json`) |
| Modello | `eleven_v3`, con i tag di recitazione di [`VOCE-ELEVENLABS.md`](VOCE-ELEVENLABS.md) |
| Impostazioni | stability 0.5 (in v3 solo 0 / 0.5 / 1), similarity_boost 0.75, style 0.3, speaker boost attivo |
| Formato | MP3 44,1 kHz 128 kbps (piano Starter) → WAV mono 48 kHz; velocità 1x (provata 1,15x, scartata) |
| Contesto tra battute | `previous_text`/`next_text` non sono supportati da `eleven_v3`: ogni battuta è generata da sola |

Provini scartati in `audio/provini/` (4 voci della libreria e altre 5 di Voice Design).
La voce è più lenta della voce guida: il cartone passa da 67 a **83 s**. La voce è alzata di 1,2 dB con un limiter
a -3 dBFS in `tools/make_audio_cartoon.py`: il mix finale è a circa -15 LUFS, come la v2.

## Rifare una battuta (dalla cartella `video-shadow-ai`)
La chiave sta in `video-shadow-ai/.env` (`ELEVENLABS_API_KEY=...`, escluso da git): la legge solo lo script.
```bash
# ambiente: Linux/macOS -> bash tools/setup_sessione.sh && source .tools/env.sh
#           Windows     -> stessi passi in PowerShell (.venv\Scripts\python.exe), PYTHONUTF8=1 e
#                          HYPERFRAMES_BROWSER_PATH / _FFMPEG_PATH / _FFPROBE_PATH impostati a mano
python tools/eleven_tts.py battute Iva39E3GwRoV8qBIVcgR 8        # rigenera solo la battuta 8 (0-12)
python tools/eleven_tts.py tempo 1                                # (facoltativo) velocità di tutte le battute
python tools/timeline.py audio/script_cartoon.json audio/eleven audio/pads_cartoon.json audio/voice_eleven.wav
python tools/fw_words.py                                          # tempi parola per parola (faster-whisper small)
python tools/make_audio_cartoon.py && python cartoon/build.py
cd cartoon/hf && npx hyperframes check && npx hyperframes render --quality delivery -o ../../design/cartone-v3-master.mp4
ffmpeg -y -i ../../design/cartone-v3-master.mp4 -c:v libx264 -preset slow -tune animation -crf 18 -pix_fmt yuv420p \
       -c:a copy -movflags +faststart ../../design/cartone-v3.mp4
```
Altri comandi: `eleven_tts.py voci` (piano, modelli, voci italiane), `provini ID ...`, `design "descrizione" nome`,
`salva GENERATED_ID nome`. Se "Shadow AI" esce all'italiana: `battute ID 7 --sciadou`.
Il render completo dura circa 6 minuti su un PC a 16 thread (worker paralleli automatici), la versione web circa 1,5.
Da salvare su GitHub (ramo `video-sono-un-contratto`, poi su `main`): `audio/eleven/`, `audio/voice_eleven.wav`,
`timeline.json`, `cartoon/hf/`, `design/cartone-v3.mp4` in un commit a parte (il master no: è troppo pesante).

## Da sapere
- `tools/fw_words.py` ora trascrive **ogni battuta dal suo file**: sulla traccia montata la prima parola finiva a volte
  nella battuta precedente, e le parole oltre i primi 6 s di una battuta venivano perse.
- `tools/make_audio_cartoon.py` usa i tempi di faster-whisper anche per gli effetti sonori, come `Anim.wordTime` in JS.
- `hyperframes check` passa con un solo avviso, non bloccante: contrasto di `#rk-st` nella scena "rischio" (1,98:1).
