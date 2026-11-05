# "Sono un contratto": Usare l'IA in sicurezza · Lezione 3

**Cartone animato 2D** di circa 83 s in 16:9 sul tema della Shadow AI. Il protagonista è un contratto che viene
copiato in un chatbot gratuito, finisce chissà dove e poi, "riavvolgendo", mostra il modo giusto: strumento approvato,
regole chiare, persone formate. Tutti i personaggi sono oggetti con caratteristiche umane, e il tono è leggero e ironico.

- **Video attuale:** [`design/cartone-v3.mp4`](design/cartone-v3.mp4), con la **voce definitiva di ElevenLabs**: voce
  "Contratto - Sono un contratto" creata con Voice Design, modello `eleven_v3` con i tag di recitazione
  (dettagli in `audio/eleven/meta.json` e in [`PROSSIMA-SESSIONE.md`](PROSSIMA-SESSIONE.md)).
- **Versione precedente:** [`design/cartone-v2.mp4`](design/cartone-v2.mp4), 67 s, stessa animazione con la voce guida provvisoria.
- **Design:** `design/foglio-modello.png` (personaggi), `design/immagine-chiave-*.png` (ambienti),
  `design/chatbot-proposte.png` e `design/protezione-proposte.png` (proposte approvate per la v2).

## Documentazione
| File | Contenuto |
|---|---|
| [`STILE.md`](STILE.md) | guida di stile della serie: mondo, disegno, tavolozza, personaggi, animazione, tipografia, suono, struttura di un episodio |
| [`NUOVO-VIDEO.md`](NUOVO-VIDEO.md) | come fare un nuovo episodio nello stesso stile, con un prompt pronto per Claude Code |
| [`REPOS.md`](REPOS.md) | i 14 repo GitHub usati (e perché) e il metodo di lavoro |
| [`VOCE-ELEVENLABS.md`](VOCE-ELEVENLABS.md) | casting della voce, impostazioni e testo delle 13 battute |
| [`PROSSIMA-SESSIONE.md`](PROSSIMA-SESSIONE.md) | la voce della v3 (voce, modello, impostazioni) e i comandi per rifare una battuta e renderizzare |

## Pipeline
| Passo | Strumento | File |
|---|---|---|
| Copione e recitazione | testo e tag di ElevenLabs | `VOCE-ELEVENLABS.md`, `audio/script_cartoon.json` |
| Voce (13 battute, una per file) | API ElevenLabs (`eleven_v3`, Voice Design) | `tools/eleven_tts.py` → `audio/eleven/NN.wav`, `meta.json` |
| Timeline (battute, pause per le gag) | faster-whisper (VAD) | `tools/timeline.py`, `audio/pads_cartoon.json` → `timeline.json` |
| Tempi parola per parola (lip-sync, gesti, sottotitoli, effetti) | faster-whisper | `tools/fw_words.py` |
| Personaggi, oggetti, sfondi | SVG parametrico disegnato in codice | `cartoon/characters.js`, `chatbots.js`, `props.js`, `scenes.js` |
| Animazione (chiavi, lip-sync, camera) | GSAP | `cartoon/anim.js`, `cartoon/shots/*.html` |
| Montaggio, sottotitoli, render | HyperFrames | `cartoon/build.py` → `cartoon/hf/` |
| Musica ed effetti, volume della voce | sintesi Python + FFmpeg | `tools/make_audio_cartoon.py` |
| Tavole di design | Chrome DevTools MCP | `cartoon/model-sheet.html`, `frame.html`, `*-proposte.html` |
| Controllo qualità | HyperFrames `check` e `snapshot`, Impeccable `detect` | |
| Ambiente per una sessione nuova | | `tools/setup_sessione.sh` |

## Rigenerare
```bash
bash tools/setup_sessione.sh && source .tools/env.sh      # solo la prima volta in un ambiente nuovo
python tools/make_audio_cartoon.py && python cartoon/build.py
cd cartoon/hf && hyperframes check && hyperframes render --quality delivery -o ../../design/cartone-master.mp4
```
