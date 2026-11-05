# Repo GitHub usati e metodo di lavoro

## Come ho lavorato
Il punto di partenza era il video originale della Lezione 3 ("Usare l'IA in sicurezza", Shadow AI) con i suoi
14 repo di riferimento. La richiesta non era copiarlo, ma **ripensarlo** usando quei repo e progettando insieme il design.

1. **Analisi dell'originale:** trascrizione e analisi delle scene e del messaggio. Da lì è nato un concept nuovo
   scelto insieme: *"Sono un contratto"*, la storia raccontata dal documento che viene incollato in un chatbot gratuito.
2. **Installazione e prova dei 14 repo** in un ambiente cloud senza GPU e con la rete filtrata (Hugging Face,
   YouTube e altri siti bloccati). Per ciascuno ho deciso se e come usarlo (tabella sotto).
3. **Prima versione, 3D di carta:** stop-motion in Blender, guidato tramite Blender MCP, scene Three.js in Remotion
   e montaggio in HyperFrames. Funzionava, ma l'obiettivo era un **cartone professionale**, non un video da "AI slop".
4. **Riprogettazione come cartone 2D:** prima i bozzetti, poi l'animazione. Foglio modello dei personaggi e tre immagini
   chiave da approvare (`design/foglio-modello.png`, `design/immagine-chiave-*.png`), poi un motore di animazione
   scritto apposta (personaggi SVG parametrici, chiavi con curve GSAP, lip-sync con visemi italiani a passo due,
   palpebre, camera) e le 10 scene animate una per una, con i gesti agganciati alle parole del copione.
5. **Revisioni con te:** v1, poi v2 con i chatbot ridisegnati come "genio digitale" (dalle proposte in
   `design/chatbot-proposte.png`) e la bolla protettiva finale ricollocata (`design/protezione-proposte.png`).
6. **Controllo qualità a ogni giro:** `hyperframes check` (errori, contrasto WCAG, audio), fotogrammi di controllo
   con `hyperframes snapshot` riguardati scena per scena, Impeccable `detect`, Chrome DevTools MCP per le tavole
   di design, loudness misurata con FFmpeg.
7. **Voce:** una voce guida provvisoria (Piper) per dare i tempi; la voce definitiva è ElevenLabs
   (`VOCE-ELEVENLABS.md`, `PROSSIMA-SESSIONE.md`). Tutto è guidato da `timeline.json`: cambiando la voce,
   animazione, lip-sync, sottotitoli, musica ed effetti si riallineano da soli.

La prima versione in 3D è stata tolta dal repo per tenerlo pulito. Resta nella cronologia di git
(commit `c3c1ead`, cartelle `video-shadow-ai/blender`, `remotion`, `hyperframes`).

## I 14 repo
| # | Repo | Ruolo nel cartone finale | Perché |
|---|------|-------------------------|--------|
| 1 | [heygen-com/hyperframes](https://github.com/heygen-com/hyperframes) | ✅ **motore di montaggio e render** | Ogni scena è una composizione HTML; `check`, `snapshot` e `render` per controllo ed export |
| 2 | [greensock/GSAP](https://github.com/greensock/GSAP) | ✅ **animazione** | Curve di movimento (back, power, expo...) per ogni chiave e la timeline di ogni scena |
| 3 | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) | ✅ **tempi della voce** | Il VAD Silero misura inizio e fine di ogni battuta e taglia la ripresa ElevenLabs; con Hugging Face raggiungibile dà anche i tempi parola per parola |
| 4 | [FFmpeg/FFmpeg](https://github.com/FFmpeg/FFmpeg) | ✅ **audio e codifica** | Compilato dal sorgente: montaggio della voce, loudness, versione web del video |
| 5 | [ChromeDevTools/chrome-devtools-mcp](https://github.com/ChromeDevTools/chrome-devtools-mcp) | ✅ **tavole di design** | Render e controllo (console senza errori) di foglio modello, immagini chiave e proposte |
| 6 | [pbakaus/impeccable](https://github.com/pbakaus/impeccable) | ✅ **revisione** | `detect` sulle composizioni: 0 problemi; linee guida di animazione e critica |
| 7 | [remotion-dev/skills](https://github.com/remotion-dev/skills) | ✅ **metodo** | Le regole di Remotion (precaricare le risorse prima di disegnare, frame-accurate) applicate al motore |
| 8 | [blender/blender](https://github.com/blender/blender) | 🗄️ prima versione | Usato per il 3D di carta; per un cartone vettoriale il 2D in SVG è più nitido e molto più veloce senza GPU |
| 9 | [ahujasid/mcp-for-blender](https://github.com/ahujasid/mcp-for-blender) | 🗄️ prima versione | Scene Blender costruite e corrette via MCP, guardando la vista con `look` |
| 10 | [remotion-dev/remotion](https://github.com/remotion-dev/remotion) | 🗄️ prima versione | Tunnel e copie in 3D; nel cartone queste scene sono rifatte in 2D per coerenza di stile |
| 11 | [mrdoob/three.js](https://github.com/mrdoob/three.js) | 🗄️ prima versione | Scene astratte (fibre, server, scudo) in Remotion |
| 12 | [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) | ↪️ sostituito | I pesi stanno su Hugging Face (bloccato); la voce definitiva la fa ElevenLabs |
| 13 | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) | ⏳ non usato | Installato e avviato, ma senza modelli (Hugging Face e Civitai bloccati) e senza GPU; gli sfondi sono vettoriali disegnati in codice |
| 14 | [yt-dlp/yt-dlp](https://github.com/yt-dlp/yt-dlp) | ❌ non usato | YouTube e Vimeo bloccati; il riferimento era già il tuo video, musica ed effetti sono sintetizzati |

Altri strumenti: **Piper** (voce guida provvisoria, `tools/tts.py`), **ElevenLabs** (voce definitiva),
font open source **Inter**, **EB Garamond**, **IBM Plex Mono**.

## Note per i prossimi video
- **Sul tuo PC** (sessione locale) Hugging Face non è bloccato: lì faster-whisper dà i tempi parola per parola
  e ComfyUI potrebbe dipingere gli sfondi partendo dai layout vettoriali, se in futuro si vuole quella strada.
  Lo stile attuale è però pensato per sfondi vettoriali: è ciò che lo rende coerente e nitido.
- Le dipendenze minime della pipeline sono in `tools/setup_sessione.sh`: Python con faster-whisper,
  Node con HyperFrames, FFmpeg, Chrome.
