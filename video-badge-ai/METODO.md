# Metodo di lavoro: episodio "Sono un badge"

Secondo episodio della serie *Usare l'IA in sicurezza* (Lezione 4: **gli strumenti autorizzati**).
Stesso stile e stesso motore del primo episodio (`video-shadow-ai`), prodotto in locale su un PC Windows potente,
con in più le app **storytold Craft** comandate da riga di comando e via MCP.

## Decisioni prese insieme
| Decisione | Scelta |
|---|---|
| Tema | Quali strumenti di IA si possono usare in azienda: Copilot con l'account aziendale per tutti, Claude solo per chi ha una licenza Team a suo nome, tutti gli altri (ChatGPT, Perplexity...) non autorizzati |
| Concept | **"Sono un badge"**: il badge aziendale fa da portiere e decide chi entra (scelto tra 3 proposte: badge, cassetta degli attrezzi, seguito del Contratto) |
| Azienda | Nome di fantasia **Nodomatic** (costante `AZIENDA` in `cartoon/badge.js`): mai il nome di un'azienda vera |
| Loghi | Si usano i **loghi ufficiali** degli strumenti, interi e mai deformati, su un medaglione bianco sul petto dei geni e sugli schermi del portatile |
| Colori dei geni | Verde acqua = autorizzato (Copilot, Claude), giallo "gratis" = non autorizzato (ChatGPT, Perplexity), come da `STILE.md` |
| Trappola chiave | "Stesso nome, porta sbagliata": anche uno strumento autorizzato, usato con l'account personale, non è autorizzato |
| Prima del video | **Storyboard** da approvare (pannelli + diapositive DeckCraft + PDF), poi bozzetti, voce e animazione |
| Organizzazione | Subagenti Sonnet in parallelo su file separati; Opus coordina, rivede ogni risultato e corregge |

Da confermare con l'IT prima della voce definitiva: quali Copilot sono abilitati e con quale account, chi assegna
le licenze Claude Team, a chi chiedere l'autorizzazione di un nuovo strumento.

## Fonti dei loghi (`loghi/`)
| Logo | Fonte |
|---|---|
| Claude, Anthropic, Perplexity, Google Gemini, DeepSeek | [Simple Icons](https://github.com/simple-icons/simple-icons) 16.36.0 (pacchetto npm), colore ufficiale del marchio |
| Microsoft Copilot | Wikimedia Commons, `Microsoft-copilot-2026-seeklogo.svg` (logo a colori); `copilot-monocromo.svg` da `Microsoft 365 Copilot Icon one-color.svg` |
| ChatGPT | Wikimedia Commons, `ChatGPT-Logo.svg` e `ChatGPT_logo.svg` |

Simple Icons non contiene più Copilot e ChatGPT/OpenAI (rimossi su richiesta dei marchi), per questo vengono da Commons.
`tools/make_loghi.js` li trasforma in `cartoon/loghi.js` (usabile nel browser e in Node): trasforma le classi CSS in
attributi `fill` (VectorCraft non legge i `<style>` interni) e rende unici gli id dei gradienti.

## Cosa c'è di nuovo rispetto al primo episodio
- `cartoon/badge.js`: il protagonista `Chars.badge` (tessera con fascia aziendale, clip, chip, codice a barre, LED
  verde/rosso), `Chatbots.genieLogo(kind, {approved})`, `Props.door` (porta con lettore di badge), `Props.windowFrame`.
- `cartoon/scenes.js`: lo schermo del portatile in `Scenes.office({ screen })` accetta `copilot`, `claude`, `chatgpt`,
  `perplexity` (logo ufficiale + nome; fascia verde acqua o gialla).
- `storyboard/`: `lib.js` carica i moduli del cartone in Node e salva pannelli 1920×1080 con sottotitolo e numero;
  i pannelli sono disegnati con **gli stessi personaggi del cartone**, quindi lo storyboard è già "in modello".

## Strumenti e dove sono usati
| Strumento | Ruolo in questo episodio |
|---|---|
| HyperFrames + GSAP | motore di animazione e render (come nel primo episodio) |
| faster-whisper | tempi parola per parola della voce (in locale Hugging Face è raggiungibile) |
| FFmpeg | audio, loudness, codifica |
| ElevenLabs | voce definitiva |
| **VectorCraft** (CLI) | conversione dei pannelli SVG in PNG (0,4 s a pannello); avvisa dei font mancanti |
| **DeckCraft** (CLI) | storyboard come presentazione, esportata in PDF e PPTX |
| **GridCraft**, **WordCraft**, **PdfCraft** | lista delle inquadrature (.xlsx) e copione (.docx/.pdf) |
| **SoundCraft**, **FilmCraft**, **EffectCraft**, **PhotoCraft/LightCraft** | in prova: mix, montaggio, grafica, ritocco (ricette in `docs/CRAFT-RICETTE.md`) |
| Blender MCP | disponibile; non usato per ora (lo stile è 2D vettoriale) |

## Avvisi noti
- VectorCraft non supporta i filtri `feTurbulence` (la grana della carta) e usa un font di riserva per IBM Plex Mono
  e per Inter ExtraBold: va bene per lo storyboard, il render finale passa da HyperFrames con i font web.

## Registro del lavoro
1. Clonata la repo, letti `PROSSIMA-SESSIONE.md`, `REPOS.md`, `NUOVO-VIDEO.md`, `STILE.md`.
2. Censite le app storytold installate (17, tutte con CLI e MCP) e gli strumenti del primo episodio presenti sul PC.
3. Proposti 3 concept; scelto il Badge con i loghi ufficiali.
4. Creata `video-badge-ai` copiando motore, personaggi e strumenti (senza i video e le voci del primo episodio).
5. Scritto il copione (`audio/script_cartoon.json`: 14 battute, 11 scene + cartello), aggiunto il Badge, i geni con logo, la porta, la finestra e gli schermi degli strumenti.
6. Pannello di prova convertito con VectorCraft e rivisto: corretti la scritta sul badge, lo schermo del portatile e la numerazione.
7. Lanciati 4 subagenti Sonnet in parallelo: pannelli 1-6, pannelli 7-12, ambiente di render su Windows, ricette delle app Craft con lista inquadrature e copione.
8. Rivisti da Opus i 12 pannelli (validi; piccoli difetti annotati: medaglione sopra la stellina GRATIS, orologio fisso alle 9:15).
9. Decisione: **2,5D** invece del 3D completo (PC con RTX 4090 adatto, ma il 3D costa molto, stacca dalla serie e rischia l'effetto generico).
10. Decisione: **personaggi da ridisegnare**. Lanciati 3 subagenti Sonnet su tre direzioni di stile (carta ritagliata, morbido premium, inchiostro e acquerello), fogli modello resi con Edge headless.
11. Ambiente Windows verificato (FFmpeg 9.0.2, HyperFrames 0.9.9 con Edge, Python 3.12 + faster-whisper, modello small); ricette Craft provate (`docs/CRAFT-RICETTE.md`).
12. Salvataggio su ramo `video-sono-un-badge` per il limite d'uso; ripartenza descritta in `PROSSIMA-SESSIONE.md`.
