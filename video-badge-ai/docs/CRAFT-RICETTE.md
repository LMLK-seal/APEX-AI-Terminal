# Ricette Craft (app storytold) per il cartone 2D

Provate il 2026-11-05 su Windows 11, con i CLI in `C:\Program Files\<App>\<app>-cli.exe`.
Tutti i file di prova sono in `docs/prove-craft/` (comandi, script e output). Ogni ricetta e marcata
**TESTATO** (eseguita davvero, output controllato) oppure **NON TESTATO** (letta solo dall'help).

Pipeline di riferimento: HyperFrames (HTML+GSAP) -> MP4 1920x1080 30 fps; voce = WAV per battuta;
musica/sfx sintetizzati in Python; mix finale a circa -15 LUFS.
**ffmpeg NON e installato**: i suoi compiti (mux, probe, misura loudness, estrazione frame) li fanno
FilmCraft + SoundCraft.

Convenzioni usate sotto: `D="/c/Program Files/DeckCraft/deckcraft-cli.exe"` ecc. (Git Bash).
Nei JSON i percorsi Windows vanno scritti con `/` (es. `C:/Users/franc/...`).
Da PowerShell o cmd usare le stesse righe con i percorsi completi `.exe`.

## Dove aiutano davvero (riassunto)

| App | Valore per il cartone | Stato |
|---|---|---|
| **DeckCraft** | Deck dello storyboard (immagine a sinistra + testo a destra), export PDF/PPTX, 14 slide in 0,6 s | TESTATO, utile |
| **SoundCraft** | Mix di piu WAV con offset e gain, **misura LUFS/true peak** di qualsiasi file, normalizza/converte | TESTATO, utile |
| **FilmCraft** | Mux video+audio, export H.264 **con target LUFS integrato** (`loudnessLufs`), probe, estrazione frame | TESTATO, utile |
| **EffectCraft** | Post (grana, vignettatura), card titolo/terzo inferiore a template con valori JSON, render MP4/PNG | TESTATO, utile |
| **GridCraft** | Shot list CSV -> XLSX formattato con formula del totale | TESTATO, utile |
| **WordCraft** | Copione DOCX (e PDF) da Markdown | TESTATO, utile |
| **PdfCraft** | Unire/estrarre pagine PDF, testo, render pagina | TESTATO, utile (secondario) |
| **VectorCraft** | SVG -> PNG/PDF, **SVG con testo convertito in tracciati** | TESTATO, utile |
| **PhotoCraft** | Batch su cartelle di PNG (resize, colore, grana), overlay texture carta | TESTATO, utile |
| **LightCraft** | Sviluppo foto/RAW; per i PNG del cartone aggiunge poco | TESTATO, marginale |
| **MaterialCraft** | Texture procedurali tileable (tessuto, grunge, nuvole) esportabili PNG | TESTATO, marginale |
| **FigCraft** | Frame/testo vettoriali -> PNG/SVG/PDF (card, terzi inferiori) | TESTATO, marginale |
| **DesignCraft** | Impaginazione DTP -> PDF | TESTATO solo sample, marginale |
| CADCraft, SolidCraft, TextureCraft, Craft Anything | Nessuna utilita per un cartone 2D | NON utili |

---

## 1. DeckCraft - deck storyboard  (TESTATO)

Il formato del file comandi (`--cmd-file`): una riga per comando, `id={json}` oppure solo `id`;
righe vuote e `# commento` ignorate; UTF-8 (anche CRLF/BOM). Slide da 960x540 pt (16:9):
x verso destra, y verso il basso, `rect=[x,y,larghezza,altezza]` in punti. Indici delle slide da 0.

File di prova: `prove-craft/storyboard.cmds` (una slide). Contenuto:

```
file.new={"blank":true,"size":[960,540]}
slide.new={"layout":"blank"}
design.background={"color":"#F4EBD9"}
insert.picture={"path":"C:/Users/franc/progetto-p-Iva/video-badge-ai/storyboard/panels/prova.png","rect":[24,40,620,349]}
insert.textBox={"rect":[664,40,272,460],"text":"Scena 1 - buongiorno\nBuongiorno. Sono il badge di Nodomatic.\nInquadratura: PP badge, leggera oscillazione\nSuono: ambiente ufficio, passi"}
text.edit
text.select={"anchor":[0,0],"caret":[0,20]}
format.size={"size":24}
format.bold={"on":true}
text.select={"anchor":[1,0],"caret":[1,38]}
format.size={"size":16}
text.select={"anchor":[2,0],"caret":[2,45]}
format.size={"size":12}
format.italic={"on":true}
text.select={"anchor":[3,0],"caret":[3,30]}
format.size={"size":12}
format.color={"color":"#B03030"}
text.exit
```

Esecuzione (salva, esporta PDF in un colpo, poi PPTX):

```
D="/c/Program Files/DeckCraft/deckcraft-cli.exe"
"$D" run --cmd-file storyboard.cmds --print --save storyboard.deckcraft --export storyboard.pdf
"$D" convert storyboard.deckcraft storyboard.pptx          # il formato lo decide l'estensione
"$D" render storyboard.deckcraft --slide 0 anteprima.png   # PNG 1920x1080 (--scale 2 di default), per controllare
```

Risultato: `storyboard.deckcraft` (140 KB), `storyboard.pdf`, `storyboard.pptx`, anteprima corretta
(immagine a sinistra circa 2/3, testo a destra con titolo 24 pt grassetto, corpo 16, corsivo 12, riga rossa 11/12).

**Storyboard completo (14 slide) generato dal copione**:
`python prove-craft/genera_storyboard_deck.py` legge `docs/shot-list.csv` e scrive `storyboard_full.cmds`;
poi `"$D" run --cmd-file storyboard_full.cmds --save storyboard_full.deckcraft --export storyboard_full.pdf`.
Tempo: 0,6 s per 267 comandi. Lo script cerca per ogni battuta `storyboard/panels/NN.png`
(o `panel-NN.png`, `N.png`); se manca usa `prova.png`. **Adattare la funzione `panel()` ai nomi veri dei pannelli.**

Gotcha:
- `text.edit` + `text.select` con `[paragrafo, carattere]` seleziona; `format.*` agisce sulla selezione.
  Senza selezione agisce su tutta la casella. Chiudere sempre con `text.exit` prima della slide successiva.
  Il `caret` conta caratteri: calcolarlo con `len()` sul paragrafo (accenti ok).
- `file.new {"blank":true}` crea un deck con **0 slide**: la prima va aggiunta con `slide.new`.
  Senza `file.new`, `run` parte con una slide titolo vuota.
- `--export x.pdf` esporta **tutte** le slide; `--export x.png` solo la slide corrente.
  PDF di 14 slide con PNG ripetuta = 9 MB (l'immagine viene incorporata a ogni slide).
- La casella testo NON si restringe da sola: se il testo e lungo, vedere `format.autofit {"mode":"shrink"}` (NON TESTATO).
- Font: default **Inter**. Incorporati: Inter, Lato, Montserrat, Poppins, Roboto, Open Sans, Nunito Sans,
  Source Sans 3, Source Serif 4, Playfair Display, Merriweather, Liberation, Carlito/Caladea. Cambio con `format.font {"family":"Lato"}` (NON TESTATO).
- Altri comandi utili (NON TESTATI): `slide.notes`, `insert.table`, `design.slideSize`, `file.export` con `gif` (animatic: `--export-params '{"secondsPerSlide":3}'`),
  `slide.fromOutline` (slide da testo/markdown). `deckcraft-cli commands` ne elenca oltre 400, `describe <id>` ne spiega uno.

## 2. SoundCraft - mix, misura LUFS, normalizza  (TESTATO)

CLI in stile Pro Tools. Sessione = `.scraft`. File di prova: `prove-craft/mix.sc`, `mkwav.py` (genera due WAV sinusoidali).

```
S="/c/Program Files/SoundCraft/soundcraft-cli.exe"
# mix.sc  (una riga per comando: ID {json})
session.new {"name":"prova","sample_rate":48000,"bit_depth":24}
track.new {"name":"Voce","format":"mono"}
track.new {"name":"Musica","format":"mono"}
file.import_audio {"path":"C:/.../voce_test.wav","track":"Voce","at":"0:01.000","new_tracks":false}
file.import_audio {"path":"C:/.../musica_test.wav","track":"Musica","at":0,"new_tracks":false}
mix.volume {"tracks":["Musica"],"db":-12}

"$S" script mix.sc --bounce C:/.../mix_sound.wav --save mix.scraft --inspect --compact
```

Output della bounce (una riga JSON): `{"bytes":...,"lufs":-10.41,"peak_db":-8.67,"true_peak_db":-8.67,"seconds":4.0,"path":...}`.
Quindi **la bounce misura LUFS e true peak**. Durata 4,0 s = voce a 1 s + musica da 4 s, offset corretto.

- **Posizioni (`at`)**: usare stringa `"m:ss.mmm"` (es. `"0:01.000"`) oppure intero = campioni (48000 = 1 s).
  **Un numero con la virgola (1.0) viene letto come 1 campione** e `"1s"` / `"1.0 sec"` danno errore.
- `mix.volume {"tracks":[nomi o numeri],"db":-144..12}`; gain di clip: `clip.gain {"clips":[...],"db":..}` (NON TESTATO).
- `--bounce` accetta `--start/--end` (secondi); `file.bounce_mix` come comando ha `format`, `bit_depth`, `normalize`, `fold_down`.
- **Misura LUFS di un file qualsiasi** (anche MP4: SoundCraft legge mp4/m4a/aac/ogg/flac/mp3):
  ```
  "$S" run --cmd 'session.new={"name":"m"}' --cmd 'file.import_audio={"path":"C:/.../file.wav"}' --bounce C:/.../_misura.wav
  ```
  e leggere `lufs`/`true_peak_db` dall'ultima riga. 0,1 s. Il file `_misura.wav` e solo di scarto.
  Provato su `edit_out.mp4` (FilmCraft): -15,00 LUFS, e su un export senza target: -7,29 LUFS.
- **Normalizza/converte** (solo picco, NON loudness): `"$S" convert in.wav out.wav --bit-depth 16 --rate 44100 --normalize`
  (picco a -0,1 dBFS; ok anche flac/aiff). `info file.wav --compact` = formato, canali, durata.
- Per portare il mix a -15 LUFS: misurare, poi correggere con `mix.volume` (o usare `loudnessLufs` di FilmCraft, piu semplice).
- Plugin interni (NON TESTATI): compressore, maximizer, EQ, riverbero, gate, de-esser (`"$S" plugins`), via `mix.insert`.
- Gotcha: `--inspect` mostra come `name` il nome del file di salvataggio, non quello di `session.new`. Un `.scraft` salvato copia anche i WAV in `Audio Files/`.

## 3. FilmCraft - mux, export H.264, probe, frame  (TESTATO)

CLI in stile Premiere. Ogni invocazione `exec`/`inspect` e un processo separato **senza stato**: usare sempre `run script.jsonl`
(un processo, una riga JSON per comando `{"id":..,"params":{..}}`, `#` = commento).

Generare un MP4 di prova senza ffmpeg (`prove-craft/gen.jsonl`): progetto, sequenza 1920x1080 30 fps, color matte da 3 s, piazzato, esportato:

```
F="/c/Program Files/FilmCraft/filmcraft-cli.exe"
{"id":"file.newProject","params":{"name":"gen"}}
{"id":"file.newSequence","params":{"name":"seq","width":1920,"height":1080,"fps":30}}
{"id":"file.newColorMatte","params":{"color":"#e07a30","seconds":3}}
{"id":"timeline.place","params":{"item":8,"track":"V1","seconds":0}}
{"id":"file.exportMedia","params":{"path":"C:/.../matte_test.mp4","format":"h264","wait":true}}
"$F" run gen.jsonl
```

**Montaggio MP4 + WAV con loudness a -15 LUFS** (`prove-craft/edit.jsonl`, e il modo consigliato per il mux finale):

```
{"id":"file.newProject","params":{"name":"edit"}}
{"id":"file.newSequence","params":{"name":"seq","width":1920,"height":1080,"fps":30}}
{"id":"file.import","params":{"paths":["C:/.../video.mp4","C:/.../voce.wav","C:/.../musica.wav"]}}   -> {"items":[8,9,10]}
{"id":"timeline.place","params":{"item":8,"track":"V1","seconds":0}}      # video (porta con se il suo audio A1 se c'e)
{"id":"timeline.place","params":{"item":9,"track":"A2","seconds":1}}      # WAV su A2 a 1 s
{"id":"timeline.place","params":{"item":10,"track":"A3","seconds":0}}     # WAV su A3
{"id":"file.exportMedia","params":{"path":"C:/.../edit_out.mp4","format":"h264","loudnessLufs":-15,"wait":true}}
```

Risultato: H.264 High 1920x1080 30 fps + AAC 48 kHz stereo, 120 frame, **loudness misurato con SoundCraft = -15,003 LUFS**
(senza `loudnessLufs`: -7,29). Velocita: 3 s di video in circa 2,4 s (50-88 fps); l'intero script 1,2 s.

- Gli id (`item`) sono assegnati in ordine (sequenza 7, poi 8, 9, ...): **leggerli dal risultato di `file.import`**, non darli per scontati.
- `probe`: `"$F" probe file.mp4` = JSON (codec, risoluzione, fps, durata in tick: 254016000000 tick = 1 s, canali audio). Funziona anche su WAV.
- **Frame singolo in PNG**: il sottocomando `render --seconds` NON funziona in modalita headless ("no active sequence", serve `--bridge` all'app aperta).
  Alternativa provata: `file.exportMedia` con `"format":"png","range":"custom","startSeconds":1,"endSeconds":1.04,"audio":false`.
  **Gotcha**: `path` e un nome base, FilmCraft aggiunge un indice a tre cifre (`nome000.png`, `nome001.png`); `%05d` NON e un pattern
  (ho ottenuto per errore file chiamati `frame_%05d000.png`). Usare un nome normale tipo `C:/.../frame.png`.
- Altri formati: hevc, prores, dnxhr, gif, wav, aac, flac, sequenze png/tiff (`"alpha":true`). Preset: `"$F" export --list-presets`.
- Altro (NON TESTATO): `captions.*` (sottotitoli, `captionSidecar:"srt"`, `burnCaptions`), `effects.apply`, transizioni (`sequence.applyVideoTransition`), `clip.audioGain`.
- Il cartone e gia renderizzato da HyperFrames: FilmCraft serve per il **mux finale** e per il controllo, non per animare.
  Verificare in produzione con un MP4 HyperFrames reale e lungo (qui provato solo su un MP4 sintetico da 3 s).

## 4. EffectCraft - post, card titolo a template  (TESTATO)

Stile After Effects. Opzioni globali: `--empty` (progetto nuovo), `--project f.ecproj`, `--save-as f.ecproj`.
Comandi in JSON Lines `{"command":..,"params":{..}}`.

**Comp + testo -> PNG/MP4** (`prove-craft/ec1.jsonl`):
```
E="/c/Program Files/EffectCraft/effectcraft-cli.exe"
{"command":"comp.new","params":{"name":"Titolo","width":1920,"height":1080,"frameRate":30,"duration":3,"background":"#F4EBD9"}}
{"command":"layer.newText","params":{"text":"Nodomatic","position":[960,540],"size":140,"fill":"#2A3A55"}}

"$E" --empty run --commands ec1.jsonl --save-as ec1.ecproj --json
"$E" --project ec1.ecproj render-frame --comp Titolo --time 1 --out ec1_frame.png      # 25 ms
"$E" --project ec1.ecproj render --comp Titolo --out ec1.mp4 --format h264              # 3 s in 0,9 s
```
Il testo e centrato, font Inter di default; `layer.newText` accetta `font`, `style`, `size`, `fill`, `stroke`.
`text.fonts` elenca i font; `layer.applyTextPreset {"preset":"typewriter|fadeUpCharacters|blurIn|..."}` per animare (NON TESTATO).

**Template `.ectemplate` con valori JSON (card titolo / terzo inferiore riusabile)**:
```
# 1. una volta sola: esporre le proprieta (Essential Graphics) e salvare il template
{"command":"essential.addProperty","params":{"comp":1,"layer":2,"path":"text/sourceText","name":"Titolo"}}
{"command":"essential.addProperty","params":{"comp":1,"layer":2,"path":"transform/position","name":"Posizione"}}
{"command":"essential.exportTemplate","params":{"comp":1,"path":"C:/.../titolo.ectemplate","name":"Titolo card"}}
"$E" --project ec1.ecproj run --commands ec2.jsonl --json

# 2. per ogni card: un JSON di valori (chiave = nome del controllo)
echo '{"Titolo":"Ore nove e quindici","Posizione":[960,300]}' > vals2.json
"$E" render-frame --template titolo.ectemplate --values vals2.json --time 1 --out tpl_frame.png
"$E" render        --template titolo.ectemplate --values vals2.json --out tpl.mp4 --format h264
```
Verificato: con `{"Titolo":"Buongiorno!"}` il PNG mostra il nuovo testo. Le proprieta si trovano con `"$E" --project f.ecproj props <comp> <layer> --flat`
(percorsi tipo `text/sourceText`, `transform/position`, `transform/opacity`). Layer e comp si indicano con id, `#n` o nome.
`templates.create` apre un template come progetto.

**Post su footage/PNG** (`prove-craft/ec3.jsonl`): `file.import` -> `comp.new` -> `layer.addItem {"item":1}` -> `effect.apply {"effect":"CC Vignette"}` +
`{"effect":"Add Grain"}` -> `render` (3 s in 2,9 s). Elenco effetti con parametri: `"$E" exec effect.list --params '{"filter":"grain"}'`.
Con `layer.addItem` su un MP4 si puo fare grana/vignettatura/correzione colore sul video di HyperFrames (NON provato su video lungo).
Altro: `render --format png|prores|webm|gif|exr`, `--channels rgba` per overlay con trasparenza (NON TESTATO), `script --eval` (JavaScript stile AE, NON TESTATO).

## 5. VectorCraft - SVG -> PNG/PDF, testo in tracciati  (TESTATO)

```
V="/c/Program Files/VectorCraft/vectorcraft-cli.exe"
"$V" convert testo.svg testo.png --scale 2                      # raster: --scale = pixel per punto
"$V" convert testo.svg testo.pdf
"$V" convert testo.svg testo_outline.svg --outline-text         # testo -> <path>: nessuna dipendenza dai font
```
Provato su `prove-craft/testo.svg` (Arial 72 bold, "Nodomatic badge"): il file `testo_outline.svg` ha 0 elementi `<text>`, solo `<path>`,
e renderizzato (`testo_outline.png`) e identico al testo, accento compreso. Utile per **loghi/scritte con font che HyperFrames non trova**
e per avere SVG robusti. Opzioni: `--artboard N`, `--decimals N` (precisione coordinate), formati scrivibili svg, pdf, png, jpg, webp, eps, psd, dxf...
Batch: `vectorcraft-cli run --in f.svg --cmd ... --export out.png --scale 2`. Warnings e `importWarnings` sono nel JSON stampato.

## 6. PhotoCraft e LightCraft - batch colore, resize, texture carta  (TESTATO)

**PhotoCraft batch** (lista azioni JSON applicata a tutti i file di una cartella, `prove-craft/carta.actions.json`):
```
{"actions":[
 ["image.imageSize",{"width":1920,"resample":"lanczos"}],
 ["image.adjustments.photoFilter",{"filter":"sepia","density":20}],
 ["filter.noise.addNoise",{"amount":4,"distribution":"gaussian","monochromatic":true,"seed":7}]
]}
P="/c/Program Files/PhotoCraft/photocraft-cli.exe"
"$P" batch --actions carta.actions.json --in batch_in --out batch_out        # 3 PNG in 0,4 s
```
Risultato: 3 PNG 960x540 -> 1920x1080 (`info` lo conferma). Il formato d'uscita e quello del file in ingresso (`<nome>.<ext>`).
Se si passa solo `width`, l'altezza e proporzionale. Parametri di ogni comando: `"$P" commands --json` (campo `params`).
Altri comandi utili: `image.adjustments.hueSaturation|vibrance|brightnessContrast|colorBalance|levels|curves`, `filter.blur.gaussianBlur`, `filter.sharpen.unsharpMask`.
**Contact sheet: non trovato un comando dedicato** (NON TESTATO) - per un provino meglio il deck DeckCraft (`insert.photoAlbum`, NON TESTATO) o il deck storyboard.

**Texture carta + overlay** (TESTATO, `p1_carta.png`):
```
# genera la carta (0,2 s)
"$P" run --new '{"width":1920,"height":1080}' --cmd edit.fill --params '{"contents":"color","color":"#f1e6cf"}' \
  --cmd filter.noise.addNoise --params '{"amount":9,"distribution":"gaussian","monochromatic":true,"seed":3}' \
  --cmd filter.blur.gaussianBlur --params '{"radius":0.8}' --out carta.png
# sovrapponila in Multiply sul pannello e appiattisci
"$P" run pannello.png --cmd file.placeEmbedded --params '{"path":"C:/.../carta.png","smartObject":false,"fit":true}' \
  --cmd layer.setProps --params '{"blend":"Multiply","opacity":0.55}' --cmd layer.flattenImage --out pannello_carta.png
```
`--cmd` e `--params` vanno in coppia (il `--params` si riferisce al `--cmd` che lo precede). `--out` non puo coincidere con il file in ingresso (serve `--in-place`).
Per un video con carta in sovrimpressione e piu pratico farlo in EffectCraft/HyperFrames (livello con blend multiply).

**LightCraft** (sviluppo foto/RAW): `lightcraft-cli render in.png -o out.png --set light.exposure=0.3 --set color.saturation=-10 --size 1920` funziona (1,4 s)
ma **`--size` non ingrandisce** (resta 960x540) e per PNG di cartone non porta nulla oltre a PhotoCraft. **Attenzione**: `lightcraft-cli run --demo app.export` scrive
`LCxxxxx.jpg` nella cartella corrente (e successo: `prove-craft/LC01347.jpg`). Valore per noi: marginale.

## 7. GridCraft - shot list CSV -> XLSX  (TESTATO)

Generazione della shot list: `python prove-craft/genera_shotlist.py` legge `audio/script_cartoon.json`
e scrive `docs/shot-list.csv` (UTF-8; colonne: n, scena, battuta, testo sottotitolo, indicazione recitazione,
durata stimata s, inquadratura, personaggi, suono). Durata = `len(testo)/13 + pre`. Totale stimato: **81,9 s (circa 1,4 min), 14 righe**.
Le colonne inquadratura/personaggi/suono sono inferite a mano dalle scene (dizionario `INF` nello script): vanno riviste da chi dirige.

Conversione in XLSX formattato (`prove-craft/shotlist.jsonl`; una riga `id={json}`):
```
G="/c/Program Files/GridCraft/gridcraft-cli.exe"
file.open={"path":"C:/.../docs/shot-list.csv"}
home.bold={"range":"A1:I1","on":true}
home.fillColor={"range":"A1:I1","color":"#2A3A55"}
home.fontColor={"range":"A1:I1","color":"#FFFFFF"}
home.columnWidth={"cols":"C:C","width":330}          # larghezza in pixel; una riga per colonna
home.wrapText={"range":"A1:I15","on":true}
cell.set={"cell":"E16","input":"TOTALE (s)"}
cell.set={"cell":"F16","input":"=SUM(F2:F15)"}
cell.set={"cell":"G16","input":"=TEXT(F16/60,\"0.0\")&\" min\""}
home.bold={"range":"A16:I16","on":true}
home.borders={"range":"A16:I16","preset":"top","style":"medium"}
view.freezeTopRow={}
file.saveAs={"path":"C:/.../docs/shot-list.xlsx"}

"$G" run --script shotlist.jsonl --quiet        # (--script accetta anche  {"command":..,"params":..})
```
Verifica: `"$G" cat docs/shot-list.xlsx --formulas --range A15:G16` mostra `=SUM(F2:F15)` -> 81,9 e "1.4 min";
`cell.get` conferma intestazione grassetto bianco su blu e riga totale in grassetto. `"$G" convert shot-list.xlsx x.pdf` = 4 pagine.
Gotcha: **`home.bold` senza `"on":true` fa da interruttore e puo spegnere** (mi ha dato `on:false` sulla riga totale): indicare sempre `on`.
Il CSV e letto correttamente in UTF-8 (accenti ok). Altri formati di conversione: `gridcraft-cli convert in out` per xlsx/csv/tsv/json/html/pdf.

## 8. WordCraft + PdfCraft - copione DOCX e PDF  (TESTATO)

Via piu semplice: **Markdown -> DOCX -> PDF** con `convert` (il formato lo decide l'estensione). `python prove-craft/genera_copione_md.py`
scrive `prove-craft/copione.md` (titolo `#`, per scena `## Scena: ...`, battuta `**n.** testo`, recitazione in `*(...)*`), poi:
```
W="/c/Program Files/WordCraft/wordcraft-cli.exe"
"$W" convert copione.md ../copione.docx
"$W" convert ../copione.docx ../copione.pdf
"$W" render ../copione.docx pagina1.png --page 1      # per controllare
```
Risultato: `docs/copione.docx` (7,5 KB, 2 pagine, titolo + 11 scene con intestazione e indicazioni in corsivo) e `docs/copione.pdf`. Nel copione uso il testo del campo `caption` (ortografia corretta),
non quello fonetico di `text` (che serve solo a ElevenLabs). Stili: titolo e intestazioni in blu-petrolio, corpo Arial/Liberation.
Alternativa con comandi nativi (`text.insert`, `text.newParagraph`, `format.italic`, `run --script`) NON TESTATA: il Markdown e piu rapido.
Altre conversioni: docx, odt, rtf, html, md, tex, txt, png, pdf. `"$W" text file.docx` stampa il testo.

**PdfCraft** (TESTATO come utilita):
```
Pd="/c/Program Files/PdfCraft/pdfcraft-cli.exe"
"$Pd" combine storyboard.pdf ../copione.pdf --out combinato.pdf        # storyboard + copione in un PDF (3 pagine, esempio)
"$Pd" extract ../copione.pdf --pages 2 --out copione_p2.pdf
"$Pd" render storyboard.pdf --page 1 --dpi 96 --out sb_pdf_p1.png       # 1280x720 in 29 ms
"$Pd" text ../copione.pdf --page 2 ; "$Pd" info ../copione.pdf          # testo / metadati JSON
```
Altro (NON TESTATO): `split`, `edit` (ruota, elimina, sposta pagine, titolo/autore), `stamp_custom`, `form_*`, `redact_*`, OCR in `batch`. WordCraft esporta gia il PDF: PdfCraft serve solo a unire/estrarre.

## 9. Le altre app, in una riga (verifica rapida)

- **DesignCraft** (DTP stile InDesign): `designcraft-cli run --sample --export x.pdf` funziona (4 pagine); per noi duplica DeckCraft/WordCraft. Marginale. (Creazione di un layout da zero: NON TESTATA.)
- **FigCraft** (stile Figma): TESTATO `frame.create` + `node.create {"type":"rectangle"}` + `text.create` -> `figcraft-cli run s.jsonl --out f.figcraft` ->
  `figcraft-cli export f.figcraft --out f.png` (frame 1920x1080, testo Inter, 60 KB; sfondo bianco, trasparenza NON verificata). Possibile per terzi inferiori vettoriali, ma EffectCraft/HTML bastano. Marginale.
- **MaterialCraft** (stile Substance): TESTATO `materialcraft-cli new fabric f.mcraft --size 512` + `export f.mcraft cartella --format png` -> basecolor/normal/height/roughness... in 0,1 s.
  Ha nodi `grunge_map`, `clouds`: texture carta/grana tileable procedurali. Marginale (la carta di PhotoCraft basta).
- **TextureCraft** (pittura texture su mesh 3D): non utile per un 2D.
- **CADCraft** (CAD 2D stile AutoCAD: dxf/dwg -> svg/pdf): non utile. (Il `sample` va chiamato con un percorso valido; non approfondito.)
- **SolidCraft** (CAD solido): non utile.
- **Craft Anything** (analisi/reverse engineering di app e file, agente AI): `capabilities` mostra motori mancanti (Ghidra, adb...); non utile.

## Raccomandazione per il flusso del cartone

1. Copione/shot list: `genera_shotlist.py` -> GridCraft (xlsx); `genera_copione_md.py` -> WordCraft (docx/pdf).
2. Storyboard: `genera_storyboard_deck.py` -> DeckCraft (pdf/pptx) con i pannelli PNG.
3. Audio: sintesi in Python -> SoundCraft per mix (offset/gain) e **misura LUFS**.
4. Video: HyperFrames -> MP4 senza audio -> (opz.) EffectCraft per grana/vignetta/card -> FilmCraft: video + mix WAV, export H.264 con `loudnessLufs:-15`.
5. Controllo: FilmCraft `probe`, SoundCraft per ri-misurare l'MP4 finale.

## File lasciati in `prove-craft/` (output di prova)

Script e comandi: `storyboard.cmds`, `storyboard_full.cmds`, `mix.sc`, `gen.jsonl`, `edit.jsonl`, `frame.jsonl`, `ec1.jsonl`, `ec2.jsonl`, `ec3.jsonl`, `fig1.jsonl`,
`shotlist.jsonl`, `carta.actions.json`, `mkwav.py`, `genera_shotlist.py`, `genera_copione_md.py`, `genera_storyboard_deck.py`.
File di scarto creati nei test (non ho potuto cancellarli, la guardia blocca la rimozione): `frame_%05d000.png`, `frame_%05d001.png`, `_tmp_misura.wav`, `LC01347.jpg`;
si possono eliminare a mano.
