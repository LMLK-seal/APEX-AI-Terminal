# Fare un nuovo video nello stesso stile

Il cartone non è un file di animazione chiuso: è **codice**. Personaggi, oggetti, sfondi e movimenti sono disegnati
e animati da programmi. Per questo un nuovo episodio riusa tutto ciò che c'è già e cambia solo storia, voce e regia.
Le regole visive da rispettare sono in [`STILE.md`](STILE.md).

## Come è fatto un episodio
```
video-shadow-ai/
├── audio/script_cartoon.json   copione: 13 battute con scena, testo per la voce, sottotitolo, indicazione di recitazione
├── audio/pads_cartoon.json     respiro prima/dopo ogni scena (spazio per le gag senza voce)
├── audio/eleven/NN.wav         una battuta per file (voce ElevenLabs)
├── timeline.json               generato: inizio/fine di ogni scena e battuta, pause, parole
├── cartoon/
│   ├── characters.js           personaggi (Contratto, Mouse, Server) + tavolozza Chars.P
│   ├── chatbots.js             i geni digitali (ChatGratis, Assistente)
│   ├── props.js                oggetti di scena (lavagna, timbro, lucchetto, bolla, scintille...)
│   ├── scenes.js               sfondi (ufficio, sala server, rete, studio)
│   ├── anim.js                 motore: palcoscenico, chiavi, lip-sync, palpebre, camera, uscita dei geni
│   ├── shots/<scena>.html      LA REGIA: una coreografia per scena
│   ├── build.py                genera il progetto HyperFrames (cartoon/hf/) da timeline + coreografie
│   └── model-sheet.html, frame.html   foglio modello e immagini chiave (tavole di design)
├── tools/                      timeline, voce, parole, musica ed effetti, preparazione dell'ambiente
└── design/                     tavole e video renderizzati
```
Mezzi già pronti da riusare: 5 personaggi, 6 versioni di chatbot, 21 oggetti di scena
(`Props.checklist`, `stamp`, `padlock`, `shield`, `bubbleDome`, `folder`, `rulebook`, `sparkle`...),
5 ambienti con varianti (`Scenes.office`, `serverRoom`, `racks`, `network`, `studio`).

## Procedura (in una sessione di Claude Code)
1. **Copia la cartella** `video-shadow-ai` in una nuova, per esempio `video-lezione-4`, e lavora lì.
   Il primo episodio resta intatto. Gli script trovano i file a partire dalla propria posizione, quindi funzionano
   anche nella copia. Nella copia svuota `cartoon/shots/` e `cartoon/hf/compositions/` dalle scene che non servono.
2. **Concept e copione.** Scegli insieme a Claude il protagonista (un oggetto) e la storia, seguendo la struttura
   in 10 momenti di `STILE.md`. Scrivi `audio/script_cartoon.json`: il campo `scene` raggruppa le battute,
   e l'ordine delle scene segue il copione. Il cartello finale si aggiunge da solo.
3. **Bozzetti prima di animare.** Se servono personaggi o ambienti nuovi, aggiungerli in `characters.js`, `props.js`
   e `scenes.js`, poi renderizzare foglio modello e immagini chiave (`model-sheet.html`, `frame.html`)
   e approvarli **prima** di scrivere le scene.
4. **Voce.** Generare con ElevenLabs una battuta per file in `audio/eleven/00.wav ... NN.wav`
   (stessa voce e stesse impostazioni del primo episodio, salvate in `audio/eleven/meta.json`). In alternativa,
   una ripresa unica da tagliare con `tools/voice_to_timeline.py`.
5. **Timeline.** `python tools/timeline.py audio/script_cartoon.json audio/eleven audio/pads_cartoon.json audio/voice_eleven.wav`,
   poi `python tools/fw_words.py` per i tempi parola per parola.
6. **Regia.** Per ogni scena, un file `cartoon/shots/<scena>.html` (vedi sotto).
7. **Suono.** Gli effetti e i cambi di musica sono agganciati alle scene in `tools/make_audio_cartoon.py`:
   per nuove scene vanno riscritti gli agganci (stessi strumenti e stessi suoni).
8. **Controllo e render.** `python cartoon/build.py`, poi in `cartoon/hf`: `hyperframes check`, `hyperframes snapshot`
   per rivedere i fotogrammi chiave, infine `hyperframes render`. I comandi esatti sono in `PROSSIMA-SESSIONE.md`.

## Scrivere una scena (`cartoon/shots/<scena>.html`)
Ogni file contiene l'HTML dei cartellini in sovrimpressione e uno `<script>` con la regia. `build.py` gli mette
a disposizione:

| Nome | Cosa è |
|---|---|
| `DUR`, `S0` | durata della scena e suo inizio nel video |
| `L[i].a`, `L[i].b` | inizio e fine della battuta i-esima della scena (tempo locale) |
| `W(i, "parola")` | l'istante in cui viene detta una parola: **usare questo per ogni gesto** |
| `MOUTH(t)` | la bocca del protagonista (lip-sync) al tempo t |
| `$(selettore)` | trova un elemento della scena |
| `tl` | la timeline GSAP della scena |

Schema minimo:
```js
const KC = [                                   // chiavi del protagonista: [tempo, posa, curva]
  [0, { x: 960, y: 864, s: 1.2 }],
  [W(0, "Gratis") - 0.1, { sq: 0.93 }, "power2.in"],            // anticipazione
  [W(0, "Gratis"), { sq: 1.06, y: 830 }, "back.out(2)"],        // scatto sulla parola
];
const update = Anim.stage($("svg.stage"), Scenes.office({ warm: 1 }), Scenes.fx("warm"), (t) => {
  const c = Anim.at(KC, t); c.mouth = MOUTH(t); c.blink = Anim.blinkAt([1.2, 3.4], t);
  const a = Anim.shadow({ ...c, ground: 864 }, 135) + Anim.place(Anim.contratto(c, t), c);
  return { cam: { x: 960, y: 540, z: 1.05 }, actors: a };
});
Anim.drive(tl, DUR, update);
```
Gli esempi completi sono le 10 scene del primo episodio: `finale.html` (geni, bolla, lavagna),
`viaggio.html` (rete e server), `riavvolgi.html` (riavvolgimento).

## Prompt per iniziare un nuovo episodio
```
Nel repo Guggumama007/Progetto-P-Iva c'è il cartone "Sono un contratto" (cartella video-shadow-ai).
Voglio un nuovo episodio nello stesso stile: leggi video-shadow-ai/README.md, STILE.md e NUOVO-VIDEO.md.
Argomento della lezione: <ARGOMENTO>. Durata 60-70 secondi, 16:9.
Crea la cartella video-<nome> copiando video-shadow-ai, poi:
1) proponimi 2-3 concept (protagonista oggetto + storia nei 10 momenti di STILE.md) e aspetta la mia scelta;
2) scrivi il copione e fammelo approvare;
3) bozzetti di eventuali personaggi o ambienti nuovi (foglio modello, immagini chiave) e aspetta l'ok;
4) voce con ElevenLabs (la chiave è in .env come ELEVENLABS_API_KEY: non aprire né stampare il file), stessa voce del primo episodio;
5) animazione, suono, controllo con snapshot, render; mostrami il video e salva tutto su GitHub.
Solo oggetti antropomorfi, niente persone; tono leggero e ironico; nessun effetto "AI slop".
```
