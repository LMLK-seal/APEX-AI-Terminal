# Guida di stile del cartone

Le regole da rispettare perché ogni nuovo video sembri lo stesso cartone, della stessa serie.
Riferimenti visivi: `design/foglio-modello.png` (personaggi), `design/immagine-chiave-*.png` (ambienti e luce),
`design/cartone-v2.mp4` (movimento e ritmo).

## 1. Il mondo
- **Solo oggetti antropomorfi, mai persone.** Documenti, mouse, server, chatbot... Gli umani esistono solo fuori
  campo: si capisce che "qualcuno ha fretta" dalle cose che si muovono (un cursore, una tazza, un orologio).
- **Tono leggero e ironico.** Il protagonista commenta quello che gli succede con una battuta asciutta.
  Il messaggio formativo arriva dalla storia, non da una lezione.
- **Lieto fine con una regola chiara.** Il pubblico deve uscire con 2-3 comportamenti concreti da ricordare.

## 2. Disegno
| Regola | Valore |
|---|---|
| Formato | 1920×1080, 30 fps, 16:9 |
| Contorno | inchiostro `#1c2230`, spessore 5 px (in `characters.js`: `S = 5`), estremità arrotondate |
| Riempimenti | piatti, un colore base + un'ombra; i gradienti solo per luci, bagliori e riflessi |
| Occhi | grandi, ovali, bianchi con pupilla nera e riflesso; sopracciglio unico e molto espressivo |
| Guance | due ovali rosati, per le emozioni positive |
| Ombre a terra | un'ellisse morbida sotto ogni personaggio (`Anim.shadow`) |
| Sfondi | vettoriali, semplici, un po' più spenti dei personaggi, con texture di carta leggera |

### Tavolozza (`Chars.P` in `cartoon/characters.js`)
| Uso | Colori |
|---|---|
| Inchiostro e carta | ink `#1c2230` · paper `#f6f1e6` · paperShade `#e2d6bf` · paperLine `#b9ab92` |
| Ufficio "a casa" (caldo) | kraft `#c99d6c` · kraftShade `#a67b4d` · copper `#b5562a` · copperLight `#de8a57` · brass `#d8a24a` |
| Fuori, rischio (freddo) | navy `#141f2e` · navyMid `#24354f` · server `#1d2a44` · led `#ff4d3d` · ledAmber `#ffb43d` |
| Strumento non approvato | gratis `#ffd23f` · gratisShade `#f0b400` · gratisPink `#e0457b` |
| Strumento approvato, sicurezza | teal `#2f8f83` · tealShade `#226c63` · tealLight `#5fb3a7` |
| Mouse | mouse `#8e9db4` · mouseShade `#6f7f98` · mouseLight `#b4c0d2` |

Regola dei colori: **caldo = al sicuro, in azienda**; **freddo e blu notte = fuori controllo**;
**giallo e rosa "gratis" = tentazione rischiosa**; **verde acqua = la scelta giusta**.

## 3. Personaggi ricorrenti (`cartoon/characters.js`, `cartoon/chatbots.js`)
| Personaggio | Chi è | Funzione |
|---|---|---|
| **Il Contratto** (`Chars.contratto`) | foglio con firma, orgoglioso, puntiglioso, ironico | protagonista e voce narrante |
| **Il Mouse** (`Chars.mouse`) | collega con il tocco da laureato; corre, suda, poi si gode il caffè | spalla comica, rappresenta il dipendente |
| **ChatGratis** (`Chatbots.genieGratis`) | genio digitale giallo con badge GRATIS e pixel | lo strumento non autorizzato |
| **L'Assistente** (`Chatbots.genieAssist`) | genio verde acqua con papillon e lucchetto, scritta "IN SEDE" | lo strumento approvato dall'azienda |
| **I Server** (`Chars.server`) | armadi blu notte con LED rossi, lontani e freddi | il "dove finiscono i dati" |

I geni escono dallo schermo del portatile con `Anim.genieOut` (anticipazione, salita, poi ondeggiano seduti).
Per un nuovo episodio si possono aggiungere personaggi, ma con lo stesso disegno: contorno 5 px, occhi
come `Chars.eye`, bocche da `Chars.mouth`, braccia da `Chars.arm`.

## 4. Animazione
- **Chiavi con curve GSAP:** ogni posa è una chiave `[tempo, proprietà, curva]` letta da `Anim.at`.
  Curve tipiche: `back.out` per gli scatti e le comparse, `power2.inOut` per gli spostamenti, `power2.in` per
  le anticipazioni.
- **Principi classici:** anticipazione prima di ogni salto, compressione e allungamento (`sq`), sovrapposizione
  dei movimenti (le braccia arrivano dopo il corpo), respiro idle sempre attivo, battito delle palpebre ogni 2-4 s.
- **Lip-sync "a passo due":** la bocca cambia 15 volte al secondo, con i visemi italiani ricavati dal testo.
- **I gesti cadono sulle parole:** nelle coreografie i tempi si scrivono come `W(i, "parola")`, mai in secondi fissi.
  Così, se cambia la voce, l'animazione la segue.
- **Camera:** spostamenti e zoom lenti (`Anim.cam`). Non esce mai dallo sfondo, e scuote solo negli urti (`Anim.shake`).

## 5. Tipografia
| Uso | Font |
|---|---|
| Sottotitoli | Inter 600, 40 px, testo crema su pillola scura semitrasparente, in basso al centro |
| Etichette e cartellini | Inter 800, maiuscolo, spaziatura larga, su cartellino di carta (`.tag`) |
| Titoli | EB Garamond 700 |
| Numeri, orari, codici | IBM Plex Mono 700 |

Le tavole di design usano i font in `cartoon/fonts/`; nel render HyperFrames incorpora gli stessi font da Google Fonts.

## 6. Suono
- **Voce:** ElevenLabs, una voce italiana "con un sorriso", non da telegiornale (vedi `VOCE-ELEVENLABS.md`).
- **Musica, sintetizzata in `tools/make_audio_cartoon.py`:** "a casa" con marimba, pizzicato e shaker in Do maggiore
  a 112 BPM; "fuori" in La minore con pulsazione bassa e campane fredde; riavvolgimento con effetto "tape stop";
  finale in maggiore e "ta-da" sul cartello.
- **Effetti:** un suono per ogni azione visibile (pop, corsa, risucchio, timbro, scintille).
- **Mix:** circa -15 LUFS integrati; la voce sta sempre sopra la musica.

## 7. Struttura di un episodio (circa 60-70 s)
1. **Presentazione:** il protagonista si presenta (chi è, cosa contiene).
2. **La tentazione:** qualcuno ha fretta e prende la scorciatoia.
3. **Nessuno ha detto niente:** la zona grigia.
4. **Il viaggio:** cosa succede davvero (i dati escono, vanno lontano).
5. **Il nome:** si dà un nome al concetto (es. "Shadow AI").
6. **Il rischio:** per l'azienda e per la persona.
7. **Riavvolgiamo:** si torna indietro nel tempo.
8. **Il modo giusto:** le 2-3 regole, con una lavagna a spunte.
9. **Lieto fine:** stesso risultato, stessa velocità, tutto al sicuro.
10. **Cartello finale:** titolo del corso e numero della lezione.
