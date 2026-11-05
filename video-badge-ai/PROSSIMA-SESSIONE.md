# Dove siamo e come ripartire (episodio "Sono un badge")

Ramo: `video-sono-un-badge`. Metodo e decisioni: [`METODO.md`](METODO.md).

## Fatto
- Copione v1: `audio/script_cartoon.json` (14 battute, 11 scene + cartello, ~82 s stimati). Azienda di fantasia **Nodomatic**.
- Loghi ufficiali in `loghi/`, generati in `cartoon/loghi.js` con `node tools/make_loghi.js`.
- Badge, geni con logo, porta, finestra (`cartoon/badge.js`), schermi degli strumenti in `Scenes.office({screen})`.
- **Storyboard v1**: 12 pannelli in `storyboard/panels/p01..p12.png` (script `storyboard/panels_a.js`, `panels_b.js`),
  disegnati ancora con lo stile grafico del primo episodio.
- Ricette provate delle app Craft: `docs/CRAFT-RICETTE.md`; shot list `docs/shot-list.xlsx`; copione `docs/copione.docx/.pdf`.
- Ambiente di render su Windows pronto: `tools/setup_windows.ps1`, poi `. .\.tools\env.ps1` (vedi `docs/AMBIENTE-WINDOWS.md`).

## Decisioni ancora valide
- **2,5D**: personaggi 2D su piani a profondità diverse, camera con parallasse, luci e ombre (non 3D completo).
- **Personaggi da rifare**: più belli e diversi dal primo episodio, stessi ruoli e stesso significato dei colori.
- Loghi ufficiali interi, mai deformati; verde acqua = autorizzato, giallo "gratis" = non autorizzato.

## In corso al momento del salvataggio
Tre direzioni di stile per i nuovi personaggi (foglio modello + prova di parallasse), in `cartoon/stili/`:
`a-carta` (carta ritagliata), `b-morbido` (morbido premium), `c-inchiostro` (inchiostro e acquerello).
Potrebbero essere incomplete: controllare `sheet.png` e `depth.png` in ciascuna cartella.

## Prossimi passi
1. Rivedere i 3 fogli modello (completare quelli mancanti), sceglierne uno con l'utente.
2. Rifare i 12 pannelli nello stile scelto e montare lo storyboard in DeckCraft (ricetta in `docs/CRAFT-RICETTE.md`,
   generatore `docs/prove-craft/genera_storyboard_deck.py`), export PDF + PPTX; farlo approvare.
3. Confermare con l'IT le regole reali (Copilot e account, licenze Claude Team, a chi chiedere nuovi strumenti).
4. Voce ElevenLabs (chiave in `.env`, mai aprirla), timeline, regia 2,5D, suono, render, salvataggio.

## Prompt per ripartire
```
Nel repo Guggumama007/progetto-p-Iva, ramo video-sono-un-badge, cartella video-badge-ai:
leggi PROSSIMA-SESSIONE.md e METODO.md e riprendi dal primo dei "Prossimi passi".
Usa subagenti Sonnet in parallelo per i lavori separabili; tu rivedi e correggi.
```
