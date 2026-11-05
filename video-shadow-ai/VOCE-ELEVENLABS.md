# La voce del Contratto: come generarla su ElevenLabs

## 1. Scegli la voce
Il Contratto è **orgoglioso, puntiglioso e ironico**: un impiegato modello che commenta, con una battuta asciutta,
quello che gli succede.
- Nella **Voice Library** cerca *Italian*, con caratteristiche come *conversational*, *warm*, *narrator* o *characterful*.
- Va bene una voce maschile o femminile, purché italiana madrelingua, con un sorriso nella voce e non da speaker di telegiornale.
- Prova la battuta n. 4 («Gratis. Che fortuna.») con 2–3 voci e tieni quella che la rende più asciutta.

## 2. Impostazioni consigliate
| Impostazione | Valore |
|---|---|
| Modello | **Eleven v3** se disponibile (permette i tag di recitazione tra parentesi quadre), altrimenti **Multilingual v2** |
| Stability | 40–50 % (più bassa = più espressiva) |
| Similarity | 75 % |
| Style | 25–35 % |
| Speaker boost | attivo |
| Formato | MP3 44,1 kHz 192 kbps, oppure WAV |

## 3. Il testo da incollare
Generalo **tutto in un'unica ripresa**, così il tono resta continuo. I tempi li taglio io con faster-whisper.
I tag tra parentesi quadre valgono solo per **Eleven v3**. Se usi Multilingual v2, cancellali: le indicazioni
di recitazione sono già nei puntini e nella punteggiatura.

```
[orgoglioso] Buongiorno! Sono un contratto. Tre pagine, due firme... [abbassando la voce] e parecchi dati riservati.

[allarmato] Ore nove e quindici. Qualcuno ha fretta.

[incredulo] Mi copia... e mi incolla in un chatbot trovato online.

[sarcastico] Gratis. Che fortuna.

Nessuno l'ha vietato. Nessuno l'ha autorizzato. [pausa] Diciamo che... nessuno ha chiesto.

[veloce] Trenta secondi, ed è fatto.

[spaesato] Peccato che adesso io sia qui. Non so nemmeno in che paese.

[chiaro, scandito] Questa si chiama Shadow AI.

[serio] E il rischio è doppio. Per l'azienda: dati dei clienti esposti, informazioni riservate fuori controllo. Per te: errori che firmi tu, e regole aziendali violate.

[sottovoce, ammiccante] Firmate da te, eh.

[deciso] Riavvolgiamo.

[sollevato] Stessa azienda, tre cose in più. Uno strumento approvato. Regole chiare. Persone formate.

[soddisfatto] Il lavoro si fa lo stesso, in trenta secondi. [sorridendo] E io... resto a casa.
```

**Pronuncia:** "Shadow AI" va detto all'inglese (*sciàdou ei-ài*). Se la voce lo legge all'italiana, scrivi
`Sciàdou èi-ài` solo in quella riga.

## 4. Mandamelo
Caricalo qui in chat (un file di circa 1 minuto pesa intorno a 1–2 MB). Se una battuta non ti convince,
rigenera solo quella e mandami anche il file corretto, con il numero della battuta.
