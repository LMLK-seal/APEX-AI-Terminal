# Genera docs/shot-list.csv e docs/copione.json-friendly dati dal JSON del copione.
import json, csv, os
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, '..', '..'))
d = json.load(open(os.path.join(ROOT, 'audio', 'script_cartoon.json'), encoding='utf-8'))
# (inquadratura, personaggi, suono) per battuta, in ordine
INF = [
 ("Campo medio frontale, badge al centro; leggero push-in", "Badge (Chars.badge), porta sullo sfondo", "Ambiente ufficio, passi, tintinnio del clip"),
 ("Primo piano badge, sguardo in camera; poi panoramica sulla stanza", "Badge; ombre/sagome di genie e pacchi", "Pacchi che strisciano, 'ding' digitale leggero"),
 ("Campo medio a due: badge a sinistra, genio Copilot sulla soglia", "Badge; genio Copilot (approvato, teal)", "Bip di consenso del lettore badge, porta che scatta"),
 ("Mezzo busto badge con dito alzato; stacco sul genio Claude (teal)", "Badge; genio Claude (approvato, licenza aziendale)", "Bip di consenso, 'tic' del dito"),
 ("Piano d'insieme dell'ufficio, orologio 9:15 in primo piano; zoom su Mouse", "Mouse (impiegato di fretta), badge in secondo piano", "Tic-tac accelerato, tazza che trema"),
 ("Campo largo sulla finestra che si apre; due ingressi dal davanzale", "Genio ChatGPT e genio Perplexity (non autorizzati, gialli), Mouse, badge", "Finestra che sbatte, fanfara comica 'gratis', ronzio"),
 ("Primo piano badge con sopracciglio alzato; inserto sul genio Claude (account personale, giallo)", "Badge; genio Claude (account personale = non autorizzato)", "Allarme breve del lettore badge, buzzer sbagliato"),
 ("Primo piano Mouse che si giustifica, rapido; poi pausa fissa sul badge", "Mouse; badge", "Voce/mormorio di scuse, silenzio di mezzo secondo"),
 ("Campo medio sul file-documento che esce dalla finestra nella notte (freddo, navy)", "File/documento antropomorfo, server nel buio; badge fuori campo", "Whoosh, sirena lontana, LED rosso che pulsa"),
 ("Piano medio badge a figura intera, titolo grafico 'strumento non autorizzato' sovraimpresso", "Badge; sagome dei genie gialli sullo sfondo", "Colpo secco di timbro, sting musicale"),
 ("Primo piano badge che si volta verso il proprio nome; stacco su server con LED rossi", "Badge; server; Mouse piccolo in campo", "Tono basso, ronzio server, 'ding' sul nome"),
 ("Effetto rewind: stessa inquadratura dell'apertura scorre indietro", "Badge, Mouse, genie (in rewind)", "Nastro che si riavvolge, whoosh inverso"),
 ("Tre inquadrature a elenco: genio Copilot / genio Claude / badge con IT", "Badge; genio Copilot e genio Claude (teal); sagoma IT", "Tre 'ding' ritmati, musica positiva in entrata"),
 ("Campo medio ufficio caldo, tutti i personaggi rilassati; chiusura sul badge", "Badge, Mouse, genie approvati (teal)", "Musica di chiusura, bip di consenso, tintinnio finale"),
]
assert len(INF) == len(d), (len(INF), len(d))
hdr = ["n","scena","battuta","testo sottotitolo","indicazione recitazione","durata stimata s","inquadratura","personaggi","suono"]
rows = []
for i,(x,(inq,pers,suono)) in enumerate(zip(d,INF),1):
    dur = round(len(x['text'])/13 + x.get('pre',0), 1)
    rows.append([i, x['scene'], x['text'], x['caption'], x['direction'], dur, inq, pers, suono])
out = os.path.join(BASE, '..', 'shot-list.csv')
with open(out,'w',encoding='utf-8',newline='') as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
print(out, len(rows), 'righe; durata totale', round(sum(r[5] for r in rows),1))
