# Genera storyboard.cmds (DeckCraft) dal copione + shot list, una slide per battuta.
# Uso: python genera_storyboard_deck.py  -> poi:
#   deckcraft-cli run --cmd-file storyboard_full.cmds --save storyboard_full.deckcraft --export storyboard_full.pdf
import json, csv, os, json as J
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, '..', '..'))
PANELS = os.path.join(ROOT, 'storyboard', 'panels')
FALLBACK = os.path.join(PANELS, 'prova.png')
def panel(n):  # cerca NN.png / panel-NN.png, altrimenti prova.png
    for name in (f'{n:02d}.png', f'panel-{n:02d}.png', f'{n}.png'):
        p = os.path.join(PANELS, name)
        if os.path.exists(p): return p
    return FALLBACK
rows = list(csv.DictReader(open(os.path.join(ROOT, 'docs', 'shot-list.csv'), encoding='utf-8')))
L = ['file.new=' + J.dumps({"blank": True, "size": [960, 540]})]
for r in rows:
    n = int(r['n'])
    paras = [  # (testo, size, bold, italic, colore)
        (f"{n}. Scena: {r['scena']}  ({r['durata stimata s']} s)", 20, True, False, None),
        (r['testo sottotitolo'], 15, False, False, None),
        (f"Recitazione: {r['indicazione recitazione']}", 12, False, True, None),
        (f"Inquadratura: {r['inquadratura']}", 11, False, False, None),
        (f"Suono: {r['suono']}", 11, False, False, '#B03030'),
    ]
    L.append('slide.new=' + J.dumps({"layout": "blank"}))
    L.append('design.background=' + J.dumps({"color": "#F4EBD9"}))
    L.append('insert.picture=' + J.dumps({"path": panel(n).replace(os.sep, '/'), "rect": [24, 40, 620, 349]}))
    L.append('insert.textBox=' + J.dumps({"rect": [664, 40, 272, 460], "text": "\n".join(p[0] for p in paras)}, ensure_ascii=False))
    L.append('text.edit')
    for i, (t, size, b, it, col) in enumerate(paras):
        L.append('text.select=' + J.dumps({"anchor": [i, 0], "caret": [i, len(t)]}))
        L.append('format.size=' + J.dumps({"size": size}))
        if b: L.append('format.bold=' + J.dumps({"on": True}))
        if it: L.append('format.italic=' + J.dumps({"on": True}))
        if col: L.append('format.color=' + J.dumps({"color": col}))
    L.append('text.exit')
open(os.path.join(BASE, 'storyboard_full.cmds'), 'w', encoding='utf-8').write("\n".join(L) + "\n")
print('comandi:', len(L))
