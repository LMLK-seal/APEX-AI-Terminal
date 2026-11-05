# Genera copione.md (Markdown) dal JSON; poi WordCraft converte in docx e pdf.
import json, os
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(BASE, '..', '..'))
d = json.load(open(os.path.join(ROOT, 'audio', 'script_cartoon.json'), encoding='utf-8'))
out = ["# Copione - Il badge di Nodomatic", "",
       "Cartone 2D, durata stimata circa 1 min 22 s. Voce: il badge (Chars.badge). Le indicazioni di recitazione sono in corsivo.", ""]
scena = None; n = 0
for x in d:
    n += 1
    if x['scene'] != scena:
        scena = x['scene']
        out += [f"## Scena: {scena}", ""]
    pre = f" (pausa prima: {x['pre']} s)" if x.get('pre') else ""
    out += [f"**{n}.** {x['caption']}", "", f"*({x['direction']}){pre}*", ""]
open(os.path.join(BASE, 'copione.md'), 'w', encoding='utf-8').write("\n".join(out))
print('ok', n)
