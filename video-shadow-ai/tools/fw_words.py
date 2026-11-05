"""faster-whisper: tempi parola per parola della voce definitiva, per sottotitoli e timbri al frame giusto.
- trascrive il file di ogni battuta (line["wav"], es. audio/eleven/NN.wav) e lo riporta sui tempi della timeline (modello small, italiano, CPU int8)
- abbina le parole alle battute per POSIZIONE (non per testo: le ripetizioni non si confondono)
- scrive line["words"] in timeline.json senza toccare inizi/fine delle battute (le scene restano valide)
- stampa quanto la trascrizione si discosta dal copione (controllo di pronuncia della voce)
Richiede huggingface.co abilitato (Systran/faster-whisper-small).
Uso: .venv/bin/python tools/fw_words.py
"""
import json, re, difflib, os
from types import SimpleNamespace
from faster_whisper import WhisperModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tl = json.load(open(f"{ROOT}/timeline.json"))
model = WhisperModel("small", device="cpu", compute_type="int8")
norm = lambda t: re.sub(r"[^\wàèéìòù]", "", t.lower())
heard, words = [], []
for l in tl["lines"]:
    # ogni battuta dal suo file: sulla traccia montata la prima parola finiva a volte nella battuta precedente
    segs, _ = model.transcribe(f"{ROOT}/{l['wav']}", language="it", word_timestamps=True, beam_size=5)
    off = l["start"] - l["trim_start"]
    ws = [SimpleNamespace(word=w.word, start=w.start + off, end=w.end + off) for s in segs for w in s.words]; words += ws
    script_words = l["caption"].split()
    # allineamento per posizione tra parole del copione e parole sentite
    sm = difflib.SequenceMatcher(a=[norm(x) for x in script_words], b=[norm(w.word) for w in ws])
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        for k in range(i2 - i1):
            j = j1 + min(k, max(0, j2 - j1 - 1)) if j2 > j1 else None
            if j is not None and j < len(ws):
                out.append(dict(w=script_words[i1 + k], start=round(ws[j].start, 3), end=round(ws[j].end, 3)))
    l["words"] = out
    heard.append((l["caption"], " ".join(w.word.strip() for w in ws), sm.ratio()))
json.dump(tl, open(f"{ROOT}/timeline.json", "w"), ensure_ascii=False, indent=1)
bad = [h for h in heard if h[2] < 0.8]
print(f"parole: {len(words)}  battute con pronuncia da ricontrollare: {len(bad)}")
for c, h, r in bad: print(f"  {r:.2f}  copione: {c}\n        sentito: {h}")
