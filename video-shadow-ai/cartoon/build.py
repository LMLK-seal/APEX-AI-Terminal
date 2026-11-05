"""Genera il progetto HyperFrames del cartone da timeline.json + coreografie in cartoon/shots/*.html.
Ogni coreografia e' un frammento: HTML degli elementi in sovrimpressione + <script> con la regia della scena.
Il generatore aggiunge davanti allo script le costanti della scena (tempi, battute) e un aiuto per il lip-sync.
Quando cambia la voce (ElevenLabs): rifare timeline.json, poi `python3 cartoon/build.py` -> tutto si riallinea.
"""
import json, os, re, shutil, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = f"{ROOT}/cartoon"; HF = f"{C}/hf"
os.makedirs(f"{HF}/compositions", exist_ok=True); os.makedirs(f"{HF}/assets/audio", exist_ok=True)
tl = json.load(open(f"{ROOT}/timeline.json")); D = tl["duration"]
S = {s["id"]: s for s in tl["scenes"]}
LINES = [{k: l[k] for k in ("scene", "start", "end", "caption", "pauses", "speech") if k in l} | ({"words": l["words"]} if l.get("words") else {}) for l in tl["lines"]]

for f in ("characters.js", "chatbots.js", "props.js", "scenes.js", "anim.js"): shutil.copy(f"{C}/{f}", f"{HF}/assets/{f}")
# gsap.min.js e paper.png sono fissi in cartoon/hf/assets (fanno parte del repo)

PAPER_CSS = """
  .tag { position:absolute; background:#f1ebde url(assets/paper.png); background-blend-mode:multiply; color:#1c2230;
         box-shadow: 0 3px 0 rgba(60,40,20,.25), 0 14px 30px rgba(0,0,0,.28); border-radius:4px; }
  .tag.cu { background-color:#b5562a; color:#fbf4ea; } .tag.navy { background-color:#16233a; color:#f1ebde; }
  .lbl { font-family:'Inter'; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
  .ttl { font-family:'EB Garamond'; font-weight:700; }
  .mono { font-family:'IBM Plex Mono'; font-weight:700; }
"""

def scene(sid, start, dur, shot_file):
    frag = open(shot_file).read()
    body, script = frag.split("<script>", 1); script = script.rsplit("</script>", 1)[0]
    prelude = f"""
  const SID = "{sid}", DUR = {dur:.3f}, S0 = {start:.3f};
  const LINES = {json.dumps(LINES, ensure_ascii=False)};
  const L = LINES.filter((l) => l.scene === SID).map((l) => ({{ ...l, a: l.start - S0, b: l.end - S0 }}));
  const W = (i, word, edge = "start") => Anim.wordTime(L[i], word, edge) - S0;      // istante locale di una parola
  const MOUTH = (t, idle = "rest") => Anim.mouthAt(LINES, S0 + t, idle);           // lip-sync del Contratto
  const $ = (s) => document.querySelector("#{sid} " + s);
  const tl = gsap.timeline({{ paused: true }});
"""
    open(f"{HF}/compositions/{sid}.html", "w").write(f"""<template>
<style>
  #{sid} {{ position:absolute; inset:0; overflow:hidden; }}
  #{sid} svg.stage {{ position:absolute; left:0; top:0; }}
{PAPER_CSS}
</style>
<div id="{sid}" data-composition-id="{sid}" data-width="1920" data-height="1080">
  <svg class="stage" width="1920" height="1080" viewBox="0 0 1920 1080" data-layout-ignore></svg>
{body.strip()}
</div>
<script>
(function () {{{prelude}
{script.strip()}
  window.__timelines[SID] = tl;
}})();
</script>
</template>
""")

hosts = []
for s in tl["scenes"]:
    f = f"{C}/shots/{s['id']}.html"
    if os.path.exists(f):
        scene(s["id"], s["start"], s["end"] - s["start"], f); hosts.append((s["id"], s["start"], s["end"] - s["start"], "graphics"))

# ---------------------------------------------------------------- sottotitoli (3-6 parole, uno alla volta)
groups = []
for l in tl["lines"]:
    words = l["caption"].split(); chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= 6 or (len(cur) >= 3 and re.search(r"[.,:;!?…]$", w)): chunks.append(cur); cur = []
    if cur:
        if chunks and len(cur) < 2: chunks[-1] += cur
        else: chunks.append(cur)
    segs = l.get("pauses") or [[0, l["speech"]]]; total = sum(b - a for a, b in segs); text = l["caption"]
    def at(ci):
        target = ci / len(text) * total; acc = 0
        for a, b in segs:
            if acc + (b - a) >= target: return l["start"] + a + (target - acc)
            acc += b - a
        return l["end"]
    pos = 0; wi = 0; Wd = l.get("words") if l.get("words") and len(l["words"]) == len(words) else None
    for ch in chunks:
        i0 = text.find(ch[0], pos); i1 = text.find(ch[-1], i0) + len(ch[-1]); pos = i1
        groups.append((" ".join(ch), Wd[wi]["start"] if Wd else at(i0), Wd[wi + len(ch) - 1]["end"] if Wd else at(i1))); wi += len(ch)
cap_body, cap_js = "", ""
for k, (txt, a, b) in enumerate(groups):
    nxt = groups[k + 1][1] if k + 1 < len(groups) else D; end = min(b + 0.35, nxt - 0.02)
    cap_body += f'  <div id="cg{k}" class="cg"><span>{html.escape(txt)}</span></div>\n'
    cap_js += (f'  tl.fromTo("#cg{k}", {{opacity:0}}, {{opacity:1, duration:0.1}}, {max(0.1, a - 0.05):.3f});\n'
               f'  tl.to("#cg{k}", {{opacity:0, duration:0.1, ease:"power2.in"}}, {end - 0.1:.3f});\n'
               f'  tl.set("#cg{k}", {{opacity:0, visibility:"hidden"}}, {end:.3f});\n')
open(f"{HF}/compositions/captions.html", "w").write(f"""<template>
<style>
  #captions {{ position:absolute; inset:0; overflow:visible; }}
  .cg {{ position:absolute; left:0; right:0; bottom:84px; text-align:center; }}
  .cg span {{ display:inline-block; font-family:'Inter'; font-weight:600; font-size:40px; color:#fbf6ec; padding:10px 24px; border-radius:12px; background:rgba(16,24,40,.66); }}
</style>
<div id="captions" data-composition-id="captions" data-width="1920" data-height="1080">
{cap_body}</div>
<script>
  const tl = gsap.timeline({{ paused: true }});
{cap_js}  window.__timelines["captions"] = tl;
</script>
</template>
""")
hosts.append(("captions", 0, D, "captions"))

# ---------------------------------------------------------------- audio e index.html
for a in ("voice", "music", "sfx"):
    src = f"{ROOT}/audio/cartoon_{a}.wav"
    if os.path.exists(src): shutil.copy(src, f"{HF}/assets/audio/{a}.wav")
media, z = "", 1
for aid, vol in (("voice", 1), ("music", 0.5), ("sfx", 0.85)):
    if os.path.exists(f"{HF}/assets/audio/{aid}.wav"):
        media += f'      <audio id="a-{aid}" class="clip" src="assets/audio/{aid}.wav" data-start="0" data-duration="{D:.3f}" data-track-index="{10 + z}" data-volume="{vol}"></audio>\n'; z += 1
hs = "".join(f'      <div id="{cid}" data-composition-id="{cid}" data-composition-src="compositions/{cid}.html" data-start="{a:.3f}" data-duration="{d:.3f}" '
             f'data-track-index="{20 + i}" data-track-kind="{k}" data-width="1920" data-height="1080" style="z-index:{10 + i}"></div>\n' for i, (cid, a, d, k) in enumerate(hosts))
open(f"{HF}/index.html", "w").write(f"""<!doctype html>
<html lang="it">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <title>Sono un contratto · Usare l'IA in sicurezza · Lezione 3</title>
    <script src="assets/gsap.min.js"></script>
    <script src="assets/characters.js"></script>
    <script src="assets/chatbots.js"></script>
    <script src="assets/props.js"></script>
    <script src="assets/scenes.js"></script>
    <script src="assets/anim.js"></script>
    <style>
      * {{ margin:0; padding:0; box-sizing:border-box; }}
      html, body {{ width:1920px; height:1080px; overflow:hidden; background:#141f2e; }}
      #root {{ position:relative; width:1920px; height:1080px; overflow:hidden; background:#141f2e; }}
    </style>
  </head>
  <body>
    <div id="root" data-composition-id="main" data-start="0" data-duration="{D:.3f}" data-width="1920" data-height="1080" data-fps="30">
{media}{hs}    </div>
    <script>
      window.__timelines["main"] = gsap.timeline({{ paused: true }});
    </script>
  </body>
</html>
""")
json.dump({"id": "sono-un-contratto-cartone", "name": "Sono un contratto · cartone"}, open(f"{HF}/meta.json", "w"))
json.dump({"paths": {"blocks": "compositions", "assets": "assets"}, "media": {"autoProxy": True}}, open(f"{HF}/hyperframes.json", "w"), indent=2)
print(f"cartone: {len(hosts) - 1} scene, {len(groups)} gruppi di sottotitoli, durata {D:.2f}s")
