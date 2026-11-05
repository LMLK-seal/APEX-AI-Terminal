"""Voce del Contratto con l'API di ElevenLabs: elenco voci, provini, le 13 battute.
- testo e tag di recitazione da VOCE-ELEVENLABS.md (stesso ordine di audio/script_cartoon.json)
- eleven_v3 se l'account ce l'ha (tag tra parentesi quadre), altrimenti eleven_multilingual_v2 senza tag
- ogni battuta -> audio/eleven/NN.wav (mono, 48 kHz, FFmpeg) + audio/eleven/meta.json (voce, modello, impostazioni)
La chiave si legge da video-shadow-ai/.env (ELEVENLABS_API_KEY) e non viene mai stampata ne' salvata altrove.
Uso (dalla cartella video-shadow-ai):
  python tools/eleven_tts.py voci                          # piano, modelli, voci dell'account e italiane della libreria
  python tools/eleven_tts.py provini ID[=nome] [ID ...]    # battute 4 e 8 -> audio/provini/
  python tools/eleven_tts.py design "descrizione" [nome]   # Voice Design: 3 voci nuove -> audio/provini/nome-1..3.wav
  python tools/eleven_tts.py salva GENERATED_ID nome       # salva una voce di Voice Design nell'account -> voice_id
  python tools/eleven_tts.py battute ID[=nome] [n ...]     # tutte, o solo le n indicate (0-12): raw/NN.wav + NN.wav
  python tools/eleven_tts.py tempo 1.15                    # riaccelera NN.wav da raw/NN.wav (atempo), senza rigenerare
Opzioni: --modello eleven_multilingual_v2 (forza il modello), --sciadou (battuta 7: "Shadow AI" -> "Sciàdou èi-ài"),
         --tempo 1.15 (velocita'; se manca si usa quella gia' in meta.json)
"""
import base64, json, os, re, subprocess, sys, time
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from urllib.parse import urlencode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://api.elevenlabs.io"
FFMPEG = os.environ.get("FFMPEG") or "ffmpeg"


def load_key():
    raw = open(f"{ROOT}/.env", "rb").read()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    rows = [r.strip() for r in txt.splitlines() if r.strip() and not r.strip().startswith("#")]
    for r in rows:
        k, eq, v = r.removeprefix("export ").partition("=")
        if eq and k.strip() in ("ELEVENLABS_API_KEY", "ELEVEN_API_KEY", "XI_API_KEY"): return v.strip().strip("\"'")
    if len(rows) == 1 and "=" not in rows[0]: return rows[0].strip("\"'")   # solo la chiave, senza nome
    sys.exit("ELEVENLABS_API_KEY non trovata in .env (serve una riga ELEVENLABS_API_KEY=...)")


KEY = load_key()


def api(path, body=None, accept="application/json", **q):
    req = Request(API + path + ("?" + urlencode(q) if q else ""), data=None if body is None else json.dumps(body).encode(),
                  headers={"xi-api-key": KEY, "Content-Type": "application/json", "Accept": accept})
    try:
        with urlopen(req, timeout=180) as r: data = r.read()
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} su {path}: {e.read().decode(errors='replace')[:500]}") from None
    return json.loads(data) if accept == "application/json" else data


def lines_from_doc():
    doc = open(f"{ROOT}/VOCE-ELEVENLABS.md", encoding="utf-8").read()
    block = re.search(r"```\n(.*?)```", doc, re.S).group(1)
    out = [p.strip() for p in block.split("\n\n") if p.strip()]
    assert len(out) == len(json.load(open(f"{ROOT}/audio/script_cartoon.json", encoding="utf-8"))), "battute diverse dal copione"
    return out


untag = lambda t: re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]", " ", t)).strip()


def tts_models():
    return [m["model_id"] for m in api("/v1/models") if m.get("can_do_text_to_speech")]


def settings(model):   # in eleven_v3 la stabilita' accetta solo 0, 0.5, 1 (0.5 = "Natural")
    return {"stability": 0.5 if model == "eleven_v3" else 0.45, "similarity_boost": 0.75, "style": 0.3, "use_speaker_boost": True}


def tts(voice, text, model, prev=None, nxt=None):
    """Restituisce (mp3, formato, contesto_usato). Prova con previous/next_text, senza se il modello li rifiuta."""
    base = {"text": text, "model_id": model, "voice_settings": settings(model)}
    ctx = (prev or nxt) and model != "eleven_v3"   # eleven_v3 rifiuta previous/next_text (unsupported_model)
    tries = ([{**base, "previous_text": prev or "", "next_text": nxt or ""}] if ctx else []) + [base]
    err = None
    for body in tries:
        for fmt in list(FORMATS):
            try: return api(f"/v1/text-to-speech/{voice}", body, accept="audio/mpeg", output_format=fmt), fmt, "previous_text" in body
            except RuntimeError as e:
                err = e
                if "Output format" in str(e): FORMATS.remove(fmt)   # 192 kbps solo dal piano Creator in su
                else: print("  ", str(e)[:200])
    raise err


FORMATS = ["mp3_44100_192", "mp3_44100_128"]


def to_wav(mp3, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", "pipe:0", "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", path], input=mp3, check=True)


def apply_tempo(i, t):
    """audio/eleven/raw/NN.wav (come esce da ElevenLabs) -> audio/eleven/NN.wav accelerata di t (atempo: stessa intonazione)."""
    raw, out = f"{ROOT}/audio/eleven/raw/{i:02d}.wav", f"{ROOT}/audio/eleven/{i:02d}.wav"
    subprocess.run([FFMPEG, "-v", "error", "-y", "-i", raw, *(["-af", f"atempo={t}"] if t != 1 else []), "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", out], check=True)


def pick_model(args):
    if "--modello" in args: return args[args.index("--modello") + 1]
    return "eleven_v3" if "eleven_v3" in tts_models() else "eleven_multilingual_v2"


def parse_voices(args):
    skip = {"--modello", "--sciadou", "--tempo"}; out = []
    for i, a in enumerate(args):
        if a in skip or (i and args[i - 1] in ("--modello", "--tempo")) or a.isdigit(): continue
        vid, _, name = a.partition("="); out.append((vid, name or vid))
    return out


def texts_for(model, sciadou=False):
    raw = lines_from_doc()
    if sciadou: raw[7] = raw[7].replace("Shadow AI", "Sciàdou èi-ài")
    return raw if model == "eleven_v3" else [untag(t) for t in raw]


def cmd_voci():
    sub = api("/v1/user/subscription")
    print(f"piano: {sub.get('tier')}  caratteri usati {sub.get('character_count')}/{sub.get('character_limit')}")
    ms = tts_models(); print("modelli TTS:", ", ".join(ms)); print("eleven_v3 disponibile:", "eleven_v3" in ms)
    print("\n== voci dell'account")
    for v in api("/v1/voices")["voices"]:
        lb = v.get("labels") or {}; langs = ",".join(sorted({x.get("language", "") for x in v.get("verified_languages") or []}))
        print(f"  {v['voice_id']}  {v['name'][:38]:<38} {v.get('category',''):<12} {lb.get('gender','')}/{lb.get('age','')} {lb.get('accent','')} [{langs}]")
    print("\n== libreria condivisa, italiano")
    seen = set()
    for q in ({}, {"use_cases": "characters_animation"}, {"use_cases": "conversational"}, {"use_cases": "narrative_story"}):
        for v in api("/v1/shared-voices", language="it", page_size=40, sort="cloned_by_count", **q)["voices"]:
            if v["voice_id"] in seen: continue
            seen.add(v["voice_id"])
            print(f"  {v['voice_id']}  {v['name'][:38]:<38} {v.get('gender','')}/{v.get('age','')} {v.get('accent','')} {v.get('locale') or ''} "
                  f"{v.get('use_case','')}/{v.get('descriptive','')} cloni={v.get('cloned_by_count')} owner={v.get('public_owner_id')}\n"
                  f"      {(v.get('description') or '').strip()[:140]}")


def cmd_provini(args):
    model = pick_model(args); T = texts_for(model, "--sciadou" in args)
    for vid, name in parse_voices(args):
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
        for i in (3, 7):
            mp3, fmt, _ = tts(vid, T[i], model)
            path = f"{ROOT}/audio/provini/{slug}_battuta{i + 1:02d}.wav"; to_wav(mp3, path)
            print(f"{name}: {os.path.relpath(path, ROOT)}  ({model}, {fmt})")


def cmd_design(args):
    """Voice Design: 3 voci nuove da una descrizione; l'anteprima recita le battute 4, 5 e 8 (provino gia' pronto)."""
    desc, tag = args[0], (args[1] if len(args) > 1 else "design")
    raw = lines_from_doc(); text = " ".join(raw[i] for i in (3, 4, 7))
    r = api("/v1/text-to-voice/design", {"voice_description": desc, "model_id": "eleven_ttv_v3", "text": text})
    dp = f"{ROOT}/audio/provini/design.json"
    log = json.load(open(dp, encoding="utf-8")) if os.path.exists(dp) else []
    for k, p in enumerate(r["previews"]):
        path = f"{ROOT}/audio/provini/{tag}-{k + 1}.wav"; to_wav(base64.b64decode(p["audio_base_64"]), path)
        log.append({"file": os.path.relpath(path, ROOT).replace("\\", "/"), "generated_voice_id": p["generated_voice_id"], "description": desc, "text": text})
        print(f"{os.path.relpath(path, ROOT)}  {p.get('duration_secs', 0):.1f}s  generated_voice_id={p['generated_voice_id']}")
    json.dump(log, open(dp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_salva(args):
    """Salva un'anteprima di Voice Design tra le voci dell'account e restituisce il voice_id da usare per le battute."""
    gid, name = args[0], args[1]
    d = next(x for x in json.load(open(f"{ROOT}/audio/provini/design.json", encoding="utf-8")) if x["generated_voice_id"] == gid)
    print("voice_id:", api("/v1/text-to-voice", {"voice_name": name, "voice_description": d["description"], "generated_voice_id": gid})["voice_id"])


def cmd_battute(args):
    model = pick_model(args); T = texts_for(model, "--sciadou" in args); plain = [untag(t) for t in T]
    (vid, name), = parse_voices(args)
    which = [int(a) for a in args if a.isdigit()] or list(range(len(T)))
    mp = f"{ROOT}/audio/eleven/meta.json"
    meta = json.load(open(mp, encoding="utf-8")) if os.path.exists(mp) else {}
    if meta.get("voice_id") not in (None, vid) or meta.get("model") not in (None, model): meta = {}   # voce o modello cambiati: si riparte
    tempo = float(args[args.index("--tempo") + 1]) if "--tempo" in args else meta.get("atempo", 1.0)
    meta.update(voice_id=vid, voice_name=name, model=model, voice_settings=settings(model), atempo=tempo, lines=meta.get("lines") or {})
    for i in which:
        mp3, fmt, ctx = tts(vid, T[i], model, plain[i - 1] if i else None, plain[i + 1] if i + 1 < len(T) else None)
        to_wav(mp3, f"{ROOT}/audio/eleven/raw/{i:02d}.wav"); apply_tempo(i, tempo)
        meta["lines"][f"{i:02d}"] = {"text": T[i], "output_format": fmt, "previous_next_text": ctx, "generated": time.strftime("%Y-%m-%d %H:%M")}
        print(f"audio/eleven/{i:02d}.wav  {fmt}  tempo={tempo}  contesto={'si' if ctx else 'no'}  {T[i][:70]}")
    meta["lines"] = dict(sorted(meta["lines"].items()))
    json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def cmd_tempo(args):
    """Riapplica la velocita' a tutte le battute gia' generate, senza richiamare ElevenLabs."""
    t = float(args[0]); mp = f"{ROOT}/audio/eleven/meta.json"; meta = json.load(open(mp, encoding="utf-8"))
    os.makedirs(f"{ROOT}/audio/eleven/raw", exist_ok=True)
    for k in meta["lines"]:
        raw, out = f"{ROOT}/audio/eleven/raw/{k}.wav", f"{ROOT}/audio/eleven/{k}.wav"
        if not os.path.exists(raw): os.replace(out, raw)   # battute generate prima che esistesse raw/
        apply_tempo(int(k), t)
    meta["atempo"] = t; json.dump(meta, open(mp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(meta['lines'])} battute a {t}x")


if __name__ == "__main__":
    a = sys.argv[1:]
    cmds = {"voci": lambda: cmd_voci(), "provini": lambda: cmd_provini(a[1:]), "design": lambda: cmd_design(a[1:]),
            "salva": lambda: cmd_salva(a[1:]), "battute": lambda: cmd_battute(a[1:]), "tempo": lambda: cmd_tempo(a[1:])}
    if not a or a[0] not in cmds: sys.exit(__doc__)
    cmds[a[0]]()
