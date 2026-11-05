"""Timeline del video: posiziona le battute della voce sulle scene.
- faster-whisper (Silero VAD incluso nel pacchetto) misura inizio/fine reali del parlato di ogni battuta
- regole di ritmo per scena (respiro prima/dopo, pausa tra battute)
- esporta timeline.json e la traccia voce montata (FFmpeg compilato dal repo)
Uso: .venv/bin/python tools/timeline.py audio/script_cartoon.json audio/eleven audio/pads_cartoon.json audio/voice_eleven.wav
"""
import json, sys, subprocess, os, shutil
from faster_whisper.audio import decode_audio
from faster_whisper.vad import get_speech_timestamps, VadOptions

FFMPEG = os.environ.get("FFMPEG") or next((p for p in ("/opt/ffmpeg-src/bin/ffmpeg", shutil.which("ffmpeg")) if p and os.path.exists(p)), "ffmpeg")
script = json.load(open(sys.argv[1])); d = sys.argv[2]
PADS = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None      # pause per scena (opzionale)
VOICE_OUT = sys.argv[4] if len(sys.argv) > 4 else "audio/voice_eleven.wav"

# respiro (s) prima della prima battuta e dopo l'ultima, per scena: lascia spazio alle azioni visive
PAD = {  # scena: (prima, dopo)
    "buongiorno": (1.4, 0.5), "fretta": (0.5, 0.6), "nessuno": (0.4, 0.6), "viaggio": (1.6, 0.6),
    "nome": (0.3, 1.2), "rischio": (0.6, 0.6), "riavvolgi": (0.5, 1.0), "giusta": (0.6, 0.6),
    "finale": (0.5, 0.9), "cartello": (0.0, 3.2),
}
GAP = 0.35  # pausa tra battute nella stessa scena
if PADS: PAD = {k: tuple(v) for k, v in PADS.items()}

lines = []
for i, l in enumerate(script):
    wav = f"{d}/{i:02d}.wav"
    a = decode_audio(wav, sampling_rate=16000)
    ts = get_speech_timestamps(a, VadOptions(min_silence_duration_ms=150, speech_pad_ms=40))
    s0 = ts[0]["start"] / 16000 if ts else 0.0
    s1 = ts[-1]["end"] / 16000 if ts else len(a) / 16000
    lines.append({**l, "i": i, "wav": wav, "trim_start": round(s0, 3), "speech": round(s1 - s0, 3),
                  "pauses": [[round(t["start"] / 16000 - s0, 3), round(t["end"] / 16000 - s0, 3)] for t in ts]})

scenes, t = [], 0.0
order = []
for l in lines:
    if not order or order[-1] != l["scene"]: order.append(l["scene"])
order.append("cartello")
for sc in order:
    pre, post = PAD[sc]; start = t; t += pre
    for l in [x for x in lines if x["scene"] == sc]:
        t += l.get("pre", 0.0)                 # pausa extra prima della battuta (gag visiva)
        l["start"] = round(t, 3); l["end"] = round(t + l["speech"], 3); t = l["end"] + GAP
    if any(x["scene"] == sc for x in lines): t -= GAP
    t += post
    scenes.append({"id": sc, "start": round(start, 3), "end": round(t, 3)})

total = round(t, 3)
json.dump({"duration": total, "scenes": scenes, "lines": lines}, open("timeline.json", "w"), ensure_ascii=False, indent=1)

# traccia voce: ogni battuta tagliata sul parlato reale e messa al suo tempo
inputs, filt = [], []
for k, l in enumerate(lines):
    inputs += ["-i", l["wav"]]
    filt.append(f"[{k}]atrim=start={l['trim_start']},asetpts=PTS-STARTPTS,aresample=48000,adelay={int(l['start']*1000)}:all=1[v{k}]")
filt.append("".join(f"[v{k}]" for k in range(len(lines))) + f"amix=inputs={len(lines)}:normalize=0,apad=whole_dur={total}[out]")
subprocess.run([FFMPEG, "-v", "error", "-y", *inputs, "-filter_complex", ";".join(filt), "-map", "[out]", "-ac", "1", VOICE_OUT], check=True)
for s in scenes: print(f"{s['id']:<11} {s['start']:6.2f} → {s['end']:6.2f}")
print("durata totale:", total, "s")
