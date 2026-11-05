"""Prossima sessione: dalla ripresa unica di ElevenLabs alle 13 battute del copione, poi timeline e video riallineati.
1) taglio della ripresa: segmenti di voce (Silero VAD di faster-whisper) raggruppati in 13 battute con una
   programmazione dinamica che rispetta la lunghezza attesa di ogni battuta (testo) e preferisce i tagli sulle pause lunghe
2) salva audio/eleven/00.wav ... 12.wav  -> poi:
   .venv/bin/python tools/timeline.py audio/script_cartoon.json audio/eleven audio/pads_cartoon.json audio/voice_eleven.wav
   .venv/bin/python tools/fw_words.py          (facoltativo, se huggingface.co e' abilitato: tempi parola per parola)
   python3 tools/make_audio_cartoon.py && python3 cartoon/build.py && (cd cartoon/hf && npx hyperframes render ...)
Uso: .venv/bin/python tools/voice_to_timeline.py <file ElevenLabs .mp3/.wav>
"""
import json, subprocess, sys, os, re, shutil
import numpy as np
from faster_whisper.audio import decode_audio
from faster_whisper.vad import get_speech_timestamps, VadOptions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.environ.get("FFMPEG") or next((p for p in ("/opt/ffmpeg-src/bin/ffmpeg", shutil.which("ffmpeg")) if p and os.path.exists(p)), "ffmpeg"); FF = FFMPEG
src = sys.argv[1]; script = json.load(open(f"{ROOT}/audio/script_cartoon.json")); n = len(script)
a = decode_audio(src, sampling_rate=16000)
seg = [(t["start"] / 16000, t["end"] / 16000) for t in get_speech_timestamps(a, VadOptions(min_silence_duration_ms=220, speech_pad_ms=60))]
m = len(seg)
if m < n: sys.exit(f"Trovati solo {m} segmenti di voce per {n} battute: rigenera con pause piu' lunghe tra i paragrafi.")
w = np.array([len(re.sub(r"\s+", "", s["caption"])) for s in script], float); w /= w.sum()
total = sum(e - s for s, e in seg); expect = w * total
gap = [seg[i + 1][0] - seg[i][1] for i in range(m - 1)]
INF = 1e18; cost = np.full((n + 1, m + 1), INF); back = np.zeros((n + 1, m + 1), int); cost[0][0] = 0
for i in range(1, n + 1):
    for j in range(i, m + 1):
        for k in range(i - 1, j):                         # la battuta i-1 copre i segmenti k..j-1
            if cost[i - 1][k] >= INF: continue
            sp = seg[j - 1][1] - seg[k][0]
            c = (sp - expect[i - 1]) ** 2 / max(expect[i - 1], 0.3)
            if j < m: c -= 1.5 * min(gap[j - 1], 1.2)    # tagliare su una pausa lunga conviene
            if cost[i - 1][k] + c < cost[i][j]: cost[i][j] = cost[i - 1][k] + c; back[i][j] = k
groups, j = [], m
for i in range(n, 0, -1): k = back[i][j]; groups.append((k, j)); j = k
groups.reverse()
os.makedirs(f"{ROOT}/audio/eleven", exist_ok=True)
for i, (k, j) in enumerate(groups):
    s0, s1 = max(0, seg[k][0] - 0.05), seg[j - 1][1] + 0.08
    subprocess.run([FF, "-v", "error", "-y", "-i", src, "-ss", f"{s0:.3f}", "-to", f"{s1:.3f}", "-ac", "1", "-ar", "48000", f"{ROOT}/audio/eleven/{i:02d}.wav"], check=True)
    print(f"{i:02d}  {s0:6.2f}-{s1:6.2f}  ({s1 - s0:4.2f}s, attesa {expect[i]:4.2f}s)  {script[i]['caption']}")
print("Ascolta i tagli (audio/eleven/*.wav); se una battuta e' tagliata male, correggi i tempi a mano e rilancia timeline.py.")
