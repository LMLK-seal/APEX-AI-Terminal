"""Musica ed effetti del cartone, sintetizzati da codice e sincronizzati a timeline.json (stessi tempi delle coreografie).
Musica: casa = marimba + pizzicato + shaker in Do maggiore (112 BPM); fuori = La minore, pulsazione bassa, campane fredde,
arpeggio veloce nel viaggio; riavvolgimento = "tape stop"; finale = maggiore piu' luminoso + ta-da sul cartello.
Effetti: ~40 suoni sulle azioni (corsa, pop, risucchio, colpi, strappi, timbri, errori, sparkle...).
Uscite: audio/cartoon_music.wav, audio/cartoon_sfx.wav, audio/cartoon_voice.wav (= voce guida provvisoria).
La versione definitiva (ElevenLabs: voce, ed eventualmente musica ed effetti) si monta nella prossima sessione."""
import json, os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # percorsi relativi alla cartella del progetto
import numpy as np, soundfile as sf

SR = 48000
tl = json.load(open("timeline.json")); D = tl["duration"]; N = int(D * SR) + SR
S = {s["id"]: s for s in tl["scenes"]}; LN = lambda sid: [l for l in tl["lines"] if l["scene"] == sid]
rng = np.random.default_rng(5)

def wt(line, word, edge="start"):   # stessa logica di Anim.wordTime (JS)
    w = next((x for x in line.get("words") or [] if word.lower() in x["w"].lower()), None)   # tempi faster-whisper se ci sono
    if w: return w["end" if edge == "end" else "start"]
    text = line["caption"]; i = text.lower().find(word.lower())
    if i < 0: return line["start"]
    pos = (i + (len(word) if edge == "end" else 0)) / len(text)
    segs = line.get("pauses") or [[0, line["speech"]]]; total = sum(b - a for a, b in segs); target = pos * total; acc = 0
    for a, b in segs:
        if acc + (b - a) >= target: return line["start"] + a + (target - acc)
        acc += b - a
    return line["end"]

T = lambda sid, dt=0.0: S[sid]["start"] + dt
env = lambda n, a=0.004, r=0.25: (np.minimum(1, np.arange(n) / SR / max(a, 1e-4)) * np.exp(-np.arange(n) / SR / r)).astype(np.float32)
midi = lambda m: 440 * 2 ** ((m - 69) / 12)
def put(buf, t, y, g=1.0):
    i = int(t * SR); j = min(len(buf), i + len(y))
    if j > max(0, i): buf[max(0, i):j] += (y[max(0, -i):j - i] * g).astype(np.float32)
def tone(f, dur, parts=((1, 1),), a=0.004, r=0.3, vib=0.0):
    n = int(dur * SR); t = np.arange(n) / SR; ph = 2 * np.pi * f * t + (vib * np.sin(2 * np.pi * 5.5 * t) if vib else 0)
    return (sum(w * np.sin(ph * k) for k, w in parts) * env(n, a, r)).astype(np.float32)
marimba = lambda m, d=0.5, v=0.12: tone(midi(m), d, ((1, 1), (4, 0.35), (10, 0.08)), 0.002, 0.22) * v
pizz = lambda m, d=0.4, v=0.16: tone(midi(m), d, ((1, 1), (2, 0.5), (3, 0.2)), 0.003, 0.16) * v
bell = lambda m, d=1.4, v=0.05: tone(midi(m), d, ((1, 1), (2.76, 0.45), (5.4, 0.2)), 0.002, 0.7) * v
def pad(ms, dur, v=0.04):
    n = int(dur * SR); t = np.arange(n) / SR; y = np.zeros(n, np.float32)
    for m in ms:
        for d in (-0.004, 0, 0.005): y += np.sin(2 * np.pi * midi(m) * (1 + d) * t).astype(np.float32)
    return y * np.minimum(1, t / 0.8) * np.minimum(1, (dur - t) / 0.8) * v / (3 * len(ms))
def noise(n): return rng.standard_normal(n).astype(np.float32)
def bp(x, lo, hi):
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / SR); X[(f < lo) | (f > hi)] = 0; return np.fft.irfft(X, len(x)).astype(np.float32)
shaker = lambda v=0.035: bp(noise(int(0.06 * SR)), 5000, 12000) * env(int(0.06 * SR), 0.002, 0.02) * v

music = np.zeros(N, np.float32); BPM = 112; b = 60 / BPM
def home(t0, t1, root=60, v=1.0, bright=False):
    prog = [(0, [0, 4, 7, 12]), (9, [9, 12, 16, 21]), (5, [5, 9, 12, 17]), (7, [7, 11, 14, 19])]
    mel = [12, None, 16, 14, None, 12, 9, None, 7, None, 9, 12, None, 14, 16, None]
    k = 0; t = t0
    while t < t1 - 0.05:
        bass, ch = prog[(k // 16) % 4]
        if k % 8 == 0: put(music, t, pizz(root - 24 + bass, 0.6, 0.2 * v))
        if k % 8 == 4: put(music, t, pizz(root - 17 + bass, 0.4, 0.14 * v))
        if k % 2 == 1: put(music, t, shaker(0.03 * v))
        if k % 4 == 2: put(music, t, marimba(root + ch[(k // 4) % 3], 0.4, 0.06 * v))
        m = mel[k % 16]
        if m is not None and (bright or (k // 16) % 2 == 0): put(music, t, marimba(root + m, 0.5, 0.09 * v))
        t += b / 2; k += 1
def outside(t0, t1, v=1.0, arp=False):
    put(music, t0, pad([45, 52, 57, 60], t1 - t0, 0.05 * v))
    t = t0; k = 0
    while t < t1:
        put(music, t, pizz(33, 0.7, 0.2 * v));
        if arp:
            for i, m in enumerate([57, 60, 64, 69, 64, 60, 57, 52]): put(music, t + i * b / 4, marimba(m, 0.2, 0.05 * v))
        elif k % 2 == 1: put(music, t + b, bell([69, 72, 76, 74][(k // 2) % 4], 1.6, 0.035 * v))
        t += b * 2; k += 1

home(0.0, T("viaggio", 0.9))
outside(T("viaggio", 0.9), T("viaggio", 3.95), arp=True)
outside(T("viaggio", 3.95), T("riavvolgi", 0.05), v=0.9)
home(T("giusta", 0.1), T("cartello"), v=1.05, bright=True)
for i, m in enumerate((72, 76, 79, 84)): put(music, T("cartello", -0.25 + i * 0.07), bell(m, 1.8, 0.07))
put(music, T("cartello"), pad([48, 60, 64, 67, 72], 3.4, 0.07))
# riavvolgimento: tape stop della musica di fuori (pitch che scende)
n = int(0.6 * SR); tt = np.arange(n) / SR; f = 220 * (1 - tt / 0.6) ** 2 + 30
put(music, T("riavvolgi", 0.0), (np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.01, 0.4) * 0.08).astype(np.float32))
# ducking sotto la voce
duck = np.ones(N, np.float32)
for l in tl["lines"]: duck[int((l["start"] - 0.12) * SR):int((l["end"] + 0.2) * SR)] = 0.5
k = int(0.1 * SR); music *= np.convolve(duck, np.ones(k) / k, mode="same").astype(np.float32)

# ---------------------------------------------------------------- effetti
fx = np.zeros(N, np.float32)
def whoosh(d, up=True, v=0.3):
    n = int(d * SR); x = noise(n); out = np.zeros(n, np.float32); seg = n // 20
    for i in range(20): f0 = 400 + (i / 20 if up else 1 - i / 20) * 4500; out[i * seg:(i + 1) * seg] = bp(x[i * seg:(i + 1) * seg], f0 * .6, f0 * 1.4)
    return out * np.sin(np.pi * np.arange(n) / n) ** 1.5 * v
def pop(v=0.35):
    n = int(0.09 * SR); t = np.arange(n) / SR; return (np.sin(2 * np.pi * (300 + 900 * t / 0.09) * t) * env(n, 0.001, 0.03) * v).astype(np.float32)
def thud(v=0.5, f=85):
    n = int(0.3 * SR); t = np.arange(n) / SR; return (np.sin(2 * np.pi * (f - 35 * t / 0.3) * t) * env(n, 0.001, 0.08) + bp(noise(n), 150, 2200) * env(n, 0.0005, 0.02) * .5) * v
def tick(v=0.12): n = int(0.025 * SR); return bp(noise(n), 2500, 8000) * env(n, 0.0005, 0.005) * v
def rustle(d=0.3, v=0.18):
    n = int(d * SR); g = np.repeat(rng.uniform(0.2, 1, n // 300 + 1), 300)[:n]; return bp(noise(n), 1800, 9000) * g * env(n, 0.01, d / 2.5) * v
def stack(parts, dur):        # somma di suoni con ritardi diversi, su una durata fissa
    out = np.zeros(int(dur * SR), np.float32)
    for off, y in parts: i = int(off * SR); j = min(len(out), i + len(y)); out[i:j] += y[: j - i]
    return out
def sparkle(v=0.06): return stack([(i * 0.06, bell(m, 0.8, v)) for i, m in enumerate((84, 88, 91, 96))], 1.1)
def boing(v=0.15):
    n = int(0.45 * SR); t = np.arange(n) / SR; f = 180 + 140 * np.sin(2 * np.pi * 9 * t) * np.exp(-t * 5); return (np.sin(2 * np.pi * np.cumsum(f) / SR) * env(n, 0.003, 0.2) * v).astype(np.float32)
def buzz(v=0.12): n = int(0.18 * SR); t = np.arange(n) / SR; return (np.sign(np.sin(2 * np.pi * 110 * t)) * env(n, 0.002, 0.08) * v * 0.6).astype(np.float32)
def rip(v=0.3): n = int(0.35 * SR); return bp(noise(n), 900, 7000) * np.repeat(rng.uniform(0.3, 1, n // 120 + 1), 120)[:n] * env(n, 0.003, 0.12) * v
def slide_whistle(d, down=True, v=0.08):
    n = int(d * SR); t = np.arange(n) / SR; f = (1400 - 1000 * t / d) if down else (400 + 1000 * t / d); return (np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, t / 0.05) * v).astype(np.float32)
def sting(v=0.2): return pizz(36, 0.6, v) + pizz(43, 0.6, v * 0.6)
def chime(v=0.07): return stack([(i * 0.09, bell(m, 1.0, v)) for i, m in enumerate((72, 76, 79))], 1.4)

# 1 buongiorno
L = LN("buongiorno")[0]; t0 = wt(L, "Buongiorno"); tw = t0 - 0.85
for k in range(int(tw) + 1): put(fx, k + 0.1, tick(0.14))
put(fx, T("buongiorno") + tw, boing(0.14)); put(fx, t0 + 0.05, whoosh(0.3, True, 0.08)); put(fx, wt(L, "Tre") + 0.1, rustle(0.35, 0.2)); put(fx, wt(L, "due"), pop(0.12))
# 2 fretta
s0 = T("fretta"); Lf = LN("fretta"); tQ, tF = wt(Lf[0], "Qualcuno"), wt(Lf[0], "fretta"); tC, tI, tCh, tG = wt(Lf[1], "copia"), wt(Lf[1], "incolla"), wt(Lf[1], "chatbot"), Lf[2]["start"]
for k in range(3): put(fx, s0 + 0.1 + k, tick(0.14))
for k in range(int((tF + 0.28 - tQ) / 0.09)): put(fx, tQ + k * 0.09, tick(0.08))
put(fx, tF + 0.15, bp(noise(int(0.3 * SR)), 2500, 6000) * env(int(0.3 * SR), 0.01, 0.12) * 0.12)
put(fx, tC, pop(0.4)); put(fx, tI - 0.3, whoosh(0.55, False, 0.25)); put(fx, tI + 0.25, pop(0.3)); put(fx, tCh - 0.05, sparkle(0.07)); put(fx, tG - 0.1, sting(0.2))
# 3 nessuno
Ln = LN("nessuno")[0]; tV, tA, tD = wt(Ln, "vietato"), wt(Ln, "autorizzato"), wt(Ln, "Diciamo")
put(fx, tV + 0.25, thud(0.25, 110)); put(fx, tA + 0.25, thud(0.25, 120)); put(fx, tD - 0.35, slide_whistle(0.7, True, 0.07))
# 4 viaggio
v0 = T("viaggio"); Lv = LN("viaggio"); tPecc, tNon = wt(Lv[1], "Peccato"), wt(Lv[1], "Non so")
put(fx, v0 + 0.2, whoosh(0.75, True, 0.35)); put(fx, v0 + 0.9, whoosh(3.0, True, 0.18))
for k in range(12): put(fx, v0 + 1.0 + k * 0.24, pop(0.05))
put(fx, v0 + 3.95, slide_whistle(0.3, True, 0.08)); put(fx, v0 + 4.25, thud(0.5))
for i in range(12): put(fx, tPecc + ((i * 5) % 12) * 0.12, tone(midi(40 + i % 3), 0.25, ((1, 1),), 0.005, 0.1) * 0.05)
put(fx, tNon - 0.3, rustle(0.3, 0.15))
put(fx, v0 + 3.95, (np.sin(2 * np.pi * 60 * np.arange(int((T("riavvolgi") - v0 - 3.95) * SR)) / SR) * 0.02).astype(np.float32))
# 5 nome
tS = wt(LN("nome")[0], "Shadow")
for i in range(8): put(fx, tS + i * 0.075 + 0.2, thud(0.28, 95 + i * 4))
# 6 rischio
Lr = LN("rischio"); g = lambda w, e="start": wt(Lr[0], w, e)
put(fx, g("doppio", "end"), rip(0.35))
for i in range(6): put(fx, g("dati") + i * 0.09, rustle(0.12, 0.12))
put(fx, g("informazioni"), pop(0.2)); put(fx, g("informazioni") + 0.15, tick(0.3)); put(fx, g("controllo") - 0.2, rustle(0.9, 0.15))
put(fx, g("errori"), bp(noise(int(0.8 * SR)), 2000, 6000) * 0.05)
for i in range(3): put(fx, g("firmi") + i * 0.18, buzz(0.1))
put(fx, g("violate"), rip(0.3)); put(fx, g("violate") + 0.1, thud(0.35)); put(fx, Lr[1]["start"] + 0.45, thud(0.5))
# 7 riavvolgi
r0 = T("riavvolgi"); tR = LN("riavvolgi")[0]["start"] - r0
put(fx, r0 + 0.05, whoosh(tR + 0.9, False, 0.3))
for k in range(16): put(fx, r0 + 0.1 + k * 0.07, tick(0.09))
put(fx, r0 + tR + 0.95, pop(0.15))
# 8 giusta
g0 = T("giusta"); Lg = LN("giusta")[0]; gg = lambda w, e="start": wt(Lg, w, e)
put(fx, g0 + 0.15, slide_whistle(0.55, False, 0.06)); put(fx, g0 + 0.62, pop(0.18)); put(fx, g0 + 0.7, sparkle(0.05))   # la bolla si gonfia
put(fx, gg("tre") - 0.1, rustle(0.4, 0.15)); put(fx, gg("strumento") - 0.1, chime(0.06))
for w in ("approvato", "chiare", "formate"): put(fx, gg(w, "end") - 0.08, thud(0.5))
for k in range(8): put(fx, gg("Persone") - 0.7 + k * 0.18, tick(0.06))
# 9 finale
Lf9 = LN("finale")[0]; put(fx, wt(Lf9, "lavoro") - 0.1, whoosh(0.9, False, 0.12)); put(fx, wt(Lf9, "trenta"), chime(0.08))
put(fx, wt(Lf9, "E io"), sparkle(0.08)); put(fx, wt(Lf9, "E io") + 0.1, boing(0.1))
# 10 cartello
put(fx, T("cartello", 0.6), pop(0.15))

mx = max(np.abs(music).max(), 1e-6); music = music / mx * 0.5
sf.write("audio/cartoon_music.wav", music[: int(D * SR)], SR); sf.write("audio/cartoon_sfx.wav", np.clip(fx, -1, 1)[: int(D * SR)], SR)
import os
VOICE = "audio/voice_eleven.wav" if os.path.exists("audio/voice_eleven.wav") else "audio/voice_guide.wav"   # ElevenLabs se c'e'
# ElevenLabs misura -18.7 LUFS (la voce guida -17.0) con punte alte: +1.2 dB e limiter a -3 dBFS -> mix finale ~-15 LUFS come la v2
VOICE_GAIN_DB = 1.2 if "eleven" in VOICE else 0.0
import subprocess
subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-v", "error", "-y", "-i", VOICE, "-af",
                f"volume={VOICE_GAIN_DB}dB,alimiter=limit=0.7:level=false", "audio/cartoon_voice.wav"], check=True)
print(f"musica, effetti e voce ({VOICE}) pronti: {D:.2f} s")
