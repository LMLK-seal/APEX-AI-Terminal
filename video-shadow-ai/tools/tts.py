"""Voce narrante italiana riga per riga con Piper (voce it_IT 'paola').
Chatterbox richiede pesi da HuggingFace, non raggiungibile da questo ambiente:
Piper e' il sostituto open-source eseguibile su CPU."""
import json, sys, os, wave
from piper import PiperVoice
from piper.config import SynthesisConfig
M="/opt/models/vits-piper-it_IT-paola-medium/it_IT-paola-medium.onnx"
v=PiperVoice.load(M)
lines=json.load(open(sys.argv[1])); out=sys.argv[2]; os.makedirs(out,exist_ok=True)
meta=[]
for i,l in enumerate(lines):
    p=f"{out}/{i:02d}.wav"
    with wave.open(p,"wb") as w:
        v.synthesize_wav(l["text"],w,syn_config=SynthesisConfig(length_scale=l.get("len",1.12),noise_scale=0.6,noise_w_scale=0.7))
    with wave.open(p) as w: d=w.getnframes()/w.getframerate()
    meta.append({"i":i,"text":l["text"],"dur":round(d,3)}); print(f"{i:02d} {d:5.2f}s  {l['text']}")
json.dump(meta,open(f"{out}/meta.json","w"),ensure_ascii=False,indent=1)
