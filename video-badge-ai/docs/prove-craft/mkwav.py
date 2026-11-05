import wave, struct, math, sys
def wav(path, freq, dur, amp, sr=48000):
    w = wave.open(path, 'wb'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    n = int(dur*sr)
    w.writeframes(b''.join(struct.pack('<h', int(amp*32767*math.sin(2*math.pi*freq*i/sr))) for i in range(n)))
    w.close()
wav('voce_test.wav', 220, 2.0, 0.5)
wav('musica_test.wav', 440, 4.0, 0.3)
