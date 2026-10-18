"""Synthetic test tones - Level 1 of the test strategy."""
import wave

import numpy as np

SR = 22050


def tone(freq, dur, harm=(1.0, 0.5, 0.25)):
    t = np.arange(int(SR * dur)) / SR
    env = np.minimum(1.0, t / 0.01) * np.exp(-t / (dur * 0.6))
    sig = sum(a * np.sin(2 * np.pi * freq * h * t)
              for h, a in enumerate(harm, start=1) if a)
    return sig * env


def save_wav(path, y):
    y = np.clip(y, -1, 1)
    pcm = (y * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def synth_scale(path):
    """A4 - C5 - E5 - G5, 0.4 s each (Level 2: easy monophonic)."""
    seq = [440.0, 523.25, 659.25, 783.99]   # MIDI 69, 72, 76, 79
    y = np.concatenate([tone(f, 0.4) for f in seq] + [np.zeros(SR // 4)])
    save_wav(path, y)


def synth_chord(path):
    """A3 + C4 + E4 sustained (Level 3: simple polyphonic)."""
    y = sum(tone(f, 1.2, harm=(1.0, 0.4, 0.2))
            for f in [220.0, 261.63, 329.63])  # MIDI 57, 60, 64
    save_wav(path, y)


def synth_drop(path):
    """32 rapid alternating notes A4/E5 - the 'note spam' pattern."""
    y = np.concatenate([tone(f, 0.11, harm=(1.0, 0.6, 0.3))
                        for f in [440.0, 659.25]] * 16)
    save_wav(path, y)
