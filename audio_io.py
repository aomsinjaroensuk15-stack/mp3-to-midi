"""Load audio: WAV via stdlib, anything else (mp3/flac/ogg) via ffmpeg."""
import os
import subprocess
import wave

import numpy as np


def load(path: str, sr: int = 22050) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        y, file_sr = _load_wav(path)
    else:
        y = _decode_with_ffmpeg(path, sr)
        file_sr = sr
    if file_sr != sr:
        y = resample(y, file_sr, sr)
    y = y.astype(np.float32)
    peak = float(np.max(np.abs(y))) if len(y) else 1.0
    if peak > 0:
        y = y / peak
    return y


def resample(y: np.ndarray, orig: int, target: int) -> np.ndarray:
    if orig == target or len(y) == 0:
        return y
    t_old = np.arange(len(y)) / orig
    t_new = np.arange(int(len(y) * target / orig)) / target
    return np.interp(t_new, t_old, y).astype(np.float32)


def _load_wav(path: str):
    with wave.open(path, "rb") as w:
        ch, sw, fr, n = (w.getnchannels(), w.getsampwidth(),
                         w.getframerate(), w.getnframes())
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width: {sw}")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, fr


def _decode_with_ffmpeg(path: str, sr: int) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le",
           "-ac", "1", "-ar", str(sr), "-"]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError:
        raise SystemExit(
            "ffmpeg not found. Install ffmpeg (https://ffmpeg.org) "
            "or convert your file to .wav first."
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"ffmpeg failed to decode '{path}':\n"
                         + e.stderr.decode(errors="replace")[:500])
    return np.frombuffer(out.stdout, dtype="<f4")
