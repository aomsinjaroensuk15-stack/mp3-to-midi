"""Spectral features: STFT and a per-semitone 'piano spectrum'."""
import numpy as np


def stft(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """Magnitude STFT, frames centered on the signal."""
    pad = n_fft // 2
    y = np.concatenate([np.zeros(pad), y, np.zeros(pad + n_fft)])
    n_frames = (len(y) - n_fft) // hop + 1
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    win = np.hanning(n_fft).astype(np.float32)
    frames = y[idx].astype(np.float32) * win
    return np.abs(np.fft.rfft(frames, axis=1))


def piano_spectrum(mag: np.ndarray, sr: int, n_fft: int,
                   fmin: float = 27.5, fmax: float = 4186.0) -> np.ndarray:
    """Fold the linear spectrum into 128 semitone bands (max per band),
    normalized per frame. Result: (n_frames, 128), like a crude CQT."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    S = np.zeros((mag.shape[0], 128), dtype=np.float32)
    for m in range(128):
        f = 440.0 * 2.0 ** ((m - 69) / 12.0)
        if f < fmin or f > fmax:
            continue
        band = np.where((freqs >= f * 2.0 ** (-0.35 / 12.0)) &
                        (freqs <= f * 2.0 ** (0.35 / 12.0)))[0]
        if len(band):
            S[:, m] = mag[:, band].max(axis=1)
    peak = S.max(axis=1, keepdims=True)
    peak[peak == 0] = 1.0
    return S / peak


def frame_rms(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(y) - n_fft) // hop)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
    fr = y[idx]
    return np.sqrt((fr ** 2).mean(axis=1))
