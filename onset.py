"""Onset detection via spectral flux."""
import numpy as np


def spectral_flux(mag: np.ndarray) -> np.ndarray:
    m = np.log1p(1000.0 * mag)
    flux = np.maximum(0.0, np.diff(m, axis=0)).sum(axis=1)
    return np.concatenate([[0.0], flux])


def detect_onsets(flux: np.ndarray, sr: int, hop: int,
                  sensitivity: float = 1.5,
                  min_dist_ms: float = 30.0):
    """Returns (times_in_seconds, strengths)."""
    pos = flux[flux > 0]
    if len(pos) == 0:
        return np.array([]), np.array([])
    th = np.median(flux) + sensitivity * (np.std(pos) + 1e-9)
    min_d = max(1, int(min_dist_ms / 1000.0 * sr / hop))
    peaks, last = [], -10 ** 9
    for i in range(2, len(flux) - 2):
        if (flux[i] > th and flux[i] == max(flux[i - 2:i + 3])
                and i - last >= min_d):
            peaks.append(i)
            last = i
    times = np.array(peaks) * hop / sr
    strengths = flux[peaks] if peaks else np.array([])
    return times, strengths
