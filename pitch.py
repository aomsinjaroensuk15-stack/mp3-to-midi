"""Pitch: YIN for monophonic tracking + note<->frequency helpers."""
import numpy as np


def freq_to_midi(f: float) -> int:
    return int(round(69 + 12.0 * np.log2(f / 440.0)))


def midi_to_freq(m: int) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def yin_pitch(x: np.ndarray, sr: int, fmin: float, fmax: float,
              thresh: float = 0.18) -> float:
    """YIN pitch of a single frame. Returns 0.0 if it looks unvoiced."""
    x = x.astype(np.float64)
    x = x - x.mean()
    W = len(x)
    if W < 16 or np.sqrt(np.mean(x ** 2)) < 1e-5:
        return 0.0
    min_tau = max(2, int(sr / fmax))
    max_tau = min(int(sr / fmin), W - 1)
    if min_tau >= max_tau:
        return 0.0

    sq = x ** 2
    cum = np.concatenate([[0.0], np.cumsum(sq)])
    corr = np.correlate(x, x, "full")[W - 1:]          # corr[tau] = sum x_j * x_{j+tau}
    taus = np.arange(1, max_tau + 1)
    d = cum[W - taus] + (cum[W] - cum[taus]) - 2.0 * corr[taus]
    d[d < 0] = 0.0

    # cumulative mean normalized difference function
    cmnd = d * taus / np.maximum(np.cumsum(d), 1e-12)

    region = cmnd[min_tau:]
    below = np.where(region < thresh)[0]
    if len(below):
        t0 = below[0] + min_tau
        tau = t0 + int(np.argmin(cmnd[t0:min(t0 + 8, len(cmnd))]))
    else:
        tau = min_tau + int(np.argmin(region))

    # parabolic interpolation on the difference function
    if 1 <= tau < len(d) - 1:
        y0, y1, y2 = d[tau - 1], d[tau], d[tau + 1]
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            tau = tau + 0.5 * (y0 - y2) / denom
    return sr / tau


def pitch_track(y: np.ndarray, sr: int, cfg) -> np.ndarray:
    """YIN pitch per frame (Hz, 0 = unvoiced)."""
    n = 1 + max(0, (len(y) - cfg.n_fft) // cfg.hop)
    idx = np.arange(cfg.n_fft)[None, :] + cfg.hop * np.arange(n)[:, None]
    return np.array([yin_pitch(y[i], sr, cfg.fmin, cfg.fmax, cfg.yin_thresh)
                     for i in idx])
