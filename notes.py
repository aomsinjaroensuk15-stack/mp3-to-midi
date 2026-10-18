"""Turn frame-level detections into Note objects (mono + poly)."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Note:
    start: float   # seconds
    end: float     # seconds
    pitch: int     # MIDI note number
    velocity: int = 80


def _velocity(amp: float) -> int:
    return int(np.clip(30 + 97.0 * amp, 30, 127))


def _merge_gaps(notes, min_gap_ms: float):
    if not notes:
        return notes
    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    out = [notes[0]]
    for n in notes[1:]:
        prev = out[-1]
        gap = (n.start - prev.end) * 1000.0
        if n.pitch == prev.pitch and 0 <= gap < min_gap_ms:
            prev.end = max(prev.end, n.end)
            prev.velocity = max(prev.velocity, n.velocity)
        else:
            out.append(n)
    return out


def track_monophonic(midis: np.ndarray, amps: np.ndarray,
                     sr: int, hop: int, cfg) -> list[Note]:
    """Group a per-frame MIDI track into notes (3-frame median filter)."""
    sm = midis.copy()
    for i in range(2, len(midis) - 2):
        sm[i] = int(np.median(midis[i - 2:i + 3]))

    notes, i, n = [], 0, len(sm)
    while i < n:
        if sm[i] <= 0:
            i += 1
            continue
        p, j = sm[i], i
        while j < n and sm[j] == p:
            j += 1
        start, end = i * hop / sr, j * hop / sr
        if (end - start) * 1000.0 >= cfg.min_note_ms:
            amp = float(amps[i:j].max()) if amps[i:j].size else 0.5
            notes.append(Note(start, end, int(p), _velocity(amp / max(1e-9, amps.max()))))
        i = j
    return _merge_gaps(notes, cfg.min_gap_ms)


def track_poly(onsets: np.ndarray, S: np.ndarray,
               sr: int, hop: int, cfg) -> list[Note]:
    """At each onset, pick the strongest semitone peaks -> note starts."""
    n_frames = S.shape[0]
    onsets_f = np.round(onsets * sr / hop).astype(int)

    def pick_peaks(f_idx: int):
        f0, f1 = max(0, f_idx - 2), min(n_frames, f_idx + 3)
        s = S[f0:f1].mean(axis=0)
        th = cfg.peak_thresh * (s.max() + 1e-9)
        cand = [m for m in range(1, 127)
                if s[m] > th and s[m] >= s[m - 1] and s[m] >= s[m + 1]]
        cand.sort(key=lambda m: -s[m])
        chosen = []
        for m in cand:
            if all(abs(m - c) > 2 for c in chosen):
                chosen.append(m)
            if len(chosen) >= cfg.max_poly:
                break
        return chosen

    notes = []
    for f_idx, t in zip(onsets_f, onsets):
        for p in pick_peaks(f_idx):
            # sustain: keep note open while its band stays reasonably hot
            j = f_idx + 1
            while j < n_frames and S[j, p] > 0.25 * max(S[f_idx, p], 1e-9):
                j += 1
            end = min(j, n_frames - 1) * hop / sr
            amp = float(S[f_idx:max(f_idx + 1, j), p].max())
            notes.append(Note(float(t), float(end), int(p), _velocity(amp)))
    return notes
