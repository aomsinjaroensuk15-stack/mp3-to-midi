"""Tempo estimation + grid quantization."""
import numpy as np

from .notes import Note


def estimate_bpm(onset_times: np.ndarray, default: float = 120.0) -> float:
    if len(onset_times) < 4:
        return default
    ioi = np.diff(onset_times)
    ioi = ioi[(ioi > 0.05) & (ioi < 2.0)]
    if len(ioi) == 0:
        return default
    bpm = 60.0 / float(np.median(ioi))
    if abs(bpm - round(bpm)) < 3:
        bpm = float(round(bpm))
    while bpm < 60:
        bpm *= 2
    while bpm > 200:
        bpm /= 2
    return bpm


def quantize_notes(notes: list[Note], bpm: float,
                   grid_div: int = 4, tpq: int = 480) -> list[Note]:
    grid_ticks = max(1, tpq // grid_div)
    sec_per_tick = 60.0 / bpm / tpq
    out = []
    for n in notes:
        s = int(round(n.start / sec_per_tick / grid_ticks)) * grid_ticks
        e = max(s + grid_ticks,
                int(round(n.end / sec_per_tick / grid_ticks)) * grid_ticks)
        out.append(Note(s * sec_per_tick, e * sec_per_tick, n.pitch, n.velocity))
    return out
