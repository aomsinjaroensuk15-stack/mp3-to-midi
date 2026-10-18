"""Full DSP pipeline: audio -> onsets -> pitches -> notes -> quantization."""
import numpy as np

from .audio_io import load
from .features import stft, piano_spectrum, frame_rms
from .onset import spectral_flux, detect_onsets
from .pitch import pitch_track, freq_to_midi
from .notes import track_monophonic, track_poly
from .quantize import estimate_bpm, quantize_notes


def run(path: str, cfg, mode: str = "poly"):
    """Transcribe an audio file. Returns (notes, bpm)."""
    y = load(path, cfg.sr)
    mag = stft(y, cfg.n_fft, cfg.hop)
    flux = spectral_flux(mag)
    onsets, strengths = detect_onsets(flux, cfg.sr, cfg.hop,
                                      cfg.onset_sensitivity,
                                      cfg.min_onset_dist_ms)
    if mode == "mono":
        freqs = pitch_track(y, cfg.sr, cfg)
        amps = frame_rms(y, cfg.n_fft, cfg.hop)
        amps = amps / max(1e-9, amps.max())
        midis = np.array([freq_to_midi(f) if f > 0 else 0 for f in freqs])
        notes = track_monophonic(midis, amps, cfg.sr, cfg.hop, cfg)
    elif mode == "poly":
        S = piano_spectrum(mag, cfg.sr, cfg.n_fft)
        notes = track_poly(onsets, S, cfg.sr, cfg.hop, cfg)
    else:
        raise ValueError(f"unknown mode: {mode}")

    bpm = cfg.bpm or estimate_bpm(onsets)
    if cfg.quantize:
        notes = quantize_notes(notes, bpm, cfg.grid_div, cfg.ticks_per_beat)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes, bpm
