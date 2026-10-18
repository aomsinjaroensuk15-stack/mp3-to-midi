from dataclasses import dataclass, replace


@dataclass
class Config:
    # audio
    sr: int = 22050          # sample rate everything is resampled to
    n_fft: int = 2048        # analysis window (~93 ms)
    hop: int = 256           # frame hop (~11.6 ms @ 22050)

    # pitch (YIN)
    fmin: float = 50.0       # lowest detectable frequency (Hz)
    fmax: float = 2000.0     # highest detectable frequency (Hz)
    yin_thresh: float = 0.18

    # onsets
    onset_sensitivity: float = 1.5
    min_onset_dist_ms: float = 30.0

    # note formation
    min_note_ms: float = 45.0
    min_gap_ms: float = 20.0
    max_poly: int = 6        # max simultaneous notes (poly mode)
    peak_thresh: float = 0.25

    # output
    bpm: float | None = None # None = auto-estimate
    quantize: bool = True
    grid_div: int = 4        # 4 = sixteenth notes
    program: int = 0         # MIDI instrument (0 = grand piano, 81 = saw lead)
    ticks_per_beat: int = 480


# The "drop" preset: aggressive settings so fast repeated notes in drops
# (note spam) survive instead of being merged into a mushy sustain.
PRESETS = {
    "standard": Config(),
    "drop": Config(
        hop=128,               # ~5.8 ms frames -> catches 32nd notes
        min_note_ms=20.0,
        min_gap_ms=10.0,
        max_poly=8,
        onset_sensitivity=1.1,
        peak_thresh=0.18,
        program=81,            # lead square/saw fits EDM drops
    ),
}


def make_config(quality: str = "standard", **overrides) -> Config:
    cfg = PRESETS.get(quality, PRESETS["standard"])
    clean = {k: v for k, v in overrides.items() if v is not None}
    return replace(cfg, **clean)
