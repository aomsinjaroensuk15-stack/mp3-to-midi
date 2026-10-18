"""Command-line interface."""
import argparse
import os

from .config import make_config


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="mp32midi",
        description="From-scratch MP3/WAV -> MIDI transcription")
    p.add_argument("input", help="input audio file (.mp3, .wav, ...)")
    p.add_argument("-o", "--output", default=None, help="output .mid path")
    p.add_argument("--engine", choices=["dsp", "ml"], default="dsp",
                   help="dsp = our own code, ml = basic-pitch (pip install basic-pitch)")
    p.add_argument("--mode", choices=["mono", "poly"], default="poly",
                   help="mono = single melody line, poly = chords/drop spam")
    p.add_argument("--quality", choices=["standard", "drop"], default="standard",
                   help="'drop' = aggressive settings for fast EDM note spam")
    p.add_argument("--bpm", type=float, default=None, help="force tempo (else auto)")
    p.add_argument("--program", type=int, default=None,
                   help="MIDI instrument (0=piano, 81=saw lead, ...)")
    p.add_argument("--no-quantize", action="store_true",
                   help="keep raw micro-timing instead of snapping to grid")
    a = p.parse_args(argv)

    if a.engine == "ml":
        from .engine_ml import run_ml
        run_ml(a.input)
        return

    cfg = make_config(a.quality, bpm=a.bpm, program=a.program)
    if a.no_quantize:
        cfg.quantize = False

    from .pipeline import run
    from .midi_out import write_midi
    notes, bpm = run(a.input, cfg, mode=a.mode)
    if not notes:
        print("WARNING: no notes detected - try --mode mono, another --quality,")
        print("or check that the input file has audible content.")
        return
    out = a.output or os.path.splitext(a.input)[0] + ".mid"
    write_midi(notes, out, bpm=bpm, program=cfg.program,
               tpq=cfg.ticks_per_beat)
    print(f"OK wrote {out} | {len(notes)} notes | {bpm:.1f} BPM "
          f"| mode={a.mode} quality={a.quality}")


if __name__ == "__main__":
    main()
