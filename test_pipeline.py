"""Run the synthetic levels through the whole pipeline."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from make_test_tones import synth_scale, synth_chord, synth_drop
from mp32midi.config import make_config
from mp32midi.pipeline import run
from mp32midi.midi_out import write_midi


def test_scale_mono():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "scale.wav")
        synth_scale(p)
        cfg = make_config("standard")
        cfg.quantize = False
        notes, bpm = run(p, cfg, mode="mono")
        pitches = sorted({n.pitch for n in notes})
        assert len(notes) >= 3, f"expected >=3 notes, got {len(notes)}: {pitches}"
        for target in (69, 72, 76):
            assert any(abs(m - target) <= 1 for m in pitches), \
                f"missed MIDI {target}, got {pitches}"
        print(f"  scale mono OK: {len(notes)} notes {pitches}")


def test_chord_poly():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "chord.wav")
        synth_chord(p)
        cfg = make_config("standard")
        cfg.quantize = False
        notes, bpm = run(p, cfg, mode="poly")
        pitches = {n.pitch for n in notes}
        for target in (57, 60, 64):
            assert any(abs(m - target) <= 1 for m in pitches), \
                f"missed chord tone {target}, got {sorted(pitches)}"
        print(f"  chord poly OK: {len(notes)} notes {sorted(pitches)}")


def test_drop_spam():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "drop.wav")
        synth_drop(p)
        out = os.path.join(d, "drop.mid")
        cfg = make_config("drop")
        cfg.quantize = False
        notes, bpm = run(p, cfg, mode="poly")
        pitches = {n.pitch for n in notes}
        assert len(notes) >= 16, f"drop spam lost notes: {len(notes)}"
        assert all(any(abs(m - t) <= 1 for m in pitches) for t in (69, 76)), \
            f"drop should be A4/E5, got {sorted(pitches)}"
        write_midi(notes, out, bpm=bpm, program=cfg.program)
        with open(out, "rb") as f:
            assert f.read(4) == b"MThd", "not a valid MIDI file"
        print(f"  drop spam OK: {len(notes)} notes - spam survived!")


if __name__ == "__main__":
    test_scale_mono()
    test_chord_poly()
    test_drop_spam()
    print("ALL TESTS PASSED")
