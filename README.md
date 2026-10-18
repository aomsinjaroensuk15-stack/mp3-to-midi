# mp3-to-midi 🎹

From-scratch MP3/WAV → MIDI transcription in pure Python (numpy only).
No plugins, no heavy dependencies, no black boxes — every stage of the
pipeline is our own code.

## Install

```bash
pip install -r requirements.txt   # just numpy
# mp3 decoding needs ffmpeg on your PATH (wav files work without it)
```

## Quick start

```bash
# basic usage
python -m mp32midi song.mp3

# melody-only input (humming, flute, lead synth)
python -m mp32midi song.mp3 --mode mono

# EDM drop with fast note spam -> aggressive settings + saw lead
python -m mp32midi open_window_vgh.mp3 --quality drop --mode poly --bpm 170

# keep human timing instead of snapping to the grid
python -m mp32midi song.mp3 --no-quantize
```

## How it works (the pipeline)

| Stage | File | What it does |
|---|---|---|
| decode | `audio_io.py` | mp3 → waveform (ffmpeg), wav via stdlib |
| features | `features.py` | STFT + per-semitone "piano spectrum" |
| onsets | `onset.py` | spectral flux → note start times |
| pitch | `pitch.py` | YIN algorithm (mono), semitone peaks (poly) |
| notes | `notes.py` | frames → Note(start, end, pitch, velocity) |
| tempo | `quantize.py` | BPM estimation + 16th-note grid snapping |
| output | `midi_out.py` | pure-stdlib Standard MIDI File writer |

## Quality presets

- `standard` — balanced settings for normal music
- `drop` — high time resolution (5.8 ms frames), low note-merge thresholds,
  8-voice polyphony, saw lead (program 81). Made so fast repeated notes in
  EDM drops ("note spam") survive transcription instead of being squashed.

## Test strategy (level up!)

```bash
python tests/test_pipeline.py
```

- **Level 1/2** — synthetic scale (mono): pitch tracking must be exact
- **Level 3** — synthetic chord (poly): all chord tones detected
- **Drop test** — 32-note spam at ~9 notes/sec: all notes must survive

Final boss: a real dense electronic track. For that, the DSP engine is the
honest attempt; if it fights back, the ML engine is the cheat code:

```bash
pip install basic-pitch
python -m mp32midi song.mp3 --engine ml
```

## Roadmap

- [x] Phases 0–6: decode → MIDI
- [x] mono + poly engines
- [x] drop preset
- [ ] drum/percussion track
- [ ] vibrato/pitch-bend export
- [ ] web UI (drag & drop)
