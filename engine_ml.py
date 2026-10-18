"""Optional ML engine for the final boss (dense polyphonic songs).

Uses Spotify basic-pitch under the hood; install with:
    pip install basic-pitch
"""
import os


def run_ml(path: str, cfg=None):
    try:
        from basic_pitch.inference import predict_and_save
    except ImportError:
        raise SystemExit(
            "ML engine needs the basic-pitch package:\n"
            "    pip install basic-pitch\n"
            "(or use the default from-scratch DSP engine instead)"
        )
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    predict_and_save(
        [path], out_dir,
        save_midi=True, sonify_midi=False,
        save_model_outputs=False, save_notes=False,
    )
    stem = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(out_dir, stem + "_basic_pitch.mid")
    print(f"ML engine wrote: {out}")
    return out
