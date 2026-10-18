"""Pure-stdlib Standard MIDI File writer (no external deps!)."""
import struct


def _varlen(n: int) -> bytes:
    n = max(0, int(round(n)))
    out = [n & 0x7F]
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _chunk(tag: bytes, body: bytes) -> bytes:
    return tag + struct.pack(">I", len(body)) + body


def _mtrk(events) -> bytes:
    body = b"".join(_varlen(d) + e for d, e in events)
    body += _varlen(0) + b"\xff\x2f\x00"          # end of track
    return _chunk(b"MTrk", body)


def write_midi(notes, path: str, bpm: float = 120.0,
               program: int = 0, tpq: int = 480) -> None:
    tempo = int(round(60000000.0 / bpm))
    meta = [(0, b"\xff\x51\x03" + tempo.to_bytes(3, "big"))]

    sec_per_tick = 60.0 / bpm / tpq
    raw = []
    for n in notes:
        raw.append((round(n.start / sec_per_tick), 1,
                    bytes([0x90, n.pitch & 0x7F, n.velocity & 0x7F])))
        raw.append((round(n.end / sec_per_tick), 0,
                    bytes([0x80, n.pitch & 0x7F, 0])))
    raw.sort(key=lambda x: (x[0], x[1]))           # note_offs before note_ons

    events, last = [], 0
    events.append((0, bytes([0xC0, program & 0x7F])))
    for t, _, e in raw:
        events.append((t - last, e))
        last = t

    data = (b"MThd" + struct.pack(">IHHH", 6, 1, 2, tpq)
            + _mtrk(meta) + _mtrk(events))
    with open(path, "wb") as f:
        f.write(data)
