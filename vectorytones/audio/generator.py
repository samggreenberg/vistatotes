"""Audio waveform generation utilities."""

import io
import math
import struct
import wave

from config import SAMPLE_RATE


def generate_wav(frequency: float, duration: float) -> bytes:
    """Return raw WAV bytes for a sine-wave tone."""
    num_samples = int(SAMPLE_RATE * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        samples = []
        for i in range(num_samples):
            t = i / SAMPLE_RATE
            value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
            samples.append(struct.pack("<h", value))
        wf.writeframes(b"".join(samples))
    return buf.getvalue()
