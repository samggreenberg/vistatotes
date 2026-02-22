"""Audio waveform generation utilities."""

import io
import math
import struct
import wave

from vtsearch.config import SAMPLE_RATE


def generate_wav(frequency: float, duration: float) -> bytes:
    """Generate a mono PCM WAV file containing a pure sine-wave tone.

    Produces a single-channel, 16-bit signed PCM WAV at the sample rate
    defined by ``SAMPLE_RATE`` (typically 48 000 Hz). The amplitude is fixed
    at 50 % of the maximum value (32767) to avoid clipping.

    Args:
        frequency: Frequency of the sine wave in Hz (e.g. 440.0 for concert A).
        duration: Length of the tone in seconds.

    Returns:
        A ``bytes`` object containing a valid WAV file that can be written
        directly to disk or streamed as ``audio/wav``.
    """
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
