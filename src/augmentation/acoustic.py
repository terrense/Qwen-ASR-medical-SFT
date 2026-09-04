# -*- coding: utf-8 -*-
"""Phase 7 - Deterministic acoustic augmentation.

Design rules:

  deterministic on seed
      Every random choice is drawn from a generator seeded with
      ``hash(global_seed, utt_id)``. Re-running the pipeline reproduces every
      waveform bit-for-bit, and a single utterance can be regenerated without
      replaying the whole corpus.

  the transcript always belongs to the foreground speaker
      Competing speech is mixed in at a signal-to-interference ratio drawn from
      U(6, 15) dB, so the target talker stays dominant. The interfering
      utterance is drawn from a *different* speaker and a *different* script,
      and its identity is recorded in the metadata so any transcript
      contamination can be audited afterwards.

  condition mix (Phase 7 target)
      40% clean, 30% environmental noise / reverberation, 15% competing speech,
      15% channel or codec degradation.

Noise and impulse responses are taken from a corpus directory when one is
supplied. When none is available the module synthesizes colored noise and
exponential-decay impulse responses instead; that fallback is recorded in the
metadata of every affected utterance so a run built on synthetic noise can never
be mistaken for one built on recorded noise.
"""
from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import numpy as np

SAMPLE_RATE = 16000

CONDITION_WEIGHTS = OrderedDict([
    ("clean", 0.40),
    ("noise", 0.18),
    ("reverb", 0.06),
    ("noise_reverb", 0.06),
    ("competing_speech", 0.15),
    ("codec", 0.15),
])

SNR_RANGE = (5.0, 20.0)     # dB, additive noise
SIR_RANGE = (6.0, 15.0)     # dB, competing speech
GAIN_RANGE_DB = (-6.0, 6.0)
RT60_RANGE = (0.15, 0.7)    # seconds


def utterance_rng(global_seed, utt_id):
    """A generator whose stream depends only on (global_seed, utt_id)."""
    digest = hashlib.sha256(("%d::%s" % (global_seed, utt_id)).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def rms(signal):
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64)) + 1e-12))


def scale_to_snr(target, noise, snr_db):
    """Scale ``noise`` so that target-to-noise ratio equals ``snr_db``."""
    target_rms = rms(target)
    noise_rms = rms(noise)
    if noise_rms < 1e-9:
        return noise
    desired = target_rms / (10.0 ** (snr_db / 20.0))
    return noise * (desired / noise_rms)


def fit_length(signal, length, rng):
    """Crop at a random offset or tile until ``signal`` has ``length`` samples."""
    if len(signal) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(signal) >= length:
        start = int(rng.integers(0, len(signal) - length + 1))
        return signal[start:start + length]
    repeats = int(math.ceil(length / len(signal)))
    return np.tile(signal, repeats)[:length]


def colored_noise(length, rng, color="pink"):
    """Synthesized broadband noise with a 1/f^alpha spectrum."""
    alpha = {"white": 0.0, "pink": 1.0, "brown": 2.0}.get(color, 1.0)
    spectrum_len = length // 2 + 1
    freqs = np.arange(spectrum_len, dtype=np.float64)
    freqs[0] = 1.0
    magnitude = freqs ** (-alpha / 2.0)
    phase = rng.uniform(0, 2 * np.pi, spectrum_len)
    spectrum = magnitude * np.exp(1j * phase)
    signal = np.fft.irfft(spectrum, n=length)
    peak = np.max(np.abs(signal))
    return (signal / peak).astype(np.float32) if peak > 0 else signal.astype(np.float32)


def synthetic_rir(rng, sample_rate=SAMPLE_RATE, rt60=None):
    """Exponential-decay impulse response.

    A first-order approximation of a small room: white noise shaped by an
    exponential envelope with the requested RT60, plus a direct-path impulse.
    Cheap, well-defined, and adequate for a reverberation *condition* - it is
    not a substitute for measured impulse responses, and the metadata says so.
    """
    if rt60 is None:
        rt60 = float(rng.uniform(*RT60_RANGE))
    length = int(sample_rate * min(rt60 * 1.5, 1.5))
    time = np.arange(length) / sample_rate
    envelope = np.exp(-6.907 * time / rt60)  # -60 dB at t = rt60
    impulse = rng.normal(0, 1, length) * envelope
    direct = int(sample_rate * rng.uniform(0.001, 0.006))
    if direct < length:
        impulse[direct] += 1.0
    energy = np.sqrt(np.sum(impulse ** 2))
    return (impulse / energy).astype(np.float32) if energy > 0 else impulse.astype(np.float32), rt60


def apply_reverb(signal, impulse):
    """Convolve and trim back to the original length, preserving alignment."""
    wet = np.convolve(signal, impulse, mode="full")[:len(signal)]
    dry_rms, wet_rms = rms(signal), rms(wet)
    if wet_rms > 1e-9:
        wet = wet * (dry_rms / wet_rms)
    return wet.astype(np.float32)


def mu_law_codec(signal, quantization_bits=8, mu=255.0):
    """G.711-style mu-law companding round trip - a narrowband channel proxy."""
    peak = np.max(np.abs(signal)) or 1.0
    normalized = signal / peak
    compressed = np.sign(normalized) * np.log1p(mu * np.abs(normalized)) / np.log1p(mu)
    levels = 2 ** quantization_bits
    quantized = np.round((compressed + 1.0) / 2.0 * (levels - 1))
    dequantized = quantized / (levels - 1) * 2.0 - 1.0
    expanded = np.sign(dequantized) * ((1 + mu) ** np.abs(dequantized) - 1) / mu
    return (expanded * peak).astype(np.float32)


def resample_degrade(signal, rng, sample_rate=SAMPLE_RATE):
    """Downsample to a telephone-ish rate and back, discarding high frequencies."""
    target = int(rng.choice([8000, 11025, 12000]))
    ratio = target / sample_rate
    down_len = max(1, int(len(signal) * ratio))
    # Linear interpolation both ways: intentionally imperfect, which is the point.
    down = np.interp(np.linspace(0, len(signal) - 1, down_len),
                     np.arange(len(signal)), signal)
    up = np.interp(np.linspace(0, down_len - 1, len(signal)),
                   np.arange(down_len), down)
    return up.astype(np.float32), target


def apply_gain(signal, gain_db):
    return (signal * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def prevent_clipping(signal, ceiling=0.99):
    """Scale down if the mix exceeds full scale. Records nothing on its own."""
    peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
    if peak > ceiling:
        return (signal * (ceiling / peak)).astype(np.float32), peak
    return signal.astype(np.float32), peak


class NoiseBank:
    """Noise and impulse-response provider.

    ``noise_paths`` / ``rir_paths`` are lists of WAV files. When either list is
    empty the corresponding material is synthesized and the metadata records
    ``source="synthetic"`` so the distinction survives into the results.
    """

    def __init__(self, noise_paths=None, rir_paths=None, sample_rate=SAMPLE_RATE):
        self.noise_paths = list(noise_paths or [])
        self.rir_paths = list(rir_paths or [])
        self.sample_rate = sample_rate
        self._cache = {}

    def _load(self, path):
        if path not in self._cache:
            import soundfile as sf

            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if sr != self.sample_rate:
                ratio = self.sample_rate / sr
                new_len = int(len(data) * ratio)
                data = np.interp(np.linspace(0, len(data) - 1, new_len),
                                 np.arange(len(data)), data).astype(np.float32)
            self._cache[path] = data
        return self._cache[path]

    def draw_noise(self, length, rng):
        if self.noise_paths:
            path = self.noise_paths[int(rng.integers(0, len(self.noise_paths)))]
            return fit_length(self._load(path), length, rng), {
                "source": "corpus", "path": path}
        color = str(rng.choice(["white", "pink", "brown"]))
        return colored_noise(length, rng, color), {
            "source": "synthetic", "color": color}

    def draw_rir(self, rng):
        if self.rir_paths:
            path = self.rir_paths[int(rng.integers(0, len(self.rir_paths)))]
            return self._load(path), {"source": "corpus", "path": path}
        impulse, rt60 = synthetic_rir(rng, self.sample_rate)
        return impulse, {"source": "synthetic", "rt60_seconds": round(rt60, 3)}


def choose_condition(rng, weights=None):
    weights = weights or CONDITION_WEIGHTS
    names = list(weights)
    probabilities = np.array([weights[n] for n in names], dtype=np.float64)
    probabilities = probabilities / probabilities.sum()
    return str(rng.choice(names, p=probabilities))


def augment(signal, utt_id, global_seed, noise_bank, condition=None,
            competing_provider=None, sample_rate=SAMPLE_RATE, weights=None):
    """Apply one augmentation to one waveform.

    ``competing_provider(rng)`` must return ``(waveform, metadata_dict)`` for an
    utterance spoken by a different speaker. When it is absent, a requested
    competing-speech condition degrades to additive noise and the metadata
    records the substitution rather than silently changing the condition mix.

    Returns ``(augmented_signal, metadata)``. The metadata is written straight
    into the manifest, so ``condition``, ``snr`` and ``sir`` are always present.
    """
    rng = utterance_rng(global_seed, utt_id)
    signal = np.asarray(signal, dtype=np.float32)
    condition = condition or choose_condition(rng, weights)

    meta = OrderedDict([
        ("condition", condition),
        ("snr", None),
        ("sir", None),
        ("augmentation_seed", global_seed),
        ("gain_db", None),
        ("noise_source", None),
        ("rir_source", None),
        ("codec", None),
        ("resample_rate", None),
        ("competing_utt_id", None),
        ("competing_speaker_id", None),
        ("input_rms", round(rms(signal), 6)),
        ("clipping_rescaled_from_peak", None),
    ])

    out = signal

    if condition in ("reverb", "noise_reverb"):
        impulse, rir_meta = noise_bank.draw_rir(rng)
        out = apply_reverb(out, impulse)
        meta["rir_source"] = rir_meta

    if condition in ("noise", "noise_reverb"):
        snr_db = float(rng.uniform(*SNR_RANGE))
        noise, noise_meta = noise_bank.draw_noise(len(out), rng)
        out = out + scale_to_snr(out, noise, snr_db)
        meta["snr"] = round(snr_db, 2)
        meta["noise_source"] = noise_meta

    if condition == "competing_speech":
        if competing_provider is None:
            snr_db = float(rng.uniform(*SNR_RANGE))
            noise, noise_meta = noise_bank.draw_noise(len(out), rng)
            out = out + scale_to_snr(out, noise, snr_db)
            meta["condition"] = "noise"
            meta["snr"] = round(snr_db, 2)
            meta["noise_source"] = noise_meta
            meta["competing_speech_unavailable"] = True
        else:
            interferer, interferer_meta = competing_provider(rng)
            sir_db = float(rng.uniform(*SIR_RANGE))
            interferer = fit_length(np.asarray(interferer, dtype=np.float32),
                                    len(out), rng)
            out = out + scale_to_snr(out, interferer, sir_db)
            meta["sir"] = round(sir_db, 2)
            meta["competing_utt_id"] = interferer_meta.get("utt_id")
            meta["competing_speaker_id"] = interferer_meta.get("speaker_id")

    if condition == "codec":
        if rng.random() < 0.5:
            out = mu_law_codec(out, quantization_bits=int(rng.choice([8, 8, 6])))
            meta["codec"] = "mu_law_g711"
        else:
            out, target_rate = resample_degrade(out, rng, sample_rate)
            meta["codec"] = "resample_degrade"
            meta["resample_rate"] = target_rate

    gain_db = float(rng.uniform(*GAIN_RANGE_DB))
    out = apply_gain(out, gain_db)
    meta["gain_db"] = round(gain_db, 2)

    out, peak = prevent_clipping(out)
    if peak > 0.99:
        meta["clipping_rescaled_from_peak"] = round(peak, 4)
    meta["output_rms"] = round(rms(out), 6)
    return out, meta


def plan_conditions(utt_ids, global_seed, weights=None):
    """Assign a condition to every utterance up front.

    Assigning first makes the realized condition mix checkable against the Phase
    7 target before any audio is written, and keeps the assignment stable when
    the corpus is regenerated.
    """
    weights = weights or CONDITION_WEIGHTS
    plan = OrderedDict()
    for utt_id in utt_ids:
        plan[utt_id] = choose_condition(utterance_rng(global_seed, utt_id), weights)
    return plan
