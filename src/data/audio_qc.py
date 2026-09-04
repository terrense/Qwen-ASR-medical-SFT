# -*- coding: utf-8 -*-
"""Phase 6 - Audio quality control.

Checks every generated waveform for defects that would corrupt training, and
normalizes the format to what Qwen3-ASR expects: 16 kHz, mono, PCM WAV.

Deliberately *not* implemented: filtering synthesized audio by Qwen3-ASR's own
CER. Discarding the utterances the target model finds hard would select the
training and test sets in the model's favour and inflate every subsequent
result - the exact selection bias the study is meant to avoid. If a transcript
consistency check is ever needed, ``independent_asr_check`` runs a *different*
ASR system with a deliberately loose threshold, catching only catastrophic TTS
failures (wrong language, empty synthesis, truncation) rather than ranking
utterances by difficulty.

Every rejected sample is written to a removal log with its measured values and
the rule that rejected it, so the corpus can be reconstructed exactly and the
removal rate can be reported.
"""
from __future__ import annotations

import os
from collections import Counter, OrderedDict

import numpy as np

TARGET_SAMPLE_RATE = 16000

# Thresholds are loose on purpose: they target broken audio, not hard audio.
DEFAULTS = OrderedDict([
    ("min_duration", 0.30),        # seconds; shorter than this is a failed render
    ("max_duration", 30.0),        # longer than the corpus design allows
    ("max_clipping_ratio", 0.01),  # >1% of samples at full scale
    ("max_silence_ratio", 0.85),   # mostly silence
    ("min_rms", 1e-4),             # effectively empty
    ("max_rms", 0.99),
    ("max_leading_silence", 2.0),  # seconds
    ("max_trailing_silence", 2.0),
    ("silence_threshold_db", -45.0),  # relative to full scale
])


def load_audio(path, target_sr=TARGET_SAMPLE_RATE):
    """Load as mono float32 at the target rate. Raises on unreadable files."""
    import soundfile as sf

    data, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sample_rate != target_sr:
        ratio = target_sr / sample_rate
        new_length = int(len(data) * ratio)
        data = np.interp(np.linspace(0, len(data) - 1, new_length),
                         np.arange(len(data)), data).astype(np.float32)
    return np.asarray(data, dtype=np.float32), target_sr


def save_audio(path, signal, sample_rate=TARGET_SAMPLE_RATE, subtype="PCM_16"):
    import soundfile as sf

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    sf.write(path, np.asarray(signal, dtype=np.float32), sample_rate, subtype=subtype)
    return path


def _silence_mask(signal, threshold_db):
    """Per-sample silence mask from a short-window energy envelope."""
    window = max(1, int(0.02 * TARGET_SAMPLE_RATE))  # 20 ms
    padded = np.pad(np.abs(signal), (window // 2, window // 2), mode="edge")
    kernel = np.ones(window) / window
    envelope = np.convolve(padded, kernel, mode="same")[window // 2:window // 2 + len(signal)]
    threshold = 10.0 ** (threshold_db / 20.0)
    return envelope < threshold


def measure(signal, sample_rate=TARGET_SAMPLE_RATE, silence_threshold_db=-45.0):
    """All quality statistics for one waveform."""
    n = len(signal)
    duration = n / float(sample_rate) if sample_rate else 0.0
    finite = np.isfinite(signal)
    n_nonfinite = int((~finite).sum())
    clean = np.where(finite, signal, 0.0)

    peak = float(np.max(np.abs(clean))) if n else 0.0
    rms = float(np.sqrt(np.mean(np.square(clean, dtype=np.float64)))) if n else 0.0
    clipping_ratio = float(np.mean(np.abs(clean) >= 0.999)) if n else 0.0

    if n:
        silent = _silence_mask(clean, silence_threshold_db)
        silence_ratio = float(silent.mean())
        voiced = np.flatnonzero(~silent)
        if len(voiced):
            leading = voiced[0] / float(sample_rate)
            trailing = (n - 1 - voiced[-1]) / float(sample_rate)
        else:
            leading = trailing = duration
    else:
        silence_ratio, leading, trailing = 1.0, 0.0, 0.0

    return OrderedDict([
        ("n_samples", n),
        ("duration", round(duration, 4)),
        ("sample_rate", sample_rate),
        ("peak", round(peak, 6)),
        ("rms", round(rms, 6)),
        ("clipping_ratio", round(clipping_ratio, 6)),
        ("silence_ratio", round(silence_ratio, 4)),
        ("leading_silence", round(leading, 4)),
        ("trailing_silence", round(trailing, 4)),
        ("n_nonfinite", n_nonfinite),
        ("is_all_zero", bool(n and peak == 0.0)),
    ])


def evaluate(stats, thresholds=None):
    """Apply the rules. Returns (passed, [reasons])."""
    rules = dict(DEFAULTS)
    rules.update(thresholds or {})
    reasons = []

    if stats["n_samples"] == 0:
        reasons.append("empty audio")
    if stats["n_nonfinite"] > 0:
        reasons.append("contains %d NaN/inf samples" % stats["n_nonfinite"])
    if stats["is_all_zero"]:
        reasons.append("all-zero waveform")
    if stats["duration"] < rules["min_duration"]:
        reasons.append("duration %.3fs below minimum %.2fs"
                       % (stats["duration"], rules["min_duration"]))
    if stats["duration"] > rules["max_duration"]:
        reasons.append("duration %.3fs above maximum %.2fs"
                       % (stats["duration"], rules["max_duration"]))
    if stats["clipping_ratio"] > rules["max_clipping_ratio"]:
        reasons.append("clipping ratio %.4f above %.4f"
                       % (stats["clipping_ratio"], rules["max_clipping_ratio"]))
    if stats["silence_ratio"] > rules["max_silence_ratio"]:
        reasons.append("silence ratio %.3f above %.3f"
                       % (stats["silence_ratio"], rules["max_silence_ratio"]))
    if stats["rms"] < rules["min_rms"]:
        reasons.append("rms %.6f below %.6f" % (stats["rms"], rules["min_rms"]))
    if stats["rms"] > rules["max_rms"]:
        reasons.append("rms %.6f above %.6f" % (stats["rms"], rules["max_rms"]))
    if stats["leading_silence"] > rules["max_leading_silence"]:
        reasons.append("leading silence %.2fs above %.2fs"
                       % (stats["leading_silence"], rules["max_leading_silence"]))
    if stats["trailing_silence"] > rules["max_trailing_silence"]:
        reasons.append("trailing silence %.2fs above %.2fs"
                       % (stats["trailing_silence"], rules["max_trailing_silence"]))

    return (not reasons), reasons


def trim_silence(signal, sample_rate=TARGET_SAMPLE_RATE, threshold_db=-45.0,
                 keep_margin=0.15):
    """Trim excessive lead-in/lead-out, keeping a small margin.

    Trimming is a repair, not a rejection: excessive edge silence is a common
    and harmless TTS artifact, and discarding those utterances would bias the
    corpus toward whatever the engine happens to render tightly.
    """
    if len(signal) == 0:
        return signal, 0.0, 0.0
    silent = _silence_mask(signal, threshold_db)
    voiced = np.flatnonzero(~silent)
    if not len(voiced):
        return signal, 0.0, 0.0
    margin = int(keep_margin * sample_rate)
    start = max(0, voiced[0] - margin)
    end = min(len(signal), voiced[-1] + margin + 1)
    return (signal[start:end],
            start / float(sample_rate),
            (len(signal) - end) / float(sample_rate))


def process_file(path, out_path=None, thresholds=None, do_trim=True):
    """Load, optionally trim, measure and judge one file."""
    try:
        signal, sample_rate = load_audio(path)
    except Exception as exc:
        return {
            "audio": path, "passed": False,
            "reasons": ["unreadable: %s" % exc], "stats": None,
            "trimmed_leading": 0.0, "trimmed_trailing": 0.0,
        }

    trimmed_lead = trimmed_tail = 0.0
    if do_trim:
        signal, trimmed_lead, trimmed_tail = trim_silence(signal, sample_rate)

    stats = measure(signal, sample_rate)
    passed, reasons = evaluate(stats, thresholds)

    if passed and out_path:
        save_audio(out_path, signal, sample_rate)

    return {
        "audio": path, "out_audio": out_path if passed else None,
        "passed": passed, "reasons": reasons, "stats": stats,
        "trimmed_leading": round(trimmed_lead, 4),
        "trimmed_trailing": round(trimmed_tail, 4),
    }


def summarize(results):
    """Aggregate report, including the exact reason distribution."""
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]
    reason_counts = Counter()
    for result in failed:
        for reason in result["reasons"]:
            # Bucket by rule, not by the formatted numbers.
            reason_counts[reason.split(" above")[0].split(" below")[0]] += 1

    durations = [r["stats"]["duration"] for r in passed if r["stats"]]
    total = sum(durations)
    report = OrderedDict([
        ("n_files", len(results)),
        ("n_passed", len(passed)),
        ("n_failed", len(failed)),
        ("removal_rate", round(len(failed) / len(results), 5) if results else 0.0),
        ("failure_reasons", dict(reason_counts.most_common())),
        ("total_duration_seconds", round(total, 2)),
        ("total_duration_hours", round(total / 3600.0, 4)),
    ])
    if durations:
        ordered = sorted(durations)
        report["duration_stats"] = {
            "min": round(ordered[0], 3), "max": round(ordered[-1], 3),
            "mean": round(total / len(ordered), 3),
            "median": round(ordered[len(ordered) // 2], 3),
            "p10": round(ordered[int(0.1 * len(ordered))], 3),
            "p90": round(ordered[int(0.9 * len(ordered))], 3),
        }
    return report


def independent_asr_check(results, transcripts, asr_fn, cer_threshold=0.60):
    """Catastrophic-failure check using an ASR system other than the target.

    ``asr_fn(paths) -> list[str]`` must be a *different* model from the one under
    study. The threshold is intentionally loose (default CER > 0.60): the goal is
    to catch renders that are empty, in the wrong language or truncated, not to
    rank utterances by difficulty. Using the target model here, or tightening
    this threshold, would introduce target-model selection bias.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluation import metrics as M
    from evaluation import normalization as norm

    paths = [r["audio"] for r in results]
    hypotheses = asr_fn(paths)

    flagged = []
    for result, reference, hypothesis in zip(results, transcripts, hypotheses):
        ref = norm.normalize(reference)
        hyp = norm.normalize(hypothesis)
        record = M.utterance_cer(ref, hyp)
        if record is None:
            continue
        if record["cer"] > cer_threshold:
            flagged.append({
                "audio": result["audio"],
                "reference_normalized": ref,
                "independent_hypothesis_normalized": hyp,
                "cer": round(record["cer"], 4),
                "reason": "catastrophic TTS failure suspected "
                          "(independent-ASR CER %.3f > %.2f)"
                          % (record["cer"], cer_threshold),
            })
    return {
        "checker": getattr(asr_fn, "__name__", "unknown"),
        "cer_threshold": cer_threshold,
        "n_checked": len(results),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "note": "loose threshold by design; the target model is never used here",
    }
