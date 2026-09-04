"""Phase 2 - JSONL manifest standard and conversion to the official SFT format.

One manifest line describes one utterance. Every field that a controlled
comparison depends on (which script it came from, which voice said it, which
acoustic condition was applied) lives in the manifest, so that a split, a
subset or a per-condition breakdown can always be reconstructed from disk.

The training-time format expected by the official Alibaba script
(``finetuning/qwen3_asr_sft.py``) is much narrower - it wants only ``audio`` and
``text``, with the language prefix baked into ``text``:

    {"audio": "/path/utt.wav", "text": "language Chinese<asr_text>转写内容"}

``to_qwen_sft_format`` performs that projection. The rich manifest remains the
source of truth; the SFT file is a derived artifact and is regenerated, never
hand-edited.
"""
from __future__ import annotations

import json
import os
from collections import Counter, OrderedDict

# (name, python type, required)
SCHEMA = [
    ("utt_id", str, True),            # globally unique, stable across regeneration
    ("audio", str, True),             # path to 16 kHz mono WAV
    ("text", str, True),              # ground-truth transcript, unnormalized
    ("speaker_id", str, True),        # synthetic voice identity or human speaker
    ("source", str, True),            # synthetic_qwen3tts | synthetic_cosyvoice3 | human | public
    ("tts_engine", (str, type(None)), True),   # engine+version, None for human speech
    ("domain_category", str, True),   # chief_complaint, examination, ...
    ("template_family", str, True),   # semantic family, the unit of script-disjoint splits
    ("duration", (int, float), True), # seconds, measured from the actual audio
    ("condition", str, True),         # clean | noise | reverb | competing_speech | codec | human_*
    ("snr", (int, float, type(None)), True),   # dB for additive noise, else None
    ("sir", (int, float, type(None)), True),   # dB for competing speech, else None
]

REQUIRED_FIELDS = [name for name, _, required in SCHEMA if required]

VALID_SOURCES = {"synthetic_qwen3tts", "synthetic_cosyvoice3", "human", "public"}
VALID_CONDITIONS = {
    "clean", "noise", "reverb", "noise_reverb", "competing_speech", "codec",
    "human_quiet", "human_farfield", "human_noisy", "public_clean", "public_noisy",
}

# Phase 3 target distribution. Used by the generator and by validation reports.
DOMAIN_CATEGORIES = OrderedDict([
    ("chief_complaint", 0.25),
    ("examination", 0.15),
    ("registration", 0.10),
    ("navigation", 0.10),
    ("disease", 0.10),
    ("medication", 0.10),
    ("numeric", 0.10),
    ("code_switch", 0.05),
    ("disfluency", 0.05),
])

LANGUAGE_PREFIX = "language Chinese<asr_text>"


def read_manifest(path):
    """Load a JSONL manifest, raising with a line number on malformed JSON."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("%s:%d malformed JSON: %s" % (path, lineno, exc)) from exc
    return rows


def write_manifest(rows, path):
    """Write JSONL with stable key order so diffs between runs are readable."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    order = [name for name, _, _ in SCHEMA]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            ordered = OrderedDict()
            for key in order:
                if key in row:
                    ordered[key] = row[key]
            for key in sorted(k for k in row if k not in ordered):
                ordered[key] = row[key]
            handle.write(json.dumps(ordered, ensure_ascii=False) + "\n")
    return path


def validate_manifest(rows, check_audio_exists=False):
    """Structural validation. Returns a report; callers decide how to react.

    Checks: required fields present and correctly typed, utt_id uniqueness,
    controlled-vocabulary fields, SNR/SIR consistency with the declared
    condition, and (optionally) that every referenced audio file exists.
    """
    errors, warnings = [], []
    seen_ids = set()
    duplicate_ids = set()

    for idx, row in enumerate(rows):
        tag = row.get("utt_id", "<row %d>" % idx)

        for name, expected_type, required in SCHEMA:
            if name not in row:
                if required:
                    errors.append("%s: missing required field '%s'" % (tag, name))
                continue
            if not isinstance(row[name], expected_type):
                errors.append("%s: field '%s' has type %s, expected %s"
                              % (tag, name, type(row[name]).__name__, expected_type))

        utt_id = row.get("utt_id")
        if utt_id in seen_ids:
            duplicate_ids.add(utt_id)
        seen_ids.add(utt_id)

        if row.get("source") not in VALID_SOURCES:
            errors.append("%s: unknown source '%s'" % (tag, row.get("source")))
        if row.get("condition") not in VALID_CONDITIONS:
            errors.append("%s: unknown condition '%s'" % (tag, row.get("condition")))
        # "general" is the label for public general-domain corpora such as
        # AISHELL-1, which deliberately sit outside the hospital taxonomy.
        if (row.get("domain_category") not in DOMAIN_CATEGORIES
                and not (row.get("domain_category") == "general"
                         and row.get("source") == "public")):
            warnings.append("%s: domain_category '%s' is outside the Phase 3 taxonomy"
                            % (tag, row.get("domain_category")))

        condition = row.get("condition")
        if condition in ("noise", "noise_reverb") and row.get("snr") is None:
            errors.append("%s: condition '%s' requires an snr value" % (tag, condition))
        if condition == "competing_speech" and row.get("sir") is None:
            errors.append("%s: condition 'competing_speech' requires an sir value" % tag)
        if condition == "clean" and (row.get("snr") is not None or row.get("sir") is not None):
            errors.append("%s: condition 'clean' must have snr=null and sir=null" % tag)

        duration = row.get("duration")
        if isinstance(duration, (int, float)):
            if duration <= 0:
                errors.append("%s: non-positive duration %s" % (tag, duration))
            elif duration > 30:
                warnings.append("%s: unusually long utterance (%.1fs)" % (tag, duration))

        if not str(row.get("text", "")).strip():
            errors.append("%s: empty text" % tag)

        if check_audio_exists and row.get("audio") and not os.path.exists(row["audio"]):
            errors.append("%s: audio file not found: %s" % (tag, row["audio"]))

    for utt_id in sorted(duplicate_ids):
        errors.append("duplicate utt_id: %s" % utt_id)

    total_duration = sum(r.get("duration", 0) or 0 for r in rows)
    return {
        "n_rows": len(rows),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "errors": errors,
        "warnings": warnings,
        "total_duration_seconds": total_duration,
        "total_duration_hours": total_duration / 3600.0,
        "by_source": dict(Counter(r.get("source") for r in rows)),
        "by_condition": dict(Counter(r.get("condition") for r in rows)),
        "by_domain_category": dict(Counter(r.get("domain_category") for r in rows)),
        "n_speakers": len({r.get("speaker_id") for r in rows}),
        "n_template_families": len({r.get("template_family") for r in rows}),
    }


def format_report(report, title="manifest"):
    """Render a validation report for a log file."""
    lines = ["== %s ==" % title,
             "rows              : %d" % report["n_rows"],
             "total duration    : %.2f h (%.0f s)" % (report["total_duration_hours"],
                                                      report["total_duration_seconds"]),
             "speakers          : %d" % report["n_speakers"],
             "template families : %d" % report["n_template_families"],
             "errors            : %d" % report["n_errors"],
             "warnings          : %d" % report["n_warnings"],
             "by source         : %s" % report["by_source"],
             "by condition      : %s" % report["by_condition"],
             "by domain category: %s" % report["by_domain_category"]]
    for err in report["errors"][:50]:
        lines.append("  ERROR   %s" % err)
    if report["n_errors"] > 50:
        lines.append("  ... %d more errors" % (report["n_errors"] - 50))
    for warn in report["warnings"][:20]:
        lines.append("  WARNING %s" % warn)
    if report["n_warnings"] > 20:
        lines.append("  ... %d more warnings" % (report["n_warnings"] - 20))
    return "\n".join(lines)


def to_qwen_sft_format(rows, path, language="Chinese"):
    """Project a manifest onto the official Qwen3-ASR fine-tuning JSONL.

    The official loader reads only ``audio`` and ``text``; ``text`` must carry
    the language prefix. ``utt_id`` is preserved as an extra key so a training
    file can still be traced back to the manifest - the official script ignores
    unknown keys.
    """
    prefix = "language %s<asr_text>" % language
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            text = row["text"]
            if "<asr_text>" not in text:
                text = prefix + text
            handle.write(json.dumps(
                {"audio": row["audio"], "text": text, "utt_id": row["utt_id"]},
                ensure_ascii=False) + "\n")
            written += 1
    return {"path": path, "n_written": written, "language_prefix": prefix}


def duration_hours(rows):
    return sum(r.get("duration", 0) or 0 for r in rows) / 3600.0
