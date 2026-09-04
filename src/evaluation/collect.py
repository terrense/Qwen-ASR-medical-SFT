# -*- coding: utf-8 -*-
"""Phase 19/20 - Collect experiment results into a single in-memory table.

Reads `experiments/*/` directories written by the training and evaluation
entry points and assembles one record per (experiment, test set).

The rule this module exists to enforce: **a number appears only if a file on
disk contains it.** Anything absent is reported as ``None`` and rendered as
``XX`` downstream. Nothing is interpolated, estimated, or carried over from a
similar run.
"""
from __future__ import annotations

import json
import re
from collections import OrderedDict
from pathlib import Path

MISSING = None

# Experiment directory names look like qwen06_dualpeft_20h_aug[_replay][_seedN]
NAME_RE = re.compile(
    r"^(?P<model>qwen\d+)"
    r"_(?P<arm>zero|full|audiolora_proj|audiolora|proj|textlora_proj|textlora|dualpeft)"
    r"(?:_(?P<budget>\d+h))?"
    r"(?P<aug>_aug)?(?P<replay>_replay)?"
    r"(?:_seed(?P<seed>\d+))?$")

ARM_LABEL = OrderedDict([
    ("zero", "A0 Zero-shot"),
    ("full", "A1 Full SFT"),
    ("audiolora", "A2 Audio-LoRA"),
    ("proj", "A3 Projection only"),
    ("textlora", "A4 Text-LoRA"),
    ("audiolora_proj", "A5 Audio-LoRA + Proj"),
    ("textlora_proj", "A6 Text-LoRA + Proj"),
    ("dualpeft", "A7 DualPEFT"),
])

MODEL_LABEL = {"qwen06": "Qwen3-ASR-0.6B", "qwen17": "Qwen3-ASR-1.7B"}

TEST_SET_LABEL = OrderedDict([
    ("test_synthetic", "Synthetic clean (held-out voices)"),
    ("test_synthetic_aug", "Synthetic augmented"),
    ("test_cross_tts", "Cross-TTS (CosyVoice3)"),
    ("test_human_quiet", "Human quiet"),
    ("test_human_farfield", "Human far-field"),
    ("test_human_noisy", "Human noisy"),
    ("test_aishell1", "AISHELL-1 (general)"),
    ("test_public_noisy", "Public noisy Mandarin"),
])


def parse_experiment_name(name):
    match = NAME_RE.match(name)
    if not match:
        return None
    parts = match.groupdict()
    return {
        "model_key": parts["model"],
        "model": MODEL_LABEL.get(parts["model"], parts["model"]),
        "arm_key": parts["arm"],
        "arm": ARM_LABEL.get(parts["arm"], parts["arm"]),
        "budget": parts["budget"] or MISSING,
        "augmented": bool(parts["aug"]),
        "replay": bool(parts["replay"]),
        "seed": int(parts["seed"]) if parts["seed"] else 42,
    }


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def collect_experiments(experiments_dir):
    """One record per (experiment, test set) found on disk."""
    root = Path(experiments_dir)
    records = []
    if not root.exists():
        return records

    for exp_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = parse_experiment_name(exp_dir.name)
        if meta is None:
            continue

        train_metrics = _read_json(exp_dir / "metrics.json") or {}
        trainable = _read_json(exp_dir / "trainable_parameters.json") or {}

        training = {
            "experiment": exp_dir.name,
            "trainable_parameters": train_metrics.get("trainable_parameters",
                                                      trainable.get("trainable_parameters", MISSING)),
            "total_parameters": train_metrics.get("total_parameters",
                                                  trainable.get("total_parameters", MISSING)),
            "trainable_percentage": train_metrics.get("trainable_percentage",
                                                      trainable.get("trainable_percentage", MISSING)),
            "peak_vram_gib": train_metrics.get("peak_vram_gib", MISSING),
            "gpu_hours": train_metrics.get("gpu_hours", MISSING),
            "train_hours_audio": train_metrics.get("train_hours_audio", MISSING),
        }
        training.update(meta)

        found_any = False
        for eval_dir in sorted(p for p in exp_dir.iterdir() if p.is_dir()):
            if eval_dir.name == "checkpoints":
                continue
            eval_metrics = _read_json(eval_dir / "metrics.json")
            if not eval_metrics or "overall" not in eval_metrics:
                continue
            found_any = True
            overall = eval_metrics["overall"]
            terms = eval_metrics.get("medical_terms", {})
            record = dict(training)
            record.update({
                "test_set": eval_dir.name,
                "test_set_label": TEST_SET_LABEL.get(eval_dir.name, eval_dir.name),
                "cer": overall.get("cer", MISSING),
                "macro_cer": overall.get("macro_cer", MISSING),
                "n_utterances": overall.get("n_utterances", MISSING),
                "substitutions": overall.get("substitutions", MISSING),
                "deletions": overall.get("deletions", MISSING),
                "insertions": overall.get("insertions", MISSING),
                "medical_term_error_rate": terms.get("medical_term_error_rate", MISSING),
                "medical_entity_recall": terms.get("medical_entity_recall", MISSING),
                "utterance_exact_match": terms.get("utterance_exact_match", MISSING),
                "by_condition": eval_metrics.get("by_condition", {}),
                "by_domain_category": eval_metrics.get("by_domain_category", {}),
                "by_term_category": terms.get("by_term_category", {}),
                "predictions_path": str(eval_dir / "predictions.jsonl"),
                "real_time_factor": (eval_metrics.get("meta") or {}).get("real_time_factor", MISSING),
            })
            records.append(record)

        if not found_any:
            record = dict(training)
            record.update({"test_set": MISSING, "test_set_label": MISSING,
                           "cer": MISSING, "medical_term_error_rate": MISSING})
            records.append(record)

    return records


def index_by(records, *keys):
    """Index records by a tuple of fields for quick table lookup."""
    out = {}
    for record in records:
        out[tuple(record.get(k) for k in keys)] = record
    return out


def fmt(value, spec="%.2f", scale=1.0, missing="XX"):
    """Format a number, or the missing marker when it is absent.

    Every table cell goes through this. A missing experiment can therefore never
    silently become a zero or an inherited value.
    """
    if value is None:
        return missing
    try:
        return spec % (value * scale)
    except (TypeError, ValueError):
        return missing


def fmt_pct(value, decimals=2, missing="XX"):
    return fmt(value, "%." + str(decimals) + "f", 100.0, missing)


def fmt_int(value, missing="XX"):
    if value is None:
        return missing
    try:
        return format(int(value), ",")
    except (TypeError, ValueError):
        return missing
