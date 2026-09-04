#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 20 - Generate CSV and LaTeX tables from experiment outputs.

Seven tables:
    table1_component_ablation
    table2_data_budget
    table3_robustness
    table4_terminology
    table5_forgetting
    table6_model_scale
    table7_parameter_efficiency

Every cell is read from a file under ``experiments/``. A cell with no
corresponding file is rendered ``XX``. No number is ever invented, interpolated
or carried over from a similar run - if a table is full of XX, the experiments
have not been run yet, and that is the honest state.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from evaluation.collect import (ARM_LABEL, collect_experiments,  # noqa: E402
                                fmt_int, fmt_pct, index_by)

LATEX_ESCAPE = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}


def escape_latex(text):
    out = str(text)
    for char, replacement in LATEX_ESCAPE.items():
        out = out.replace(char, replacement)
    return out


class Table:
    def __init__(self, key, title, columns, caption=None):
        self.key = key
        self.title = title
        self.columns = columns
        self.caption = caption or title
        self.rows = []

    def add(self, *values):
        if len(values) != len(self.columns):
            raise ValueError("%s: expected %d values, got %d"
                             % (self.key, len(self.columns), len(values)))
        self.rows.append(list(values))

    def write_csv(self, outdir):
        path = Path(outdir) / ("%s.csv" % self.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(self.columns)
            writer.writerows(self.rows)
        return path

    def write_json(self, outdir):
        """JSON twin of the CSV.

        The Windows authoring machine runs a DLP agent that encrypts .csv files
        on write, so a CSV read back there is unusable. Tables generated on the
        Linux training host are unaffected, but the JSON copy makes every table
        machine-readable from either machine.
        """
        import json as _json

        path = Path(outdir) / ("%s.json" % self.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"key": self.key, "title": self.title,
                   "caption": self.caption, "columns": self.columns,
                   "rows": self.rows,
                   "n_missing_cells": self.n_missing(),
                   "n_cells": self.n_cells()}
        path.write_text(_json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        return path

    def write_latex(self, outdir):
        path = Path(outdir) / ("%s.tex" % self.key)
        align = "l" + "r" * (len(self.columns) - 1)
        lines = [r"\begin{table}[t]", r"\centering",
                 r"\caption{%s}" % escape_latex(self.caption),
                 r"\label{tab:%s}" % self.key,
                 r"\begin{tabular}{%s}" % align, r"\toprule",
                 " & ".join(escape_latex(c) for c in self.columns) + r" \\",
                 r"\midrule"]
        for row in self.rows:
            lines.append(" & ".join(escape_latex(v) for v in row) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def n_missing(self):
        return sum(1 for row in self.rows for value in row if value == "XX")

    def n_cells(self):
        return sum(len(row) for row in self.rows)


def get(index, key):
    return index.get(key)


def value(record, field):
    return record.get(field) if record else None


def table_component_ablation(records):
    """Table 1: all arms at the 20 h budget on 0.6B."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    table = Table(
        "table1_component_ablation",
        "Component ablation",
        ["Adaptation", "Trainable params", "% of model", "CER synth (%)",
         "CER cross-TTS (%)", "MTER (%)", "Entity recall (%)"],
        "Component-wise adaptation of Qwen3-ASR-0.6B at the 20 h budget. "
        "CER is character error rate; MTER is medical-term error rate. "
        "XX marks an experiment that has not been run.")

    for arm_key, label in ARM_LABEL.items():
        budget = None if arm_key == "zero" else "20h"
        synth = get(index, ("qwen06", arm_key, budget, False, False, 42, "test_synthetic"))
        cross = get(index, ("qwen06", arm_key, budget, False, False, 42, "test_cross_tts"))
        table.add(label,
                  fmt_int(value(synth, "trainable_parameters")),
                  fmt_pct(value(synth, "trainable_percentage") and
                          value(synth, "trainable_percentage") / 100.0, 4),
                  fmt_pct(value(synth, "cer")),
                  fmt_pct(value(cross, "cer")),
                  fmt_pct(value(synth, "medical_term_error_rate")),
                  fmt_pct(value(synth, "medical_entity_recall")))
    return table


def table_data_budget(records):
    """Table 2: CER against training hours for the three arms of interest."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    table = Table(
        "table2_data_budget",
        "Data budget",
        ["Adaptation", "1 h", "5 h", "10 h", "20 h"],
        "CER (%) on the synthetic held-out test set against target-domain "
        "training hours. Subsets are nested: D1 subset of D5 subset of D10 "
        "subset of D20.")

    zero = get(index, ("qwen06", "zero", None, False, False, 42, "test_synthetic"))
    zero_cer = fmt_pct(value(zero, "cer"))
    table.add("A0 Zero-shot", zero_cer, zero_cer, zero_cer, zero_cer)

    for arm_key in ("full", "textlora", "dualpeft"):
        cells = []
        for budget in ("1h", "5h", "10h", "20h"):
            record = get(index, ("qwen06", arm_key, budget, False, False, 42,
                                 "test_synthetic"))
            cells.append(fmt_pct(value(record, "cer")))
        table.add(ARM_LABEL[arm_key], *cells)
    return table


def table_robustness(records):
    """Table 3: augmented vs non-augmented DualPEFT across conditions."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    test_sets = ["test_synthetic", "test_cross_tts", "test_human_quiet",
                 "test_human_farfield", "test_human_noisy", "test_public_noisy"]
    labels = ["Synthetic clean", "Cross-TTS", "Human quiet", "Human far-field",
              "Human noisy", "Public noisy"]

    table = Table(
        "table3_robustness",
        "Robustness to acoustic condition",
        ["Test set", "Zero-shot", "DualPEFT 20 h", "DualPEFT 20 h + aug"],
        "CER (%) by test condition. The augmentation comparison holds arm, "
        "budget and seed fixed, so augmentation is the only variable.")

    for test_set, label in zip(test_sets, labels):
        zero = get(index, ("qwen06", "zero", None, False, False, 42, test_set))
        plain = get(index, ("qwen06", "dualpeft", "20h", False, False, 42, test_set))
        aug = get(index, ("qwen06", "dualpeft", "20h", True, False, 42, test_set))
        table.add(label, fmt_pct(value(zero, "cer")),
                  fmt_pct(value(plain, "cer")), fmt_pct(value(aug, "cer")))
    return table


def table_terminology(records):
    """Table 4: terminology error by term category."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    arms = ["zero", "full", "textlora", "dualpeft"]
    categories = ["imaging_exam", "lab_test", "disease", "medication",
                  "department", "symptom", "abbreviation"]

    table = Table(
        "table4_terminology",
        "Medical terminology error by category",
        ["Term category"] + [ARM_LABEL[a] for a in arms],
        "Medical-term error rate (%) by term category on the synthetic "
        "held-out test set. Term inventory is fixed before any model is run.")

    for category in categories:
        cells = []
        for arm_key in arms:
            budget = None if arm_key == "zero" else "20h"
            record = get(index, ("qwen06", arm_key, budget, False, False, 42,
                                 "test_synthetic"))
            by_category = (record or {}).get("by_term_category") or {}
            entry = by_category.get(category)
            cells.append(fmt_pct(entry.get("term_error_rate") if entry else None))
        table.add(category, *cells)
    return table


def table_forgetting(records):
    """Table 5: hospital gain against general-domain degradation."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    table = Table(
        "table5_forgetting",
        "Catastrophic forgetting",
        ["Model", "Hospital CER (%)", "AISHELL-1 CER (%)", "AISHELL-1 delta (pp)"],
        "Hospital-domain gain against general-domain degradation. The delta is "
        "measured against the zero-shot AISHELL-1 CER of the same base model.")

    zero_general = get(index, ("qwen06", "zero", None, False, False, 42, "test_aishell1"))
    baseline = value(zero_general, "cer")

    rows = [("A0 Zero-shot", "zero", None, False, False),
            ("A1 Full SFT 20 h", "full", "20h", False, False),
            ("A4 Text-LoRA 20 h", "textlora", "20h", False, False),
            ("A7 DualPEFT 20 h", "dualpeft", "20h", False, False),
            ("A7 DualPEFT 20 h + replay", "dualpeft", "20h", False, True)]

    for label, arm_key, budget, aug, replay in rows:
        hospital = get(index, ("qwen06", arm_key, budget, aug, replay, 42, "test_synthetic"))
        general = get(index, ("qwen06", arm_key, budget, aug, replay, 42, "test_aishell1"))
        general_cer = value(general, "cer")
        delta = (None if (general_cer is None or baseline is None)
                 else general_cer - baseline)
        table.add(label, fmt_pct(value(hospital, "cer")), fmt_pct(general_cer),
                  fmt_pct(delta) if delta is not None else "XX")
    return table


def table_model_scale(records):
    """Table 6: 0.6B against 1.7B."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    table = Table(
        "table6_model_scale",
        "Model scale transfer",
        ["Model", "Adaptation", "CER synth (%)", "CER cross-TTS (%)",
         "MTER (%)", "Trainable params"],
        "Scale transfer. The full grid is deliberately not repeated at 1.7B.")

    for model_key in ("qwen06", "qwen17"):
        for arm_key, budget in (("zero", None), ("dualpeft", "20h"), ("full", "20h")):
            synth = get(index, (model_key, arm_key, budget, False, False, 42, "test_synthetic"))
            cross = get(index, (model_key, arm_key, budget, False, False, 42, "test_cross_tts"))
            table.add(
                {"qwen06": "Qwen3-ASR-0.6B", "qwen17": "Qwen3-ASR-1.7B"}[model_key],
                ARM_LABEL[arm_key] + (" %s" % budget if budget else ""),
                fmt_pct(value(synth, "cer")), fmt_pct(value(cross, "cer")),
                fmt_pct(value(synth, "medical_term_error_rate")),
                fmt_int(value(synth, "trainable_parameters")))
    return table


def table_parameter_efficiency(records):
    """Table 7: cost per unit of benefit."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    table = Table(
        "table7_parameter_efficiency",
        "Parameter efficiency",
        ["Adaptation", "Trainable params", "% of model", "Peak VRAM (GiB)",
         "GPU hours", "CER (%)", "CER reduction vs zero-shot (pp)"],
        "Cost against benefit at the 20 h budget on Qwen3-ASR-0.6B.")

    zero = get(index, ("qwen06", "zero", None, False, False, 42, "test_synthetic"))
    baseline = value(zero, "cer")

    for arm_key, label in ARM_LABEL.items():
        budget = None if arm_key == "zero" else "20h"
        record = get(index, ("qwen06", arm_key, budget, False, False, 42, "test_synthetic"))
        cer = value(record, "cer")
        reduction = (None if (cer is None or baseline is None) else baseline - cer)
        percentage = value(record, "trainable_percentage")
        table.add(label,
                  fmt_int(value(record, "trainable_parameters")),
                  fmt_pct(percentage / 100.0 if percentage is not None else None, 4),
                  fmt_pct(value(record, "peak_vram_gib") and
                          value(record, "peak_vram_gib") / 100.0, 2),
                  fmt_pct(value(record, "gpu_hours") and
                          value(record, "gpu_hours") / 100.0, 2),
                  fmt_pct(cer),
                  fmt_pct(reduction) if reduction is not None else "XX")
    return table


BUILDERS = [table_component_ablation, table_data_budget, table_robustness,
            table_terminology, table_forgetting, table_model_scale,
            table_parameter_efficiency]


def main():
    ap = argparse.ArgumentParser(description="Generate result tables.")
    ap.add_argument("--experiments", default=str(_ROOT / "experiments"))
    ap.add_argument("--outdir", default=str(_ROOT / "results" / "tables"))
    args = ap.parse_args()

    records = collect_experiments(args.experiments)
    print("collected %d experiment/test-set records from %s"
          % (len(records), args.experiments))
    if records:
        experiments = sorted({r["experiment"] for r in records})
        print("experiments found: %s" % ", ".join(experiments))

    outdir = Path(args.outdir)
    summary = OrderedDict()
    for builder in BUILDERS:
        table = builder(records)
        table.write_csv(outdir)
        table.write_json(outdir)
        table.write_latex(outdir)
        missing = table.n_missing()
        summary[table.key] = (missing, table.n_cells())
        print("  %-32s %2d rows, %3d/%3d cells missing (XX)"
              % (table.key, len(table.rows), missing, table.n_cells()))

    total_missing = sum(m for m, _ in summary.values())
    total_cells = sum(c for _, c in summary.values())
    print("")
    print("%d/%d cells are XX (no experiment on disk yet)"
          % (total_missing, total_cells))
    print("wrote CSV and LaTeX to %s" % outdir)


if __name__ == "__main__":
    main()
