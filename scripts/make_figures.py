#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 19 - Generate publication figures from experiment outputs.

Figures:
    figure1_component_diagram   which arm adapts which component (from the real
                                module inventory, not a sketch)
    figure2_data_pipeline       data generation flow with real corpus counts
    figure3_cer_vs_hours        CER against target-domain training hours
    figure4_cer_by_condition    CER by acoustic condition
    figure5_terminology         medical-term error by term category
    figure6_forgetting          hospital CER against general-domain CER

**No figure is ever drawn from invented numbers.** A figure whose inputs are not
on disk is skipped, and the script says exactly which file was missing. An empty
`results/figures/` therefore means "these experiments have not run", never
"these experiments produced nothing".

All figure text is English so the output does not depend on a CJK font being
installed on the training host.

Colors come from a CVD-validated categorical palette, assigned in fixed slot
order and never cycled. Series are direct-labeled as well as legended, so
identity never rests on color alone.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from evaluation.collect import ARM_LABEL, collect_experiments, index_by  # noqa: E402

# Validated categorical palette, light surface. Fixed order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#1a1a19"
INK_SECONDARY = "#5c5b55"
GRID = "#e3e2dd"
SURFACE = "#fcfcfb"

COMPONENT_COLOR = {"AUDIO_ENCODER": SERIES[0], "AUDIO_PROJECTION": SERIES[1],
                   "TEXT_DECODER": SERIES[2]}


def style_axes(ax, xlabel=None, ylabel=None, title=None):
    """Recessive grid and axes; ink-colored text; no chartjunk."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def save(fig, outdir, name):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(outdir / ("%s.%s" % (name, extension)), dpi=200,
                    bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("  wrote %s.{png,pdf}" % (outdir / name))


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Figure 1 - component adaptation diagram
# ---------------------------------------------------------------------------

def figure1_component_diagram(outdir, results_dir):
    """Which arm touches which component, with real parameter counts."""
    inventory = _read_json(Path(results_dir) / "lora_target_modules.json")
    if not inventory:
        print("  SKIP figure1: results/lora_target_modules.json missing "
              "(run scripts/inspect_model.py first)")
        return False

    summary_path = Path(results_dir) / "model_component_summary.json"
    summary = _read_json(summary_path)
    counts = {}
    if summary:
        for row in summary:
            counts[row["component"]] = row.get("linear_params")

    roots = inventory["roots"]
    fig, (ax_arch, ax_grid) = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.05, 1.0]})

    # --- left: the architecture, one box per component --------------------
    ax_arch.set_xlim(0, 10)
    ax_arch.set_ylim(0, 10)
    ax_arch.axis("off")
    ax_arch.set_title("Qwen3-ASR components", color=INK, fontsize=12,
                      loc="left", pad=12)

    blocks = [
        ("AUDIO_ENCODER", "Audio encoder body\n%s.layers.0-17\nq/k/v/out_proj, fc1, fc2"
         % roots["audio_root"].split(".")[-1], 6.9),
        ("AUDIO_PROJECTION", "Audio projection head\n%s\n%d → %d"
         % (", ".join(p.split(".")[-1] for p in roots["projection_paths"]),
            roots["d_model"], roots["output_dim"]), 4.3),
        ("TEXT_DECODER", "Language decoder\n%s.layers.0-27\nq/k/v/o_proj, MLP, lm_head"
         % roots["text_root"].split(".")[-1], 1.7),
    ]
    for component, label, y in blocks:
        ax_arch.add_patch(FancyBboxPatch(
            (0.6, y), 8.8, 1.9, boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor=COMPONENT_COLOR[component], edgecolor="none", alpha=0.16,
            zorder=1))
        ax_arch.add_patch(FancyBboxPatch(
            (0.6, y), 8.8, 1.9, boxstyle="round,pad=0.08,rounding_size=0.18",
            facecolor="none", edgecolor=COMPONENT_COLOR[component],
            linewidth=1.6, zorder=2))
        ax_arch.text(1.0, y + 1.32, label.split("\n")[0], color=INK,
                     fontsize=10.5, fontweight="bold", va="center")
        ax_arch.text(1.0, y + 0.72, "\n".join(label.split("\n")[1:]),
                     color=INK_SECONDARY, fontsize=8.4, va="center",
                     linespacing=1.5)
        if counts.get(component):
            ax_arch.text(9.1, y + 1.32, "%.1fM" % (counts[component] / 1e6),
                         color=COMPONENT_COLOR[component], fontsize=10.5,
                         fontweight="bold", va="center", ha="right")
            ax_arch.text(9.1, y + 0.78, "linear params", color=INK_SECONDARY,
                         fontsize=7.6, va="center", ha="right")

    for y_from, y_to in ((6.85, 6.25), (4.25, 3.65)):
        ax_arch.add_patch(FancyArrowPatch(
            (5.0, y_from), (5.0, y_to), arrowstyle="-|>", mutation_scale=13,
            color=INK_SECONDARY, linewidth=1.3))
    ax_arch.text(5.0, 9.2, "audio in", color=INK_SECONDARY, fontsize=9,
                 ha="center")
    ax_arch.add_patch(FancyArrowPatch((5.0, 9.0), (5.0, 8.85),
                                      arrowstyle="-|>", mutation_scale=13,
                                      color=INK_SECONDARY, linewidth=1.3))
    ax_arch.text(5.0, 1.1, "transcript out", color=INK_SECONDARY, fontsize=9,
                 ha="center")

    # --- right: arm x component matrix ------------------------------------
    from models.components import ARMS

    arms = [a for a in ARMS]
    components = ["AUDIO_ENCODER", "AUDIO_PROJECTION", "TEXT_DECODER"]
    labels = {"AUDIO_ENCODER": "Audio\nencoder",
              "AUDIO_PROJECTION": "Audio\nprojection",
              "TEXT_DECODER": "Language\ndecoder"}

    ax_grid.set_xlim(-0.6, len(components) - 0.4)
    ax_grid.set_ylim(-0.6, len(arms) - 0.4)
    ax_grid.invert_yaxis()
    ax_grid.set_xticks(range(len(components)))
    ax_grid.set_xticklabels([labels[c] for c in components], fontsize=9)
    ax_grid.set_yticks(range(len(arms)))
    ax_grid.set_yticklabels([ARM_LABEL.get(a.split("_", 1)[0].lower(), a)
                             if False else a.replace("_", " ")
                             for a in arms], fontsize=9)
    ax_grid.set_facecolor(SURFACE)
    for side in ("top", "right", "left", "bottom"):
        ax_grid.spines[side].set_visible(False)
    ax_grid.tick_params(colors=INK_SECONDARY, length=0)
    ax_grid.set_title("What each arm adapts", color=INK, fontsize=12,
                      loc="left", pad=12)

    for row, arm in enumerate(arms):
        spec = ARMS[arm]
        state = {
            "AUDIO_ENCODER": "full" if spec["full_sft"] else
                             ("lora" if spec["audio_lora"] else "frozen"),
            "AUDIO_PROJECTION": "full" if spec["train_projection"] else "frozen",
            "TEXT_DECODER": "full" if spec["full_sft"] else
                            ("lora" if spec["text_lora"] else "frozen"),
        }
        for col, component in enumerate(components):
            kind = state[component]
            color = COMPONENT_COLOR[component]
            if kind == "frozen":
                ax_grid.add_patch(plt.Rectangle(
                    (col - 0.42, row - 0.36), 0.84, 0.72, facecolor=GRID,
                    edgecolor="none", zorder=1))
                ax_grid.text(col, row, "frozen", ha="center", va="center",
                             fontsize=7.4, color=INK_SECONDARY, zorder=2)
            else:
                alpha = 0.85 if kind == "full" else 0.38
                ax_grid.add_patch(plt.Rectangle(
                    (col - 0.42, row - 0.36), 0.84, 0.72, facecolor=color,
                    alpha=alpha, edgecolor="none", zorder=1))
                ax_grid.text(col, row, "trained" if kind == "full" else "LoRA",
                             ha="center", va="center", fontsize=7.6,
                             color="white" if kind == "full" else INK,
                             fontweight="bold", zorder=2)

    fig.suptitle("Figure 1  Component-wise adaptation of Qwen3-ASR",
                 color=INK, fontsize=13.5, x=0.02, ha="left", y=1.02)
    save(fig, outdir, "figure1_component_diagram")
    return True


# ---------------------------------------------------------------------------
# Figure 2 - data generation pipeline
# ---------------------------------------------------------------------------

def figure2_data_pipeline(outdir, data_dir):
    """Pipeline flow annotated with the real corpus counts."""
    generation = _read_json(Path(data_dir) / "scripts" / "generation_report.json")
    split = _read_json(Path(data_dir) / "manifests" / "splits" / "split_report.json")
    if not generation or not split:
        print("  SKIP figure2: generation_report.json or split_report.json missing")
        return False

    fig, ax = plt.subplots(figsize=(13.5, 4.8))
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 34)
    ax.axis("off")

    # Colour groups the stages by kind rather than decorating them: the text
    # corpus stages share one hue, speech synthesis a second, audio processing
    # a third. Six arbitrary hues would encode nothing.
    text_hue, tts_hue, audio_hue = SERIES[0], SERIES[1], SERIES[2]
    stages = [
        ("Template families", ["%d semantic patterns" % generation["n_template_families"],
                               "9 domain categories"], text_hue),
        ("Utterance generation", ["%s unique" % format(generation["generated_total"], ","),
                                  "dedup + near-dup control"], text_hue),
        ("Family-disjoint split", ["train %s / dev %s" % (
            format(split["train"]["n_scripts"], ","),
            format(split["dev"]["n_scripts"], ",")),
            "test %s" % format(split["test"]["n_scripts"], ",")], text_hue),
        ("Qwen3-TTS synthesis", ["32 designed voices",
                                 "20 train / 6 dev / 6 test"], tts_hue),
        ("Quality control", ["16 kHz mono WAV", "QC gates + removal log"], audio_hue),
        ("Acoustic augmentation", ["4 condition classes",
                                   "40 / 30 / 15 / 15 %"], audio_hue),
    ]

    width, gap = 14.2, 2.6
    x = 1.8
    for title, body, color in stages:
        ax.add_patch(FancyBboxPatch(
            (x, 11.5), width, 11.5, boxstyle="round,pad=0.22,rounding_size=0.7",
            facecolor=color, alpha=0.13, edgecolor="none", zorder=1))
        ax.add_patch(FancyBboxPatch(
            (x, 11.5), width, 11.5, boxstyle="round,pad=0.22,rounding_size=0.7",
            facecolor="none", edgecolor=color, linewidth=1.4, zorder=2))
        ax.text(x + width / 2, 20.2, title, ha="center", va="center",
                fontsize=8.0, color=INK, fontweight="bold")
        for offset, line in enumerate(body):
            ax.text(x + width / 2, 17.0 - offset * 2.5, line, ha="center",
                    va="center", fontsize=7.0, color=INK_SECONDARY)
        if x + width + gap < 100:
            ax.add_patch(FancyArrowPatch(
                (x + width + 0.3, 17.2), (x + width + gap - 0.3, 17.2),
                arrowstyle="-|>", mutation_scale=11, color=INK_SECONDARY,
                linewidth=1.1))
        x += width + gap

    # Group captions under the boxes.
    ax.text(1.8 + (3 * width + 2 * gap) / 2, 8.6, "text corpus",
            ha="center", fontsize=8.0, color=text_hue, fontweight="bold")
    ax.text(1.8 + 3 * (width + gap) + width / 2, 8.6, "speech synthesis",
            ha="center", fontsize=8.0, color=tts_hue, fontweight="bold")
    ax.text(1.8 + 4 * (width + gap) + (2 * width + gap) / 2, 8.6,
            "audio processing", ha="center", fontsize=8.0, color=audio_hue,
            fontweight="bold")

    ax.text(1.8, 29.5, "Figure 2  Data generation pipeline", color=INK,
            fontsize=13.0, ha="left", va="center")
    ax.text(1.8, 26.0,
            "Cross-TTS test (%d scripts, CosyVoice3) branches from the held-out "
            "test families and is never trained on." % split["cross_tts"]["n_scripts"],
            color=INK_SECONDARY, fontsize=8.2, ha="left", va="center")
    ax.text(1.8, 4.6,
            "Splits are disjoint by template family, not by waveform. "
            "Speaker pools are disjoint across train / dev / test.",
            color=INK_SECONDARY, fontsize=8.2, ha="left", va="center")
    save(fig, outdir, "figure2_data_pipeline")
    return True


# ---------------------------------------------------------------------------
# Figures 3-6 - result figures
# ---------------------------------------------------------------------------

def figure3_cer_vs_hours(outdir, records):
    """CER against training hours. Line chart: change over a budget axis."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    budgets = ["1h", "5h", "10h", "20h"]
    x_values = [1, 5, 10, 20]

    series = []
    for slot, arm_key in enumerate(("full", "textlora", "dualpeft")):
        points = []
        for budget, x in zip(budgets, x_values):
            record = index.get(("qwen06", arm_key, budget, False, False, 42,
                                "test_synthetic"))
            if record and record.get("cer") is not None:
                points.append((x, 100.0 * record["cer"]))
        if points:
            series.append((ARM_LABEL[arm_key], points, SERIES[slot]))

    zero = index.get(("qwen06", "zero", None, False, False, 42, "test_synthetic"))
    zero_cer = 100.0 * zero["cer"] if zero and zero.get("cer") is not None else None

    if not series:
        print("  SKIP figure3: no data-budget runs found under experiments/")
        return False

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    style_axes(ax, "Target-domain training data (hours)", "CER (%)",
               "Figure 3  CER against target-domain training hours")

    if zero_cer is not None:
        ax.axhline(zero_cer, color=INK_SECONDARY, linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.text(20.3, zero_cer, "zero-shot %.1f%%" % zero_cer, fontsize=8.5,
                color=INK_SECONDARY, va="center")

    for label, points, color in series:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(xs, ys, color=color, linewidth=2.0, marker="o",
                markersize=5.5, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label, zorder=3)
        ax.text(xs[-1] + 0.5, ys[-1], label, color=color, fontsize=9,
                va="center", fontweight="bold")

    ax.set_xscale("log")
    ax.set_xticks(x_values)
    ax.set_xticklabels(["%d h" % v for v in x_values])
    ax.set_xlim(0.85, 34)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, outdir, "figure3_cer_vs_hours")
    return True


def figure4_cer_by_condition(outdir, records):
    """CER by acoustic condition. Grouped bars: magnitude across categories."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    systems = [
        ("Zero-shot", ("qwen06", "zero", None, False, False, 42, "test_synthetic")),
        ("DualPEFT 20 h", ("qwen06", "dualpeft", "20h", False, False, 42, "test_synthetic")),
        ("DualPEFT 20 h + aug", ("qwen06", "dualpeft", "20h", True, False, 42, "test_synthetic")),
    ]

    available = []
    for label, key in systems:
        record = index.get(key)
        by_condition = (record or {}).get("by_condition") or {}
        if by_condition:
            available.append((label, by_condition))
    if not available:
        print("  SKIP figure4: no by-condition metrics found under experiments/")
        return False

    conditions = []
    for _, by_condition in available:
        for name in by_condition:
            if name not in conditions:
                conditions.append(name)

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    style_axes(ax, None, "CER (%)", "Figure 4  CER by acoustic condition")

    n = len(available)
    span = 0.78
    bar_width = span / n
    for slot, (label, by_condition) in enumerate(available):
        xs, ys = [], []
        for position, condition in enumerate(conditions):
            entry = by_condition.get(condition)
            if entry and entry.get("cer") is not None:
                xs.append(position - span / 2 + bar_width * (slot + 0.5))
                ys.append(100.0 * entry["cer"])
        ax.bar(xs, ys, width=bar_width * 0.86, color=SERIES[slot],
               label=label, zorder=3, linewidth=0)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.25, "%.1f" % y, ha="center", va="bottom",
                    fontsize=7.6, color=INK_SECONDARY)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels([c.replace("_", "\n") for c in conditions], fontsize=8.6)
    legend = ax.legend(frameon=False, fontsize=9, ncol=len(available),
                       loc="upper left")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, outdir, "figure4_cer_by_condition")
    return True


def figure5_terminology(outdir, records):
    """Medical-term error by term category. Horizontal bars: many categories."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    systems = [
        ("Zero-shot", ("qwen06", "zero", None, False, False, 42, "test_synthetic")),
        ("Text-LoRA 20 h", ("qwen06", "textlora", "20h", False, False, 42, "test_synthetic")),
        ("DualPEFT 20 h", ("qwen06", "dualpeft", "20h", False, False, 42, "test_synthetic")),
    ]

    available = []
    for label, key in systems:
        record = index.get(key)
        by_category = (record or {}).get("by_term_category") or {}
        if by_category:
            available.append((label, by_category))
    if not available:
        print("  SKIP figure5: no per-term-category metrics found under experiments/")
        return False

    categories = []
    for _, by_category in available:
        for name in by_category:
            if name not in categories:
                categories.append(name)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    ax.set_xlabel("Medical-term error rate (%)", color=INK_SECONDARY, fontsize=10)
    ax.set_title("Figure 5  Medical terminology error by category", color=INK,
                 fontsize=12, loc="left", pad=12)

    n = len(available)
    span = 0.78
    bar_height = span / n
    for slot, (label, by_category) in enumerate(available):
        ys, xs = [], []
        for position, category in enumerate(categories):
            entry = by_category.get(category)
            if entry and entry.get("term_error_rate") is not None:
                ys.append(position - span / 2 + bar_height * (slot + 0.5))
                xs.append(100.0 * entry["term_error_rate"])
        ax.barh(ys, xs, height=bar_height * 0.86, color=SERIES[slot],
                label=label, zorder=3, linewidth=0)
        for x, y in zip(xs, ys):
            ax.text(x + 0.3, y, "%.1f" % x, va="center", fontsize=7.6,
                    color=INK_SECONDARY)

    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.replace("_", " ") for c in categories], fontsize=9)
    ax.invert_yaxis()
    legend = ax.legend(frameon=False, fontsize=9, loc="lower right")
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    save(fig, outdir, "figure5_terminology")
    return True


def figure6_forgetting(outdir, records):
    """Hospital gain against general-domain loss. Scatter: two measures."""
    index = index_by(records, "model_key", "arm_key", "budget", "augmented",
                     "replay", "seed", "test_set")
    systems = [
        ("Zero-shot", ("qwen06", "zero", None, False, False, 42)),
        ("Full SFT 20 h", ("qwen06", "full", "20h", False, False, 42)),
        ("Text-LoRA 20 h", ("qwen06", "textlora", "20h", False, False, 42)),
        ("DualPEFT 20 h", ("qwen06", "dualpeft", "20h", False, False, 42)),
        ("DualPEFT 20 h + replay", ("qwen06", "dualpeft", "20h", False, True, 42)),
    ]

    points = []
    for slot, (label, base_key) in enumerate(systems):
        hospital = index.get(base_key + ("test_synthetic",))
        general = index.get(base_key + ("test_aishell1",))
        if (hospital and hospital.get("cer") is not None
                and general and general.get("cer") is not None):
            points.append((label, 100.0 * hospital["cer"],
                           100.0 * general["cer"], SERIES[slot]))

    if not points:
        print("  SKIP figure6: need both hospital and AISHELL-1 CER for a system")
        return False

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    style_axes(ax, "Hospital-domain CER (%)", "AISHELL-1 general CER (%)",
               "Figure 6  Hospital gain against general-domain forgetting")

    baseline = next((p for p in points if p[0] == "Zero-shot"), None)
    if baseline:
        ax.axhline(baseline[2], color=INK_SECONDARY, linewidth=1.0,
                   linestyle=(0, (4, 3)), zorder=2)
        ax.text(ax.get_xlim()[1], baseline[2],
                " zero-shot general CER", fontsize=8.2, color=INK_SECONDARY,
                va="bottom", ha="right")

    for label, hospital_cer, general_cer, color in points:
        ax.scatter([hospital_cer], [general_cer], s=110, color=color,
                   edgecolor=SURFACE, linewidth=1.8, zorder=4)
        ax.annotate(label, (hospital_cer, general_cer),
                    textcoords="offset points", xytext=(9, 5), fontsize=8.8,
                    color=color, fontweight="bold")

    ax.text(0.02, 0.02,
            "lower-left is better: hospital gain without general-domain loss",
            transform=ax.transAxes, fontsize=8.4, color=INK_SECONDARY)
    save(fig, outdir, "figure6_forgetting")
    return True


def main():
    ap = argparse.ArgumentParser(description="Generate publication figures.")
    ap.add_argument("--experiments", default=str(_ROOT / "experiments"))
    ap.add_argument("--results", default=str(_ROOT / "results"))
    ap.add_argument("--data", default=str(_ROOT / "data"))
    ap.add_argument("--outdir", default=str(_ROOT / "results" / "figures"))
    args = ap.parse_args()

    records = collect_experiments(args.experiments)
    print("collected %d experiment/test-set records" % len(records))

    made = []
    made.append(("figure1", figure1_component_diagram(args.outdir, args.results)))
    made.append(("figure2", figure2_data_pipeline(args.outdir, args.data)))
    made.append(("figure3", figure3_cer_vs_hours(args.outdir, records)))
    made.append(("figure4", figure4_cer_by_condition(args.outdir, records)))
    made.append(("figure5", figure5_terminology(args.outdir, records)))
    made.append(("figure6", figure6_forgetting(args.outdir, records)))

    done = [name for name, ok in made if ok]
    skipped = [name for name, ok in made if not ok]
    print("")
    print("generated: %s" % (", ".join(done) if done else "none"))
    if skipped:
        print("skipped  : %s  (inputs not on disk - no figure is drawn from "
              "invented numbers)" % ", ".join(skipped))


if __name__ == "__main__":
    main()
