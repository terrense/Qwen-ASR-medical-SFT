#!/usr/bin/env python3
"""Phase 1 - Model architecture audit for Qwen3-ASR.

Loads the released checkpoint (no training, no weight mutation) and writes a
complete inventory of every ``nn.Linear``, each assigned to one of
AUDIO_ENCODER / AUDIO_PROJECTION / TEXT_DECODER / OTHER.

The component boundaries are derived from the loaded module graph by
``src/models/components.py`` - the same module the training arms use - so the
audit and the experiments can never disagree about what "the projection head"
means.

Outputs (suffixed with the model tag, plus unsuffixed copies for the primary
0.6B model so configs can reference stable filenames):

    results/model_module_inventory.csv    one row per nn.Linear
    results/model_component_summary.csv   parameter counts per component
    results/lora_target_modules.json      verified branch-disjoint LoRA targets

Why the LoRA target file exists: the audio tower and the text decoder both
define ``q_proj`` / ``k_proj`` / ``v_proj``. A PEFT config written as
``target_modules=["q_proj", "v_proj"]`` therefore adapts both branches at once,
which would silently destroy the component ablation. This script reports how
many modules such a naive target would have hit, and emits full-path regexes
that are asserted disjoint against the real inventory instead.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import torch  # noqa: E402

from models import components as C  # noqa: E402


def load_model(model_path, dtype, device):
    try:
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        sys.exit("qwen-asr is not installed in this interpreter (%s).\n"
                 "This script deliberately installs nothing itself." % exc)

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    print("loading %s (dtype=%s, device=%s) ..." % (model_path, dtype, device))
    wrapper = Qwen3ASRModel.from_pretrained(
        model_path, dtype=torch_dtype, device_map=device)
    inner = C.unwrap(wrapper)
    inner.eval()
    return inner


def summarize(rows, model):
    total = sum(p.numel() for p in model.parameters())
    params, counts = {}, {}
    for row in rows:
        comp = row["component"]
        params[comp] = params.get(comp, 0) + row["n_params"]
        counts[comp] = counts.get(comp, 0) + 1

    out = []
    for comp in (C.AUDIO_ENCODER, C.AUDIO_PROJECTION, C.TEXT_DECODER, C.OTHER):
        n = params.get(comp, 0)
        out.append(OrderedDict([
            ("component", comp),
            ("n_linear_modules", counts.get(comp, 0)),
            ("linear_params", n),
            ("pct_of_all_model_params", round(100.0 * n / total, 4) if total else 0.0),
        ]))
    out.append(OrderedDict([
        ("component", "MODEL_TOTAL (all params, incl. non-Linear)"),
        ("n_linear_modules", len(rows)),
        ("linear_params", total),
        ("pct_of_all_model_params", 100.0),
    ]))
    return out, total


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Audit Qwen3-ASR module structure.")
    ap.add_argument("--model_path", default="Qwen/Qwen3-ASR-0.6B")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cpu",
                    help="cpu is sufficient for structural inspection")
    ap.add_argument("--outdir", default=str(_ROOT / "results"))
    ap.add_argument("--tag", default=None, help="defaults to the model basename")
    ap.add_argument("--print_all", action="store_true",
                    help="print every Linear instead of layer 0 plus the rest")
    args = ap.parse_args()

    tag = args.tag or args.model_path.rstrip("/").replace("\\", "/").split("/")[-1]
    outdir = Path(args.outdir)

    model = load_model(args.model_path, args.dtype, args.device)
    rows, roots = C.classify_linears(model)
    summary, total = summarize(rows, model)
    regexes = C.attention_regexes(rows, roots)
    collisions = C.leaf_name_collisions(rows)

    print("")
    print("audio encoder root : %s" % roots["audio_root"])
    print("text decoder root  : %s" % roots["text_root"])
    print("audio d_model      : %s" % roots["d_model"])
    print("audio output_dim   : %s  (text hidden size)" % roots["output_dim"])
    print("audio projection   : %s" % ", ".join(roots["projection_paths"]))
    print("")
    print("total parameters   : %s" % format(total, ","))
    print("nn.Linear modules  : %d" % len(rows))
    print("")

    header = "%-56s %-17s %7s %7s %13s %s" % (
        "MODULE PATH", "COMPONENT", "IN", "OUT", "PARAMS", "RQ_GRAD")
    print(header)
    print("-" * len(header))
    for row in rows:
        if not args.print_all and row["layer_index"] not in (None, 0):
            continue
        print("%-56s %-17s %7d %7d %13s %s" % (
            row["module_path"][-56:], row["component"], row["in_features"],
            row["out_features"], format(row["n_params"], ","), row["requires_grad"]))
    if not args.print_all:
        print("... layer_index >= 1 suppressed; use --print_all for the full list")

    print("")
    print("%-45s %6s %16s %10s" % ("COMPONENT", "#LIN", "LINEAR PARAMS", "% MODEL"))
    print("-" * 80)
    for entry in summary:
        print("%-45s %6d %16s %9.4f%%" % (
            entry["component"], entry["n_linear_modules"],
            format(entry["linear_params"], ","), entry["pct_of_all_model_params"]))

    print("")
    print("LEAF-NAME COLLISIONS ACROSS COMPONENTS (PEFT targeting hazard)")
    print("-" * 80)
    if collisions:
        for leaf, comps in sorted(collisions.items()):
            print("  %-12s occurs in: %s" % (leaf, ", ".join(comps)))
    else:
        print("  none")

    naive = [r for r in rows if r["leaf_name"] in ("q_proj", "v_proj")]
    naive_components = sorted({r["component"] for r in naive})

    print("")
    print("VERIFIED BRANCH-DISJOINT LORA TARGETS")
    print("-" * 80)
    print("  audio regex : %s" % regexes["audio_regex"])
    print("                matches %d modules" % len(regexes["audio_matched"]))
    print("  text  regex : %s" % regexes["text_regex"])
    print("                matches %d modules" % len(regexes["text_matched"]))
    print("  projection  : %s" % ", ".join(roots["projection_paths"]))
    print("  overlap     : none (asserted by components.attention_regexes)")
    print("")
    print("  a naive target_modules=['q_proj','v_proj'] would match %d modules "
          "spanning %s" % (len(naive), naive_components))

    payload = {
        "model_path": args.model_path,
        "dtype": args.dtype,
        "total_parameters": total,
        "roots": roots,
        "leaf_name_collisions": collisions,
        "lora_targets": {
            "audio_attention_regex": regexes["audio_regex"],
            "text_attention_regex": regexes["text_regex"],
            "audio_attention_leaves": regexes["audio_leaves"],
            "text_attention_leaves": regexes["text_leaves"],
            "audio_regex_n_matched": len(regexes["audio_matched"]),
            "text_regex_n_matched": len(regexes["text_matched"]),
            "projection_module_paths": roots["projection_paths"],
            "naive_qv_target_n_matched": len(naive),
            "naive_qv_target_components": naive_components,
        },
    }

    write_csv(rows, outdir / ("model_module_inventory_%s.csv" % tag))
    write_csv(summary, outdir / ("model_component_summary_%s.csv" % tag))
    (outdir / ("lora_target_modules_%s.json" % tag)).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if "0.6B" in tag:
        write_csv(rows, outdir / "model_module_inventory.csv")
        write_csv(summary, outdir / "model_component_summary.csv")
        (outdir / "lora_target_modules.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("")
    for name in ("model_module_inventory_%s.csv" % tag,
                 "model_component_summary_%s.csv" % tag,
                 "lora_target_modules_%s.json" % tag):
        print("wrote %s" % (outdir / name))


if __name__ == "__main__":
    main()
