#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 16 - Execute one experiment config end to end.

    python scripts/run_experiment.py configs/qwen06_dualpeft_20h.yaml

Produces the self-contained experiment directory the specification requires:

    experiments/<name>/
        config.yaml               resolved configuration
        command.txt               exact command line
        environment.txt           versions captured at run time
        train.log                 full stdout/stderr of this run
        trainable_parameters.txt  what was trainable, by component
        metrics.json              training metrics
        checkpoints/              saved checkpoints
        <test_set>/predictions.jsonl
        <test_set>/metrics.json

The training step refuses to start unless the Phase 11 safety check passes, so
an experiment directory can only exist if the arm was verified to isolate its
component.

A zero-shot config (`arm: A0_zero_shot`) skips training and runs evaluation
directly against the base checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def load_config(path):
    """Read the generated YAML. Uses PyYAML when present, else a small parser.

    The fallback exists so a config can still be read in a stripped environment;
    it understands exactly the shapes ``scripts/make_configs.py`` emits: scalars,
    one level of nesting, and lists of scalars.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml

        return yaml.safe_load(text)
    except ImportError:
        pass

    def parse_scalar(raw):
        raw = raw.strip()
        if raw in ("null", "~", ""):
            return None
        if raw in ("true", "false"):
            return raw == "true"
        if raw.startswith('"') and raw.endswith('"'):
            return raw[1:-1]
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            return raw

    config, current_key = {}, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()

        if stripped.startswith("- "):
            config.setdefault(current_key, [])
            if not isinstance(config[current_key], list):
                config[current_key] = []
            config[current_key].append(parse_scalar(stripped[2:]))
            continue

        key, _, raw = stripped.partition(":")
        key = key.strip()
        if indent == 0:
            if raw.strip() == "":
                config[key] = {}
                current_key = key
            else:
                config[key] = parse_scalar(raw)
                current_key = key
        else:
            if not isinstance(config.get(current_key), dict):
                config[current_key] = {}
            config[current_key][key] = parse_scalar(raw)
    return config


class Tee:
    """Mirror stdout/stderr into train.log without losing the console."""

    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8", buffering=1)
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def __enter__(self):
        sys.stdout = sys.stderr = self
        return self

    def __exit__(self, *exc):
        sys.stdout, sys.stderr = self.stdout, self.stderr
        self.file.close()


def resolve(path, root):
    """Config paths are written relative to the project root."""
    if path is None:
        return None
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else Path(root) / candidate)


def classify_checkpoint(path):
    """Is this a PEFT adapter, a standalone model, or nothing usable?

    The five PEFT arms write an ``adapter_config.json`` that is loaded on top of
    the base model. The two non-PEFT arms (A1 full SFT, A3 projection-only)
    write a full model directory instead. Passing a full-model directory as
    ``--adapter_path`` fails with "Can't find 'adapter_config.json'", so the
    kind has to be detected rather than assumed from the arm name.
    """
    if path is None:
        return None
    path = Path(path)
    if (path / "adapter_config.json").exists():
        return "adapter"
    if (path / "config.json").exists():
        return "full"
    return None


def run_evaluations(cfg, outdir, root, adapter_path=None):
    """Evaluate one checkpoint on every configured test manifest."""
    results = {}
    kind = classify_checkpoint(adapter_path)
    model_path = resolve(cfg["model_path"], root)
    if kind == "full":
        print("checkpoint is a standalone model; evaluating it directly")
        model_path = str(adapter_path)
        adapter_path = None
    elif kind == "adapter":
        print("checkpoint is a PEFT adapter; applying it to the base model")
    elif adapter_path is not None:
        print("no usable checkpoint at %s; evaluating the base model"
              % adapter_path)
        adapter_path = None
    for manifest in cfg.get("test_manifests") or []:
        manifest_path = resolve(manifest, root)
        name = Path(manifest_path).stem
        if not os.path.exists(manifest_path):
            print("  SKIP %s: manifest not found (%s)" % (name, manifest_path))
            results[name] = {"skipped": "manifest not found"}
            continue

        eval_dir = Path(outdir) / name
        command = [
            sys.executable, str(Path(root) / "src" / "evaluation" / "run_eval.py"),
            "--model_path", model_path,
            "--manifest", manifest_path,
            "--outdir", str(eval_dir),
            "--lexicon", resolve(cfg.get("lexicon"), root),
            "--device", cfg.get("device", "cuda:0"),
            "--dtype", cfg.get("dtype", "bfloat16"),
            "--language", cfg.get("eval_language", "Chinese"),
            "--context", cfg.get("eval_context", ""),
            "--tag", "%s/%s" % (cfg["experiment_name"], name),
        ]
        if adapter_path:
            command += ["--adapter_path", str(adapter_path)]

        print("\n=== evaluating on %s ===" % name)
        completed = subprocess.run(command)
        results[name] = {"returncode": completed.returncode,
                         "outdir": str(eval_dir)}
        if completed.returncode != 0:
            print("  evaluation FAILED on %s (exit %d)" % (name, completed.returncode))
    return results


def main():
    ap = argparse.ArgumentParser(description="Run one experiment config.")
    ap.add_argument("config")
    ap.add_argument("--root", default=str(_ROOT))
    ap.add_argument("--skip_train", action="store_true",
                    help="evaluate an existing checkpoint without retraining")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = args.root
    outdir = Path(resolve(cfg.get("output_dir",
                                  "experiments/%s" % cfg["experiment_name"]), root))
    outdir.mkdir(parents=True, exist_ok=True)

    from training.train_arm import write_run_provenance

    write_run_provenance(cfg, outdir, sys.argv)
    # Keep the source config verbatim alongside the resolved one.
    (outdir / "config.source.yaml").write_text(
        Path(args.config).read_text(encoding="utf-8"), encoding="utf-8")

    started = time.time()
    with Tee(outdir / "train.log"):
        print("=" * 74)
        print("EXPERIMENT %s" % cfg["experiment_name"])
        print("config     : %s" % args.config)
        print("arm        : %s" % cfg["arm"])
        print("model      : %s" % cfg["model_path"])
        print("budget     : %s" % cfg.get("data_budget"))
        print("seed       : %s" % cfg["seed"])
        print("started    : %s" % time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        print("=" * 74)

        adapter_path = None
        if cfg["arm"] != "A0_zero_shot" and not args.skip_train:
            train_manifest = resolve(cfg.get("train_manifest"), root)
            if not train_manifest or not os.path.exists(train_manifest):
                print("\nTRAIN MANIFEST MISSING: %s" % train_manifest)
                print("Generate the audio and duration subsets first "
                      "(scripts/generate_tts.py, scripts/build_duration_subsets.py).")
                return 2

            resolved = dict(cfg)
            resolved["train_manifest"] = train_manifest
            resolved["model_path"] = resolve(cfg["model_path"], root)

            from training.train_arm import train

            train(resolved, outdir)
            adapter_path = outdir / "checkpoints" / "final"
        elif cfg["arm"] != "A0_zero_shot":
            adapter_path = outdir / "checkpoints" / "final"
            print("\n--skip_train: evaluating existing checkpoint %s" % adapter_path)

        eval_results = {}
        if not args.skip_eval:
            eval_results = run_evaluations(
                cfg, outdir, root,
                adapter_path if (adapter_path and Path(adapter_path).exists()) else None)

        elapsed = time.time() - started
        print("")
        print("=" * 74)
        print("FINISHED %s in %.1f s" % (cfg["experiment_name"], elapsed))
        for name, info in eval_results.items():
            print("  %-24s %s" % (name, info))
        print("=" * 74)

    (outdir / "run_summary.json").write_text(json.dumps({
        "experiment": cfg["experiment_name"],
        "config": args.config,
        "arm": cfg["arm"],
        "elapsed_seconds": round(elapsed, 2),
        "evaluations": eval_results,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
