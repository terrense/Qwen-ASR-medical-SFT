# -*- coding: utf-8 -*-
"""Phase 10/11/16 - Train one adaptation arm.

Wraps the official Alibaba fine-tuning contract (``finetuning/qwen3_asr_sft.py``
in the Qwen3-ASR repository, Apache-2.0) so that the prompt format, label
masking and loss are identical to the reference implementation. The only thing
this file changes is *which parameters are trainable*, which is the variable
under study.

Every run is refused unless the Phase 11 safety check passes first: parameter
accounting, frozen-module assertions, one forward/backward pass, a finite loss,
and gradients present only on the intended parameters. A run that cannot prove
it is isolating its component does not start.

Each run writes a self-contained experiment directory (Phase 16):

    config.yaml               the resolved configuration
    command.txt               the exact command line
    environment.txt           versions captured at run time
    trainable_parameters.txt  what was trainable, with counts by component
    train.log                 stdout/stderr
    metrics.json              losses, timings, peak VRAM, GPU hours
    checkpoints/              saved checkpoints
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from models import components as C  # noqa: E402


def load_audio(path, sample_rate=16000):
    import librosa

    wav, _ = librosa.load(path, sr=sample_rate, mono=True)
    return wav


class Collator:
    """The official collator, reimplemented against the same processor contract.

    The prefix is rendered with the chat template, the target plus EOS is
    appended, and every prefix or padding position is masked to -100 so the loss
    covers only the transcription. Keeping this identical to the reference
    implementation is what makes the arms comparable to a standard fine-tune.
    """

    def __init__(self, processor, sample_rate=16000, prompt=""):
        self.processor = processor
        self.sample_rate = sample_rate
        self.prompt = prompt

    def __call__(self, features):
        audios = [load_audio(f["audio"], self.sample_rate) for f in features]
        targets = [f["text"] for f in features]

        messages = [
            [{"role": "system", "content": self.prompt},
             {"role": "user", "content": [{"type": "audio", "audio": None}]}]
            for _ in features
        ]
        prefix_texts = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefix_texts, targets)]

        full = self.processor(text=full_texts, audio=audios,
                              return_tensors="pt", padding=True, truncation=False)
        prefix = self.processor(text=list(prefix_texts), audio=audios,
                                return_tensors="pt", padding=True, truncation=False)

        labels = full["input_ids"].clone()
        for i, prefix_len in enumerate(prefix["attention_mask"].sum(dim=1).tolist()):
            labels[i, :int(prefix_len)] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        full["labels"] = labels
        return full


def patch_outer_forward(model):
    """Give Trainer a ``forward(**batch)`` that delegates to ``.thinker``.

    Patched on the class, not the instance, because Trainer may rebind the
    model object during training. Mirrors the official script.
    """
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("model has no .thinker; incompatible qwen-asr build")

    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels, **kwargs)

    cls.forward = forward
    cls._forward_patched = True


def sanitize_generation_config(model):
    """Drop generation flags that transformers refuses to serialize.

    Qwen3-ASR ships ``generation_config`` with ``temperature=1e-06`` while
    ``do_sample`` is False. At load time transformers only logs "the following
    generation flags are not valid and may be ignored: ['temperature']", but
    ``save_pretrained`` validates in strict mode and *raises*:

        ValueError: GenerationConfig is invalid

    That makes the two arms which do not go through PEFT - A1 full SFT and A3
    projection-only - unable to write any checkpoint at all, while the five PEFT
    arms save fine because they serialize an adapter instead. A3 is the cheapest
    and most interesting arm in the study, so this had to be fixed rather than
    worked around.

    The flag is already ignored during greedy decoding, so clearing it changes
    no behaviour; it is only removed from what gets written to disk. Returns the
    original values so the caller can restore them.
    """
    saved = []
    seen = set()
    candidates = [model]
    if hasattr(model, "thinker"):
        candidates.append(model.thinker)

    for module in candidates:
        config = getattr(module, "generation_config", None)
        if config is None or id(config) in seen:
            continue
        seen.add(id(config))
        for field in ("temperature", "top_p", "top_k"):
            value = getattr(config, field, None)
            if value is None:
                continue
            # Only meaningful when sampling; harmless to drop otherwise.
            if not getattr(config, "do_sample", False):
                saved.append((config, field, value))
                setattr(config, field, None)
    if saved:
        print("sanitized generation_config for saving: cleared %s"
              % ", ".join(sorted({f for _, f, _ in saved})))
    return saved


def restore_generation_config(saved):
    for config, field, value in saved:
        setattr(config, field, value)


def make_checkpoint_inferable(base_model_path, checkpoint_dir):
    """Copy the sidecar files a multimodal checkpoint needs to be reloadable.

    ``trainer.save_model`` writes weights and ``config.json`` but not the
    tokenizer, processor or chat template. A Qwen3-ASR checkpoint is not
    self-describing from weights alone, so without these a full-fine-tune
    checkpoint has the right parameters and still cannot be loaded back for
    evaluation. Mirrors ``copy_required_hf_files_for_qwen_asr`` in the official
    fine-tuning script.

    Only needed for the non-PEFT arms (A1, A3); a PEFT adapter directory is
    loaded on top of the base model, which already has these files.
    """
    required = [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json", "merges.txt",
        "vocab.json",
    ]
    import shutil

    os.makedirs(checkpoint_dir, exist_ok=True)
    copied = []
    for name in required:
        source = os.path.join(base_model_path, name)
        target = os.path.join(checkpoint_dir, name)
        if os.path.exists(source) and not os.path.exists(target):
            shutil.copy2(source, target)
            copied.append(name)
    if copied:
        print("copied %d sidecar files into the checkpoint so it can be "
              "reloaded: %s" % (len(copied), ", ".join(copied)))
    return copied


def read_manifest(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def run_safety_check(wrapper, arm, cfg, outdir, manifest_rows):
    """Phase 11. Refuses to return unless every check passes."""
    sys.path.insert(0, str(_ROOT / "tests"))
    from test_trainable_parameters import build_batch

    inner = C.unwrap(wrapper)
    report = C.apply_arm(wrapper, arm, lora_r=cfg["lora"]["r"],
                         lora_alpha=cfg["lora"]["alpha"],
                         lora_dropout=cfg["lora"]["dropout"], verbose=True)
    inner = C.unwrap(wrapper)
    intended = set(report["trainable_parameter_names"])

    wrongly_trainable = [n for n, p in inner.named_parameters()
                         if p.requires_grad and n not in intended]
    if wrongly_trainable:
        raise AssertionError("%d parameters trainable but unreported: %s"
                             % (len(wrongly_trainable), wrongly_trainable[:5]))

    if arm == "A0_zero_shot":
        return report, None

    sample = manifest_rows[:2] if len(manifest_rows) >= 2 else manifest_rows[:1]
    audios = [load_audio(r["audio"]) for r in sample]
    targets = [r["text"] for r in sample]
    batch = build_batch(wrapper.processor, audios, targets)
    device = next(inner.parameters()).device
    batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(
            next(inner.parameters()).dtype)

    thinker = inner.thinker if hasattr(inner, "thinker") else inner
    inner.train()
    out = thinker(input_ids=batch.get("input_ids"),
                  attention_mask=batch.get("attention_mask"),
                  input_features=batch.get("input_features"),
                  feature_attention_mask=batch.get("feature_attention_mask"),
                  labels=batch.get("labels"))
    loss_value = float(out.loss.detach().float())
    if not np.isfinite(loss_value):
        raise AssertionError("initial loss is not finite: %s" % loss_value)
    out.loss.backward()

    # PEFT initialises lora_B to zeros, so on the first backward pass
    # dL/d(lora_A) = lora_B^T (...) = 0. A zero gradient on lora_A is therefore
    # the expected mathematics of LoRA, not a detached adapter. Graph membership
    # is checked by the presence of a gradient tensor; the stronger proof (perturb
    # lora_B, re-run backward, require non-zero lora_A gradient) lives in
    # tests/test_trainable_parameters.py, which does not have to preserve weights.
    received = {n for n, p in inner.named_parameters() if p.grad is not None}
    nonzero = {n for n, p in inner.named_parameters()
               if p.grad is not None and p.grad.abs().sum().item() > 0}

    unexpected = sorted(nonzero - intended)
    if unexpected:
        raise AssertionError("gradient leaked onto %d unintended parameters: %s"
                             % (len(unexpected), unexpected[:3]))

    missing = sorted(n for n in intended if n not in received)
    if missing:
        raise AssertionError(
            "%d intended parameters received no gradient tensor: %s"
            % (len(missing), missing[:3]))

    non_lora_zero = sorted(n for n in intended
                           if n in received and n not in nonzero
                           and "lora_" not in n)
    if non_lora_zero:
        raise AssertionError(
            "%d non-LoRA trainable parameters have a zero gradient: %s"
            % (len(non_lora_zero), non_lora_zero[:3]))

    with_grad = nonzero
    inner.zero_grad(set_to_none=True)

    (Path(outdir) / "trainable_parameters.txt").write_text(
        C.format_report(report)
        + "\n\nsafety check\n"
        + json.dumps({"initial_loss": loss_value,
                      "n_params_with_gradient": len(with_grad),
                      "batch_source": "first %d manifest rows" % len(sample)},
                     indent=2),
        encoding="utf-8")
    print("\nsafety check passed: initial loss %.4f, gradients on %d parameters"
          % (loss_value, len(with_grad)))
    return report, loss_value


def train(cfg, outdir):
    """Run one arm end to end."""
    from qwen_asr import Qwen3ASRModel
    from transformers import Trainer, TrainingArguments

    outdir = Path(outdir)
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg["dtype"]]

    train_rows = read_manifest(cfg["train_manifest"], cfg.get("limit"))
    print("train manifest: %s (%d utterances, %.3f h)"
          % (cfg["train_manifest"], len(train_rows),
             sum(r.get("duration", 0) or 0 for r in train_rows) / 3600.0))

    wrapper = Qwen3ASRModel.from_pretrained(
        cfg["model_path"], dtype=dtype, device_map=cfg["device"])

    report, initial_loss = run_safety_check(
        wrapper, cfg["arm"], cfg, outdir, train_rows)

    inner = C.unwrap(wrapper)
    patch_outer_forward(inner)
    # Applied before training, not just before the final save, so that
    # intermediate `save_steps` checkpoints do not hit the same failure.
    saved_generation_flags = sanitize_generation_config(inner)
    if cfg.get("gradient_checkpointing"):
        try:
            inner.gradient_checkpointing_enable()
        except Exception as exc:
            print("gradient checkpointing unavailable: %s" % exc)

    collator = Collator(wrapper.processor, prompt=cfg.get("prompt", ""))

    args = TrainingArguments(
        output_dir=str(outdir / "checkpoints"),
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_acc"],
        learning_rate=cfg["lr"],
        num_train_epochs=cfg["epochs"],
        warmup_ratio=cfg.get("warmup_ratio", 0.03),
        lr_scheduler_type=cfg.get("scheduler", "cosine"),
        logging_steps=cfg.get("logging_steps", 10),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=cfg.get("save_total_limit", 3),
        bf16=(cfg["dtype"] == "bfloat16"),
        fp16=(cfg["dtype"] == "float16"),
        seed=cfg["seed"],
        data_seed=cfg["seed"],
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=cfg.get("num_workers", 2),
    )

    trainer = Trainer(model=inner, args=args, train_dataset=train_rows,
                      data_collator=collator)

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    result = trainer.train()
    wall_seconds = time.time() - t0

    final_dir = outdir / "checkpoints" / "final"
    trainer.save_model(str(final_dir))
    restore_generation_config(saved_generation_flags)

    # PEFT arms write an adapter that is applied on top of the base model; the
    # non-PEFT arms write a standalone model that needs its sidecar files.
    if not (final_dir / "adapter_config.json").exists():
        make_checkpoint_inferable(cfg["model_path"], str(final_dir))

    peak_vram = (torch.cuda.max_memory_allocated() / 1024 ** 3
                 if torch.cuda.is_available() else None)

    metrics = OrderedDict([
        ("arm", cfg["arm"]),
        ("config", cfg),
        ("initial_loss", initial_loss),
        ("train_runtime_seconds", round(wall_seconds, 2)),
        ("gpu_hours", round(wall_seconds / 3600.0, 4)),
        ("peak_vram_gib", round(peak_vram, 3) if peak_vram else None),
        ("total_parameters", report["total_parameters"]),
        ("trainable_parameters", report["trainable_parameters"]),
        ("trainable_percentage", report["trainable_percentage"]),
        ("trainable_by_component", report["trainable_by_component"]),
        ("n_train_utterances", len(train_rows)),
        ("train_hours_audio", round(
            sum(r.get("duration", 0) or 0 for r in train_rows) / 3600.0, 4)),
        ("hf_train_metrics", dict(result.metrics)),
        ("log_history", trainer.state.log_history),
        ("finished_at", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
    ])
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    print("")
    print("arm                 : %s" % cfg["arm"])
    print("trainable parameters: %s (%.4f%%)"
          % (format(report["trainable_parameters"], ","),
             report["trainable_percentage"]))
    print("train runtime       : %.1f s (%.3f GPU-hours)"
          % (wall_seconds, wall_seconds / 3600.0))
    if peak_vram:
        print("peak VRAM           : %.2f GiB" % peak_vram)
    print("wrote %s" % (outdir / "metrics.json"))
    return metrics


def write_run_provenance(cfg, outdir, argv):
    """config.yaml + command.txt + environment.txt, before training starts."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lines = ["# Resolved configuration for this run - written by train_arm.py"]
    for key, value in cfg.items():
        lines.append("%s: %s" % (key, json.dumps(value, ensure_ascii=False)))
    (outdir / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (outdir / "command.txt").write_text(
        "%s %s\n" % (sys.executable, " ".join(argv)), encoding="utf-8")

    try:
        subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "audit_environment.py"),
             "--outdir", str(outdir)],
            check=False, capture_output=True, timeout=300)
    except Exception as exc:
        (outdir / "environment.txt").write_text(
            "environment audit failed: %s\n" % exc, encoding="utf-8")
