"""Phase 11 - Mandatory training safety check.

Run this before every real training job. It executes all seven required checks
for one adaptation arm:

  1. print total / trainable parameter counts and the percentage
  2. list the trainable module names
  3. assert every module that should be frozen has requires_grad=False
  4. run one forward/backward pass on a real batch
  5. assert gradients exist ONLY on the intended parameters
  6. assert the loss is finite
  7. save the trainable-module report to disk

Usage:
    python tests/test_trainable_parameters.py \
        --model_path /path/Qwen3-ASR-0.6B --arm A7_dualpeft \
        --manifest data/manifests/smoke.jsonl --outdir experiments/smoke

Without ``--manifest`` the check runs on a synthetic waveform, which is enough
to verify gradient routing but does not exercise real audio loading.

The exit code is non-zero if any assertion fails, so this can gate a launcher
script. Under pytest, the arm list is swept with a synthetic batch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from models import components as C  # noqa: E402

MODEL_PATH_ENV = "QWEN3_ASR_MODEL_PATH"


def build_batch(processor, audio_arrays, targets, prompt=""):
    """Reproduce the official collator contract from finetuning/qwen3_asr_sft.py.

    The prefix is rendered with the chat template, the target plus EOS is
    appended, and every prefix/padding position is masked to -100 so the loss
    covers only the transcription.
    """
    prefix_messages = [
        [{"role": "system", "content": prompt},
         {"role": "user", "content": [{"type": "audio", "audio": None}]}]
        for _ in audio_arrays
    ]
    prefix_texts = processor.apply_chat_template(
        prefix_messages, add_generation_prompt=True, tokenize=False)

    eos = processor.tokenizer.eos_token or ""
    full_texts = [p + t + eos for p, t in zip(prefix_texts, targets)]

    full_inputs = processor(text=full_texts, audio=list(audio_arrays),
                            return_tensors="pt", padding=True, truncation=False)
    prefix_inputs = processor(text=list(prefix_texts), audio=list(audio_arrays),
                              return_tensors="pt", padding=True, truncation=False)

    prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
    labels = full_inputs["input_ids"].clone()
    for i, prefix_len in enumerate(prefix_lens):
        labels[i, :int(prefix_len)] = -100
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    full_inputs["labels"] = labels
    return full_inputs


def load_batch_audio(manifest, n, sample_rate=16000):
    """Load the first N waveforms from a manifest, or synthesize if absent."""
    if manifest and os.path.exists(manifest):
        import librosa

        sys.path.insert(0, str(_ROOT / "src"))
        from data.manifest import read_manifest

        rows = read_manifest(manifest)[:n]
        audios = [librosa.load(r["audio"], sr=sample_rate, mono=True)[0] for r in rows]
        targets = [r["text"] for r in rows]
        return audios, targets, "manifest:%s" % manifest

    rng = np.random.default_rng(42)
    audios = [rng.normal(0, 0.01, sample_rate * 2).astype(np.float32) for _ in range(n)]
    targets = ["请问神经内科门诊在几楼", "帮我预约明天上午的增强CT"][:n]
    return audios, targets, "synthetic noise (2s, seed 42)"


def check_arm(model_path, arm, manifest=None, outdir=None, batch_size=2,
              device="cuda:0", dtype="bfloat16", lora_r=16, lora_alpha=32,
              lora_dropout=0.05):
    """Run the seven checks for one arm. Raises AssertionError on failure."""
    from qwen_asr import Qwen3ASRModel

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    print("=" * 74)
    print("TRAINING SAFETY CHECK - arm=%s model=%s" % (arm, model_path))
    print("=" * 74)

    wrapper = Qwen3ASRModel.from_pretrained(
        model_path, dtype=torch_dtype, device_map=device)
    processor = wrapper.processor

    # --- checks 1 and 2: parameter accounting and trainable module names ------
    report = C.apply_arm(wrapper, arm, lora_r=lora_r, lora_alpha=lora_alpha,
                         lora_dropout=lora_dropout, verbose=True)
    inner = C.unwrap(wrapper)

    intended = set(report["trainable_parameter_names"])
    assert report["trainable_parameters"] > 0 or arm == "A0_zero_shot", (
        "arm %s has zero trainable parameters" % arm)

    # --- check 3: everything else really is frozen ---------------------------
    wrongly_trainable = [n for n, p in inner.named_parameters()
                         if p.requires_grad and n not in intended]
    assert not wrongly_trainable, (
        "%d parameters are trainable but were not in the report: %s"
        % (len(wrongly_trainable), wrongly_trainable[:5]))

    frozen = [n for n, p in inner.named_parameters() if not p.requires_grad]
    print("\nfrozen parameter tensors : %d" % len(frozen))
    print("trainable parameter tensors: %d" % len(intended))

    if arm == "A0_zero_shot":
        print("\nA0 is inference-only; skipping the forward/backward check.")
        _save(report, outdir, arm, extra={"batch_source": "n/a"})
        return report

    # --- check 4: one real forward/backward ----------------------------------
    audios, targets, source = load_batch_audio(manifest, batch_size)
    print("\nbatch source: %s (%d samples)" % (source, len(audios)))

    batch = build_batch(processor, audios, targets)
    target_device = next(inner.parameters()).device
    batch = {k: (v.to(target_device) if hasattr(v, "to") else v)
             for k, v in batch.items()}
    if "input_features" in batch:
        batch["input_features"] = batch["input_features"].to(torch_dtype)

    thinker = inner.thinker if hasattr(inner, "thinker") else inner
    inner.train()
    outputs = thinker(
        input_ids=batch.get("input_ids"),
        attention_mask=batch.get("attention_mask"),
        input_features=batch.get("input_features"),
        feature_attention_mask=batch.get("feature_attention_mask"),
        labels=batch.get("labels"),
    )
    loss = outputs.loss
    assert loss is not None, "model returned no loss; check the label masking"

    # --- check 6: finite loss ------------------------------------------------
    loss_value = float(loss.detach().float())
    print("loss: %.6f" % loss_value)
    assert np.isfinite(loss_value), "loss is not finite: %s" % loss_value
    assert loss_value > 0, "loss is %s, which indicates a masking bug" % loss_value

    loss.backward()

    # --- check 5: gradients exist only where intended ------------------------
    # PEFT initialises lora_B to zeros so that the adapter is a no-op before
    # training starts. A direct consequence is that on the very first backward
    # pass
    #
    #     dL/d(lora_A) = lora_B^T (...) = 0
    #
    # so a *zero* gradient on lora_A is the expected mathematics of LoRA, not
    # evidence of a detached adapter. Treating zero as "no gradient" would fail
    # every LoRA arm for no reason.
    #
    # Membership in the computation graph is therefore checked by the presence
    # of a gradient tensor. The adapter is then separately *proved* to be on the
    # forward path: lora_B is perturbed away from zero and the backward pass is
    # repeated, after which lora_A must receive a non-zero gradient. If the
    # adapter were genuinely detached, lora_A would stay at None/zero and this
    # second check would fail - so it has real discriminating power.
    def forward_backward():
        inner.zero_grad(set_to_none=True)
        out = thinker(
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            input_features=batch.get("input_features"),
            feature_attention_mask=batch.get("feature_attention_mask"),
            labels=batch.get("labels"),
        )
        out.loss.backward()
        return float(out.loss.detach().float())

    received = {n for n, p in inner.named_parameters() if p.grad is not None}
    nonzero = {n for n, p in inner.named_parameters()
               if p.grad is not None and p.grad.abs().sum().item() > 0}

    unexpected = sorted(nonzero - intended)
    missing = sorted(n for n in intended if n not in received)
    zero_at_init = sorted(n for n in intended if n in received and n not in nonzero)

    print("")
    print("parameters with a gradient tensor : %d" % len(received))
    print("parameters with non-zero gradient : %d" % len(nonzero))
    if unexpected:
        print("UNEXPECTED gradients on %d parameters:" % len(unexpected))
        for name in unexpected[:10]:
            print("   %s" % name)
    if missing:
        print("NO gradient tensor on %d intended parameters:" % len(missing))
        for name in missing[:10]:
            print("   %s" % name)

    assert not unexpected, (
        "gradient leaked onto %d unintended parameters (first: %s). The arm is "
        "not isolating the component it claims to." % (len(unexpected), unexpected[:3]))
    assert not missing, (
        "%d intended parameters received no gradient tensor (first: %s). The "
        "adapter is attached but not on the computation path."
        % (len(missing), missing[:3]))

    non_lora_zero = [n for n in zero_at_init if "lora_" not in n]
    assert not non_lora_zero, (
        "%d non-LoRA trainable parameters have a zero gradient (first: %s); "
        "only lora_A is expected to be zero at initialisation"
        % (len(non_lora_zero), non_lora_zero[:3]))

    lora_a_zero = [n for n in zero_at_init if "lora_A" in n]
    if lora_a_zero:
        print("")
        print("%d lora_A tensors have zero gradient at step 0 "
              "(expected: lora_B is zero-initialised)." % len(lora_a_zero))
        print("perturbing lora_B and repeating the backward pass to prove the "
              "adapter is on the forward path ...")
        with torch.no_grad():
            for name, param in inner.named_parameters():
                if "lora_B" in name and param.requires_grad:
                    param.add_(torch.randn_like(param) * 0.01)
        second_loss = forward_backward()
        still_zero = [n for n in lora_a_zero
                      if inner.get_parameter(n).grad is None
                      or inner.get_parameter(n).grad.abs().sum().item() == 0]
        assert not still_zero, (
            "%d lora_A tensors still have zero gradient after lora_B was "
            "perturbed (first: %s). The adapter really is off the computation "
            "path." % (len(still_zero), still_zero[:3]))
        print("   confirmed: all %d lora_A tensors receive gradient once "
              "lora_B is non-zero (loss %.6f)" % (len(lora_a_zero), second_loss))
        # Recompute the gradient sets from the perturbed pass so the component
        # breakdown below reflects a fully active adapter.
        nonzero = {n for n, p in inner.named_parameters()
                   if p.grad is not None and p.grad.abs().sum().item() > 0}
        leaked = sorted(nonzero - intended)
        assert not leaked, (
            "after perturbation, gradient leaked onto %d unintended parameters "
            "(first: %s)" % (len(leaked), leaked[:3]))

    with_grad = nonzero

    # --- component-level confirmation ---------------------------------------
    roots = report["roots"]
    grad_components = {}
    for name in with_grad:
        clean = name
        for prefix in ("base_model.model.", "base_model."):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        import re as _re

        module_path = _re.sub(r"\.(lora_[AB])\.default$", "", clean.rsplit(".", 1)[0])
        comp = C.component_of(module_path, roots)
        grad_components[comp] = grad_components.get(comp, 0) + 1

    spec = C.ARMS[arm]
    print("\ncomponents receiving gradient: %s" % grad_components)
    if not spec["full_sft"]:
        if not spec["audio_lora"]:
            assert C.AUDIO_ENCODER not in grad_components, (
                "arm %s must not train the audio encoder body, but %d of its "
                "tensors got gradient" % (arm, grad_components[C.AUDIO_ENCODER]))
        if not spec["text_lora"]:
            assert C.TEXT_DECODER not in grad_components, (
                "arm %s must not train the text decoder, but %d of its tensors "
                "got gradient" % (arm, grad_components[C.TEXT_DECODER]))
        if not spec["train_projection"]:
            assert C.AUDIO_PROJECTION not in grad_components, (
                "arm %s must not train the audio projection, but %d of its "
                "tensors got gradient" % (arm, grad_components[C.AUDIO_PROJECTION]))
        if spec["train_projection"]:
            assert C.AUDIO_PROJECTION in grad_components, (
                "arm %s must train the audio projection but it got no gradient" % arm)

    inner.zero_grad(set_to_none=True)

    # --- check 7: persist the report ----------------------------------------
    _save(report, outdir, arm, extra={
        "batch_source": source,
        "loss": loss_value,
        "n_params_with_gradient": len(with_grad),
        "grad_components": grad_components,
    })

    print("\nALL CHECKS PASSED for arm %s" % arm)
    return report


def _save(report, outdir, arm, extra=None):
    if not outdir:
        return
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    if extra:
        payload.update(extra)
    (out / "trainable_parameters.txt").write_text(
        C.format_report(report) + "\n\n" +
        json.dumps(extra or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "trainable_parameters.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")
    print("\nwrote %s" % (out / "trainable_parameters.txt"))


def test_all_arms_isolate_their_components():
    """pytest entry point. Skipped unless a model path is exported."""
    import pytest

    model_path = os.environ.get(MODEL_PATH_ENV)
    if not model_path:
        pytest.skip("set %s to run the trainable-parameter checks" % MODEL_PATH_ENV)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = "bfloat16" if torch.cuda.is_available() else "float32"
    for arm in ["A2_audio_lora", "A3_projection_only", "A4_text_lora",
                "A5_audio_lora_proj", "A6_text_lora_proj", "A7_dualpeft"]:
        check_arm(model_path, arm, device=device, dtype=dtype, batch_size=1)


def main():
    ap = argparse.ArgumentParser(description="Phase 11 training safety check.")
    ap.add_argument("--model_path", default=os.environ.get(MODEL_PATH_ENV))
    ap.add_argument("--arm", default="A7_dualpeft",
                    help="one arm, or 'all' to sweep every trainable arm")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    if not args.model_path:
        sys.exit("--model_path is required (or export %s)" % MODEL_PATH_ENV)

    arms = ([a for a in C.ARMS if a != "A0_zero_shot"] if args.arm == "all"
            else [args.arm])
    failures = []
    for arm in arms:
        outdir = (Path(args.outdir) / arm) if args.outdir and len(arms) > 1 else args.outdir
        try:
            check_arm(args.model_path, arm, manifest=args.manifest, outdir=outdir,
                      batch_size=args.batch_size, device=args.device,
                      dtype=args.dtype, lora_r=args.lora_r,
                      lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
        except AssertionError as exc:
            print("\nFAILED arm %s: %s" % (arm, exc))
            failures.append((arm, str(exc)))
        print("")

    if failures:
        print("=" * 74)
        print("%d/%d arms FAILED:" % (len(failures), len(arms)))
        for arm, msg in failures:
            print("  %-22s %s" % (arm, msg.splitlines()[0]))
        sys.exit(1)
    print("=" * 74)
    print("%d/%d arms passed all safety checks." % (len(arms), len(arms)))


if __name__ == "__main__":
    main()
