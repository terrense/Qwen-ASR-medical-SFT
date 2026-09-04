#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 5 - Does the anchor + x-vector clone procedure hold an identity?

Runs in ``env_tts``.

The corpus design assumes that calling VoiceDesign once per utterance would let
the voice drift, and that rendering an anchor and then cloning its speaker
embedding keeps an identity fixed. This script tests that assumption instead of
trusting it, by comparing the two procedures on the same identities and the same
sentences.

The measurement is **median fundamental frequency (F0)**, not a learned speaker
embedding. F0 was chosen after the learned embedding from the Base model's
speaker encoder turned out to have no discriminative power here: it scored
cosine 0.95 between voices whose F0 differs by a factor of three, so it cannot
distinguish identities and must not be used to judge them. F0 is coarse - it
captures pitch, not full timbre - but it is physically interpretable, and a
voice that swings from 83 Hz to 251 Hz across sentences is not one identity by
any definition.

Reported per procedure:

  within-speaker F0 spread   std of median-F0 across sentences of one identity.
                             **Lower is better** - the voice stays put.
  between-speaker F0 spread  std of the per-identity mean F0.
                             **Higher is better** - identities differ.
  stability ratio            between / within. **Higher is better**: identities
                             that differ from each other by much more than they
                             wobble internally.

    env_tts/bin/python scripts/verify_identity_stability.py --n_identities 6 --n_texts 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from data import audio_qc  # noqa: E402
from data.speakers import build_speaker_inventory  # noqa: E402


def load_texts(n, split="test"):
    path = _ROOT / "data" / "manifests" / "splits" / ("%s_scripts.jsonl" % split)
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    by_category = {}
    for row in rows:
        by_category.setdefault(row["domain_category"], []).append(row)
    picked, categories = [], sorted(by_category)
    index = 0
    while len(picked) < n:
        bucket = by_category[categories[index % len(categories)]]
        picked.append(bucket[(index // len(categories)) % len(bucket)]["text"])
        index += 1
    return picked


def median_f0(path, fmin=60.0, fmax=400.0):
    """Median voiced F0 in Hz, or None when nothing voiced is detected."""
    import librosa

    signal, sample_rate = librosa.load(str(path), sr=16000)
    f0 = librosa.yin(signal, fmin=fmin, fmax=fmax, sr=sample_rate)
    voiced = f0[(f0 > fmin) & (f0 < fmax)]
    return float(np.median(voiced)) if len(voiced) else None


def save_and_measure(signal, sample_rate, path):
    signal = np.asarray(signal, dtype="float32")
    if sample_rate != audio_qc.TARGET_SAMPLE_RATE:
        ratio = audio_qc.TARGET_SAMPLE_RATE / sample_rate
        new_len = int(len(signal) * ratio)
        signal = np.interp(np.linspace(0, len(signal) - 1, new_len),
                           np.arange(len(signal)), signal).astype("float32")
    signal, _, _ = audio_qc.trim_silence(signal)
    stats = audio_qc.measure(signal)
    passed, reasons = audio_qc.evaluate(stats)
    audio_qc.save_audio(str(path), signal)
    return stats, passed, reasons


def summarize(f0_by_speaker):
    """within / between spread and the ratio between them."""
    within = []
    per_speaker = OrderedDict()
    for speaker, values in f0_by_speaker.items():
        clean = [v for v in values if v is not None]
        if len(clean) >= 2:
            per_speaker[speaker] = {
                "median_f0": round(float(np.median(clean)), 1),
                "std": round(float(np.std(clean)), 1),
                "min": round(float(np.min(clean)), 1),
                "max": round(float(np.max(clean)), 1),
                "values": [round(v, 1) for v in clean],
            }
            within.append(np.std(clean))

    means = [np.mean([v for v in vals if v is not None])
             for vals in f0_by_speaker.values()
             if any(v is not None for v in vals)]
    within_spread = float(np.mean(within)) if within else float("nan")
    between_spread = float(np.std(means)) if len(means) > 1 else float("nan")
    return OrderedDict([
        ("within_speaker_f0_std", round(within_spread, 2)),
        ("between_speaker_f0_std", round(between_spread, 2)),
        ("stability_ratio", round(between_spread / within_spread, 3)
         if within_spread and within_spread == within_spread else None),
        ("per_speaker", per_speaker),
    ])


def main():
    ap = argparse.ArgumentParser(description="Verify identity persistence.")
    ap.add_argument("--models_dir", default=str(_ROOT / "models"))
    ap.add_argument("--voicedesign", default="Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ap.add_argument("--base", default="Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--n_identities", type=int, default=6)
    ap.add_argument("--n_texts", type=int, default=4)
    ap.add_argument("--outdir", default=str(_ROOT / "data" / "synthetic" / "identity_check"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    import torch
    from qwen_tts import Qwen3TTSModel

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[args.dtype]
    models_dir = Path(args.models_dir)
    outdir = Path(args.outdir)
    (outdir / "anchors").mkdir(parents=True, exist_ok=True)
    (outdir / "direct").mkdir(parents=True, exist_ok=True)
    (outdir / "cloned").mkdir(parents=True, exist_ok=True)

    speakers = build_speaker_inventory(42)[:args.n_identities]
    texts = load_texts(args.n_texts)
    print("identities: %s" % ", ".join(s["speaker_id"] for s in speakers))
    print("sentences :")
    for text in texts:
        print("   %s" % text)

    # --- procedure A: VoiceDesign called once per utterance -----------------
    print("")
    print("=== A. direct VoiceDesign (one call per utterance) ===")
    design = Qwen3TTSModel.from_pretrained(
        str(models_dir / args.voicedesign), device_map=args.device, dtype=torch_dtype)

    direct_f0 = OrderedDict()
    anchors = OrderedDict()
    for speaker in speakers:
        speaker_id = speaker["speaker_id"]
        direct_f0[speaker_id] = []
        for index, text in enumerate(texts):
            audios, sample_rate = design.generate_voice_design(
                text=[text], instruct=[speaker["instruct"]], language=["Chinese"])
            path = outdir / "direct" / ("%s_t%02d.wav" % (speaker_id, index))
            save_and_measure(audios[0], sample_rate, path)
            value = median_f0(path)
            direct_f0[speaker_id].append(value)
            print("  %s t%02d  F0 %s" % (speaker_id, index,
                                         "%.1f" % value if value else "n/a"))

        # Anchor for this identity, rendered once from a neutral sentence.
        audios, sample_rate = design.generate_voice_design(
            text=[speaker["anchor_text"]], instruct=[speaker["instruct"]],
            language=["Chinese"])
        anchor_path = outdir / "anchors" / ("%s.wav" % speaker_id)
        save_and_measure(audios[0], sample_rate, anchor_path)
        anchors[speaker_id] = str(anchor_path)
        print("  %s anchor F0 %.1f" % (speaker_id, median_f0(anchor_path) or 0.0))

    del design
    torch.cuda.empty_cache()

    # --- procedure B: anchor once, then clone for every utterance -----------
    print("")
    print("=== B. anchor + x-vector clone (the corpus procedure) ===")
    base = Qwen3TTSModel.from_pretrained(
        str(models_dir / args.base), device_map=args.device, dtype=torch_dtype)

    cloned_f0 = OrderedDict()
    t0 = time.time()
    n_rendered = 0
    for speaker in speakers:
        speaker_id = speaker["speaker_id"]
        prompt = base.create_voice_clone_prompt(
            ref_audio=anchors[speaker_id], x_vector_only_mode=True)[0]
        cloned_f0[speaker_id] = []
        for index, text in enumerate(texts):
            audios, sample_rate = base.generate_voice_clone(
                text=[text], language=["Chinese"], voice_clone_prompt=[prompt])
            path = outdir / "cloned" / ("%s_t%02d.wav" % (speaker_id, index))
            stats, passed, reasons = save_and_measure(audios[0], sample_rate, path)
            value = median_f0(path)
            cloned_f0[speaker_id].append(value)
            n_rendered += 1
            print("  %s t%02d  F0 %s  %.2fs%s"
                  % (speaker_id, index, "%.1f" % value if value else "n/a",
                     stats["duration"], "" if passed else "  QC FAIL %s" % reasons))
    clone_seconds = time.time() - t0

    direct_summary = summarize(direct_f0)
    cloned_summary = summarize(cloned_f0)

    report = {
        "texts": texts,
        "n_identities": len(speakers),
        "direct_voicedesign": direct_summary,
        "anchor_plus_clone": cloned_summary,
        "clone_throughput": {
            "n_utterances": n_rendered,
            "seconds": round(clone_seconds, 1),
            "seconds_per_utterance": round(clone_seconds / max(1, n_rendered), 2),
        },
    }
    (outdir / "identity_stability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("%-26s %14s %15s %10s" % ("PROCEDURE", "WITHIN F0 std", "BETWEEN F0 std",
                                    "RATIO"))
    print("-" * 70)
    for label, summary in (("direct VoiceDesign", direct_summary),
                           ("anchor + x-vector clone", cloned_summary)):
        print("%-26s %14.2f %15.2f %10s"
              % (label, summary["within_speaker_f0_std"],
                 summary["between_speaker_f0_std"],
                 summary["stability_ratio"]))
    print("")
    print("within  = how much one identity wobbles across sentences (lower better)")
    print("between = how far identities sit apart (higher better)")
    print("ratio   = between / within; higher means the identity inventory holds")
    print("")
    print("per-identity F0 under the clone procedure:")
    for speaker_id, stats in cloned_summary["per_speaker"].items():
        print("  %-8s median %6.1f Hz  std %5.1f  range %.0f-%.0f  %s"
              % (speaker_id, stats["median_f0"], stats["std"], stats["min"],
                 stats["max"], stats["values"]))
    print("")
    print("clone throughput: %.2f s per utterance (batch size 1)"
          % report["clone_throughput"]["seconds_per_utterance"])
    print("wrote %s" % (outdir / "identity_stability_report.json"))


if __name__ == "__main__":
    main()
