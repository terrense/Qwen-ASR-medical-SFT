#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 5 decision - VoiceDesign vs CustomVoice.

Runs in ``env_tts``.

The corpus needs 24-32 speaker identities that stay the same voice across
thousands of utterances while being clearly distinguishable from each other.
Those are two measurable properties, so this script measures them rather than
relying on listening impressions:

  identity stability (within-identity)
      One identity speaks K different sentences. Speaker embeddings are
      extracted from each rendering and pairwise cosine similarity is averaged.
      **Higher is better** - the voice did not drift between utterances.

  timbre diversity (between-identity)
      Average cosine similarity between embeddings of *different* identities.
      **Lower is better** - the identities do not collapse onto one voice.

  separability = within - between
      The single number that decides the question. A large gap means identities
      are simultaneously stable and mutually distinct, which is exactly what a
      speaker-disjoint train/dev/test split requires. A small gap means the
      "32 identities" would be a fiction.

Embeddings come from the Base model's speaker encoder (the same one used to
build voice-clone prompts), so the measurement uses the model's own notion of
speaker identity rather than an external one.

Also reported: synthesis throughput (RTF) and audio QC pass rate, which feed the
decision about whether full-corpus synthesis is affordable.

    env_tts/bin/python scripts/compare_tts_variants.py --n_identities 8 --n_texts 4
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
    """Draw evaluation sentences from a held-out split, never from training."""
    path = _ROOT / "data" / "manifests" / "splits" / ("%s_scripts.jsonl" % split)
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # Spread across domain categories so the comparison is not dominated by one
    # sentence shape.
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


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else 0.0


def embed(base_model, wav_path):
    """Speaker embedding via the Base model's speaker encoder."""
    item = base_model.create_voice_clone_prompt(
        ref_audio=str(wav_path), x_vector_only_mode=True)[0]
    for attribute in ("ref_spk_embedding", "spk_embedding", "x_vector"):
        vector = getattr(item, attribute, None)
        if vector is not None:
            if hasattr(vector, "detach"):
                vector = vector.detach().float().cpu().numpy()
            return np.asarray(vector).ravel()
    raise RuntimeError("no speaker embedding on VoiceClonePromptItem; "
                       "fields are %s" % dir(item))


def synthesize(model, variant, identities, texts, outdir, language="Chinese"):
    """Render every (identity, text) pair. Returns per-file records."""
    records = []
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for identity in identities:
        for index, text in enumerate(texts):
            t0 = time.time()
            if variant == "voicedesign":
                audios, sample_rate = model.generate_voice_design(
                    text=[text], instruct=[identity["instruct"]],
                    language=[language])
            else:
                audios, sample_rate = model.generate_custom_voice(
                    text=[text], speaker=[identity["speaker"]],
                    language=[language])
            elapsed = time.time() - t0

            signal = np.asarray(audios[0], dtype="float32")
            if sample_rate != audio_qc.TARGET_SAMPLE_RATE:
                ratio = audio_qc.TARGET_SAMPLE_RATE / sample_rate
                new_len = int(len(signal) * ratio)
                signal = np.interp(np.linspace(0, len(signal) - 1, new_len),
                                   np.arange(len(signal)), signal).astype("float32")
            signal, _, _ = audio_qc.trim_silence(signal)
            stats = audio_qc.measure(signal)
            passed, reasons = audio_qc.evaluate(stats)

            path = outdir / ("%s_%s_t%02d.wav" % (variant, identity["id"], index))
            audio_qc.save_audio(str(path), signal)

            records.append({
                "variant": variant, "identity": identity["id"], "text": text,
                "text_index": index, "path": str(path),
                "duration": stats["duration"], "synth_seconds": round(elapsed, 3),
                "rtf": round(elapsed / max(stats["duration"], 1e-6), 3),
                "qc_passed": passed, "qc_reasons": reasons,
            })
            print("  %-12s %-8s t%02d  %.2fs audio in %.2fs (RTF %.2f)%s"
                  % (variant, identity["id"], index, stats["duration"], elapsed,
                     records[-1]["rtf"], "" if passed else "  QC FAIL %s" % reasons))
    return records


def analyse(records, base_model):
    """Within/between identity similarity from speaker embeddings."""
    usable = [r for r in records if r["qc_passed"]]
    embeddings = {}
    for record in usable:
        embeddings.setdefault(record["identity"], []).append(embed(base_model, record["path"]))

    within_scores = []
    per_identity = OrderedDict()
    for identity, vectors in embeddings.items():
        pairs = [cosine(vectors[i], vectors[j])
                 for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
        if pairs:
            per_identity[identity] = round(float(np.mean(pairs)), 4)
            within_scores.extend(pairs)

    between_scores = []
    identities = list(embeddings)
    for i in range(len(identities)):
        for j in range(i + 1, len(identities)):
            for a in embeddings[identities[i]]:
                for b in embeddings[identities[j]]:
                    between_scores.append(cosine(a, b))

    within = float(np.mean(within_scores)) if within_scores else float("nan")
    between = float(np.mean(between_scores)) if between_scores else float("nan")
    durations = [r["duration"] for r in usable]
    rtfs = [r["rtf"] for r in usable]

    return OrderedDict([
        ("n_files", len(records)),
        ("n_qc_passed", len(usable)),
        ("qc_pass_rate", round(len(usable) / max(1, len(records)), 4)),
        ("n_identities", len(embeddings)),
        ("within_identity_similarity", round(within, 4)),
        ("between_identity_similarity", round(between, 4)),
        ("separability", round(within - between, 4)),
        ("within_min", round(float(np.min(within_scores)), 4) if within_scores else None),
        ("between_max", round(float(np.max(between_scores)), 4) if between_scores else None),
        ("per_identity_stability", per_identity),
        ("mean_rtf", round(float(np.mean(rtfs)), 3) if rtfs else None),
        ("mean_audio_seconds", round(float(np.mean(durations)), 3) if durations else None),
        ("total_synth_seconds", round(sum(r["synth_seconds"] for r in records), 1)),
    ])


def main():
    ap = argparse.ArgumentParser(description="Compare Qwen3-TTS variants.")
    ap.add_argument("--models_dir", default=str(_ROOT / "models"))
    ap.add_argument("--voicedesign", default="Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    ap.add_argument("--customvoice", default="Qwen3-TTS-12Hz-1.7B-CustomVoice")
    ap.add_argument("--base", default="Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--n_identities", type=int, default=8)
    ap.add_argument("--n_texts", type=int, default=4)
    ap.add_argument("--outdir", default=str(_ROOT / "data" / "synthetic" / "variant_comparison"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--skip", default="", help="comma list: voicedesign,customvoice")
    args = ap.parse_args()

    import torch
    from qwen_tts import Qwen3TTSModel

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[args.dtype]
    models_dir = Path(args.models_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    texts = load_texts(args.n_texts)
    print("evaluation sentences (from the held-out test split):")
    for text in texts:
        print("   %s" % text)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    results = OrderedDict()
    all_records = []

    if "voicedesign" not in skip:
        print("")
        print("=== VoiceDesign ===")
        inventory = build_speaker_inventory(42)[:args.n_identities]
        identities = [{"id": s["speaker_id"], "instruct": s["instruct"]}
                      for s in inventory]
        for identity, spec in zip(identities, inventory):
            print("  %s: %s" % (identity["id"], spec["instruct"]))
        model = Qwen3TTSModel.from_pretrained(
            str(models_dir / args.voicedesign), device_map=args.device,
            dtype=torch_dtype)
        records = synthesize(model, "voicedesign", identities, texts,
                             outdir / "voicedesign")
        all_records.extend(records)
        results["voicedesign"] = {"records": records}
        del model
        torch.cuda.empty_cache()

    if "customvoice" not in skip:
        print("")
        print("=== CustomVoice ===")
        model = Qwen3TTSModel.from_pretrained(
            str(models_dir / args.customvoice), device_map=args.device,
            dtype=torch_dtype)
        speakers = model.get_supported_speakers() or []
        print("  model exposes %d built-in speakers: %s"
              % (len(speakers), speakers[:12]))
        chosen = speakers[:args.n_identities]
        identities = [{"id": name, "speaker": name} for name in chosen]
        records = synthesize(model, "customvoice", identities, texts,
                             outdir / "customvoice")
        all_records.extend(records)
        results["customvoice"] = {"records": records, "n_available": len(speakers),
                                  "speakers_used": chosen}
        del model
        torch.cuda.empty_cache()

    print("")
    print("=== extracting speaker embeddings with the Base model ===")
    base = Qwen3TTSModel.from_pretrained(
        str(models_dir / args.base), device_map=args.device, dtype=torch_dtype)

    summary = OrderedDict()
    for variant in results:
        summary[variant] = analyse(results[variant]["records"], base)

    payload = {"texts": texts, "n_identities": args.n_identities,
               "n_texts": args.n_texts, "summary": summary,
               "records": all_records}
    report_path = outdir / "comparison_report.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    print("")
    print("%-14s %10s %10s %12s %9s %8s" % (
        "VARIANT", "WITHIN", "BETWEEN", "SEPARABILITY", "QC PASS", "RTF"))
    print("-" * 70)
    for variant, stats in summary.items():
        print("%-14s %10.4f %10.4f %12.4f %8.1f%% %8.3f" % (
            variant, stats["within_identity_similarity"],
            stats["between_identity_similarity"], stats["separability"],
            100 * stats["qc_pass_rate"], stats["mean_rtf"] or 0.0))
    print("")
    print("within  = same identity, different sentences (higher = more stable)")
    print("between = different identities (lower = more diverse)")
    print("separability = within - between; the larger, the better the identity")
    print("               inventory holds up across thousands of utterances")
    print("")
    if summary:
        best = max(summary, key=lambda v: summary[v]["separability"])
        print("higher separability: %s" % best)
    print("wrote %s" % report_path)


if __name__ == "__main__":
    main()
