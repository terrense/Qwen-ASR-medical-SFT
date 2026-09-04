#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 5 - Synthesize the hospital corpus with Qwen3-TTS.

Runs in ``env_tts`` (transformers 4.57.3), never in the ASR training env.

Two stages:

  ``--stage anchors``
      For every synthetic identity, render one domain-neutral anchor sentence
      with VoiceDesign, then extract a speaker embedding from that anchor with
      ``create_voice_clone_prompt(..., x_vector_only_mode=True)``. Anchors and
      their SHA-256 checksums are written to disk. This is what makes an
      identity persistent: VoiceDesign re-interprets its instruction on every
      call and would otherwise drift between utterances.

  ``--stage corpus``
      Render every script in a split with ``generate_voice_clone`` using the
      stored embedding of its assigned speaker, run Phase 6 quality control,
      and write a Phase 2 manifest.

Speaker pools are disjoint: train voices are never used for dev or test.

    python scripts/generate_tts.py --stage anchors
    python scripts/generate_tts.py --stage corpus --split train --limit 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from data import audio_qc  # noqa: E402
from data.manifest import validate_manifest, format_report, write_manifest  # noqa: E402
from data.speakers import (assign_speakers, build_speaker_inventory,  # noqa: E402
                           inventory_report, pool_members)

DEFAULT_VOICEDESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

POOL_OF_SPLIT = {"train": "train", "dev": "dev", "test": "test",
                 "cross_tts": "test"}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scripts(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tts(model_path, device, dtype):
    import torch
    from qwen_tts import Qwen3TTSModel

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    print("loading TTS %s ..." % model_path)
    return Qwen3TTSModel.from_pretrained(model_path, device_map=device,
                                         dtype=torch_dtype)


def stage_anchors(args):
    """Render one anchor per identity and store its speaker embedding."""
    import numpy as np

    speakers = build_speaker_inventory(args.seed)
    outdir = Path(args.outdir) / "anchors"
    outdir.mkdir(parents=True, exist_ok=True)

    report = inventory_report(speakers)
    print("speaker inventory: %d identities  %s"
          % (report["n_speakers"], report["by_pool"]))
    for pool, composition in report["pool_composition"].items():
        print("  %-6s %s" % (pool, json.dumps(composition, ensure_ascii=False)))

    design = load_tts(args.voicedesign_model, args.device, args.dtype)

    texts = [s["anchor_text"] for s in speakers]
    instructs = [s["instruct"] for s in speakers]
    print("\nrendering %d anchors with VoiceDesign ..." % len(texts))

    waveforms = []
    for start in range(0, len(texts), args.batch_size):
        chunk_text = texts[start:start + args.batch_size]
        chunk_instruct = instructs[start:start + args.batch_size]
        audios, sample_rate = design.generate_voice_design(
            text=chunk_text, instruct=chunk_instruct,
            language=["Chinese"] * len(chunk_text))
        waveforms.extend(audios)
        print("  %d/%d" % (min(start + args.batch_size, len(texts)), len(texts)))

    for speaker, audio in zip(speakers, waveforms):
        path = outdir / ("%s.wav" % speaker["speaker_id"])
        audio_qc.save_audio(str(path), np.asarray(audio, dtype="float32"),
                            sample_rate)
        stats = audio_qc.measure(np.asarray(audio, dtype="float32"), sample_rate)
        passed, reasons = audio_qc.evaluate(stats)
        speaker["anchor_path"] = str(path)
        speaker["anchor_sha256"] = sha256_file(path)
        speaker["anchor_duration"] = stats["duration"]
        speaker["anchor_qc_passed"] = passed
        speaker["anchor_qc_reasons"] = reasons
        if not passed:
            print("  WARNING %s anchor failed QC: %s" % (speaker["speaker_id"], reasons))

    payload = {
        "seed": args.seed,
        "voicedesign_model": args.voicedesign_model,
        "sample_rate": sample_rate,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "inventory_report": report,
        "speakers": speakers,
    }
    out_json = Path(args.outdir) / "speakers.json"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    n_failed = sum(1 for s in speakers if not s["anchor_qc_passed"])
    print("\nwrote %d anchors to %s" % (len(speakers), outdir))
    print("anchors failing QC: %d" % n_failed)
    print("wrote %s" % out_json)
    if n_failed:
        print("\nRe-render the failing identities before generating the corpus; "
              "a bad anchor poisons every utterance of that speaker.")


def stage_corpus(args):
    """Render one split with the stored speaker embeddings."""
    import numpy as np

    speakers_file = Path(args.outdir) / "speakers.json"
    if not speakers_file.exists():
        sys.exit("run --stage anchors first (%s missing)" % speakers_file)
    payload = json.loads(speakers_file.read_text(encoding="utf-8"))
    speakers = payload["speakers"]

    bad = [s["speaker_id"] for s in speakers if not s.get("anchor_qc_passed", True)]
    if bad and not args.allow_bad_anchors:
        sys.exit("anchors failed QC for %s; re-render them or pass "
                 "--allow_bad_anchors to proceed deliberately" % bad)

    scripts_path = (Path(args.scripts) if args.scripts else
                    _ROOT / "data" / "manifests" / "splits" /
                    ("%s_scripts.jsonl" % args.split))
    scripts = load_scripts(scripts_path)
    if args.limit:
        scripts = scripts[:args.limit]
    print("split %s: %d scripts from %s" % (args.split, len(scripts), scripts_path))

    pool = POOL_OF_SPLIT[args.split]
    members = pool_members(speakers, pool)
    print("speaker pool '%s': %d voices (%s)"
          % (pool, len(members), ", ".join(s["speaker_id"] for s in members)))

    assignment, counts = assign_speakers(
        [s["script_id"] for s in scripts], speakers, pool, seed=args.seed,
        max_share=args.max_speaker_share)
    print("per-speaker utterances: min %d max %d" % (min(counts.values()),
                                                     max(counts.values())))

    base = load_tts(args.base_model, args.device, args.dtype)

    # One clone prompt per speaker, reused for every utterance of that voice.
    print("\nextracting speaker embeddings from anchors ...")
    prompts = {}
    for speaker in members:
        item = base.create_voice_clone_prompt(
            ref_audio=speaker["anchor_path"], x_vector_only_mode=True)
        prompts[speaker["speaker_id"]] = item[0]

    audio_dir = Path(args.outdir) / "audio" / args.split
    audio_dir.mkdir(parents=True, exist_ok=True)

    by_speaker = OrderedDict()
    for row in scripts:
        by_speaker.setdefault(assignment[row["script_id"]], []).append(row)

    manifest_rows = []
    qc_results = []
    removed = []
    engine = "qwen3-tts-12hz-1.7b (voicedesign anchor + x-vector clone)"
    t_start = time.time()
    done = 0

    for speaker_id, rows in by_speaker.items():
        prompt = prompts[speaker_id]
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            audios, sample_rate = base.generate_voice_clone(
                text=[r["text"] for r in chunk],
                language=["Chinese"] * len(chunk),
                voice_clone_prompt=[prompt] * len(chunk))

            for row, audio in zip(chunk, audios):
                utt_id = "%s_%s" % (row["script_id"], speaker_id)
                path = audio_dir / ("%s.wav" % utt_id)
                signal = np.asarray(audio, dtype="float32")

                if sample_rate != audio_qc.TARGET_SAMPLE_RATE:
                    ratio = audio_qc.TARGET_SAMPLE_RATE / sample_rate
                    new_len = int(len(signal) * ratio)
                    signal = np.interp(np.linspace(0, len(signal) - 1, new_len),
                                       np.arange(len(signal)), signal).astype("float32")

                signal, _, _ = audio_qc.trim_silence(signal)
                stats = audio_qc.measure(signal)
                passed, reasons = audio_qc.evaluate(stats)
                qc_results.append({"audio": str(path), "passed": passed,
                                   "reasons": reasons, "stats": stats})

                if not passed:
                    removed.append({"utt_id": utt_id, "script_id": row["script_id"],
                                    "speaker_id": speaker_id, "text": row["text"],
                                    "reasons": reasons, "stats": stats})
                    continue

                audio_qc.save_audio(str(path), signal)
                manifest_rows.append(OrderedDict([
                    ("utt_id", utt_id),
                    ("audio", str(path)),
                    ("text", row["text"]),
                    ("speaker_id", speaker_id),
                    ("source", "synthetic_qwen3tts"),
                    ("tts_engine", engine),
                    ("domain_category", row["domain_category"]),
                    ("template_family", row["template_family"]),
                    ("duration", stats["duration"]),
                    ("condition", "clean"),
                    ("snr", None),
                    ("sir", None),
                    ("script_id", row["script_id"]),
                    ("split", args.split),
                ]))

            done += len(chunk)
            if done % args.log_every < args.batch_size or done >= len(scripts):
                elapsed = time.time() - t_start
                print("  %d/%d  (%.1fs, %.2f utt/s)"
                      % (done, len(scripts), elapsed, done / max(elapsed, 1e-6)))

    manifest_path = Path(args.manifest or
                         (_ROOT / "data" / "manifests" / ("%s_synthetic.jsonl" % args.split)))
    write_manifest(manifest_rows, manifest_path)

    report = validate_manifest(manifest_rows)
    qc_report = audio_qc.summarize(qc_results)

    removal_log = Path(args.outdir) / ("removed_%s.jsonl" % args.split)
    with removal_log.open("w", encoding="utf-8") as handle:
        for item in removed:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "split": args.split, "scripts": str(scripts_path),
        "speaker_pool": pool, "n_speakers": len(members),
        "per_speaker_counts": counts,
        "tts_engine": engine, "seed": args.seed,
        "qc": qc_report, "manifest": report,
        "n_removed": len(removed),
        "removal_log": str(removal_log),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    summary_path = Path(args.outdir) / ("generation_%s.json" % args.split)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       default=str), encoding="utf-8")

    print("")
    print(format_report(report, "manifest %s" % args.split))
    print("")
    print("QC: %d/%d passed, removal rate %.4f"
          % (qc_report["n_passed"], qc_report["n_files"], qc_report["removal_rate"]))
    if qc_report["failure_reasons"]:
        for reason, count in qc_report["failure_reasons"].items():
            print("   %-45s %d" % (reason, count))
    print("audio hours: %.4f" % qc_report["total_duration_hours"])
    print("")
    print("wrote %s" % manifest_path)
    print("wrote %s" % summary_path)
    print("wrote %s (%d removed)" % (removal_log, len(removed)))


def main():
    ap = argparse.ArgumentParser(description="Qwen3-TTS corpus synthesis.")
    ap.add_argument("--stage", required=True, choices=["anchors", "corpus"])
    ap.add_argument("--split", default="train",
                    choices=["train", "dev", "test", "cross_tts"])
    ap.add_argument("--scripts", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--outdir", default=str(_ROOT / "data" / "synthetic"))
    ap.add_argument("--voicedesign_model", default=DEFAULT_VOICEDESIGN)
    ap.add_argument("--base_model", default=DEFAULT_BASE)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--max_speaker_share", type=float, default=0.10)
    ap.add_argument("--allow_bad_anchors", action="store_true")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    if args.stage == "anchors":
        stage_anchors(args)
    else:
        stage_corpus(args)


if __name__ == "__main__":
    main()
