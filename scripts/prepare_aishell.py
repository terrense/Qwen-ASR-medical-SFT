#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 14 - Build the general-domain (AISHELL-1) evaluation manifest.

AISHELL-1 is the general-domain control: every important checkpoint is scored on
it so that a hospital-domain gain can be reported against whatever general
capability it cost. Without this control, "our adaptation improves hospital CER"
is an incomplete claim.

Expected on-disk layout (the standard AISHELL-1 release):

    <root>/wav/test/S0764/BAC009S0764W0121.wav
    <root>/transcript/aishell_transcript_v0.8.txt

The transcript file has one line per utterance:

    BAC009S0764W0121 甚 至 出 现 交 易 几 乎 停 滞 的 情 况

AISHELL transcripts are space-separated by character. The spaces are a
tokenization artifact of the release, not part of the reference, and the
project's normalizer removes whitespace anyway - but they are stripped here too
so the stored manifest text matches what a human would call the transcript.

    python scripts/prepare_aishell.py --root /data/.../aishell1 --split test

Also supports ``--limit`` to build a fixed-size subsample for faster iteration;
the subsample is drawn with a fixed seed and its utt_id list is written out, so
the same subset is reused across every checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from data.manifest import format_report, validate_manifest, write_manifest  # noqa: E402

TRANSCRIPT_CANDIDATES = [
    "transcript/aishell_transcript_v0.8.txt",
    "data_aishell/transcript/aishell_transcript_v0.8.txt",
    "aishell_transcript_v0.8.txt",
]
WAV_DIR_CANDIDATES = ["wav/{split}", "data_aishell/wav/{split}", "{split}"]


def find_transcript(root):
    for candidate in TRANSCRIPT_CANDIDATES:
        path = Path(root) / candidate
        if path.exists():
            return path
    return None


def find_wav_dir(root, split):
    for candidate in WAV_DIR_CANDIDATES:
        path = Path(root) / candidate.format(split=split)
        if path.is_dir():
            return path
    return None


def read_transcripts(path):
    """utt_id -> transcript, with the release's inter-character spaces removed."""
    transcripts = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            utt_id, text = parts
            transcripts[utt_id] = "".join(text.split())
    return transcripts


def probe_duration(path):
    """Read duration without decoding the whole file. None if unreadable."""
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return round(info.frames / float(info.samplerate), 4)
    except Exception:
        return None


def rows_from_parquet(parquet_dir, wav_dir, split):
    """Unpack a HuggingFace parquet export of AISHELL-1 into WAV + rows.

    The parquet carries the audio inline as bytes plus a `transcription` string
    whose words are space separated. The spaces are a tokenization artifact of
    the export; they are removed so the stored text matches what a human would
    write, and the project normalizer strips whitespace anyway.
    """
    import glob

    import pyarrow.parquet as pq

    wav_dir = Path(wav_dir)
    wav_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(Path(parquet_dir) / ("%s-*.parquet" % split))))
    if not files:
        sys.exit("no %s-*.parquet under %s" % (split, parquet_dir))
    print("parquet shards: %d" % len(files))

    rows = []
    for path in files:
        table = pq.read_table(path)
        for record in table.to_pylist():
            audio = record["audio"]
            name = Path(audio.get("path") or "").name or ("%06d.wav" % len(rows))
            utt_id = Path(name).stem
            text = "".join(str(record["transcription"]).split())
            if not text:
                continue

            wav_path = wav_dir / ("%s.wav" % utt_id)
            if not wav_path.exists():
                wav_path.write_bytes(audio["bytes"])

            # AISHELL ids look like BAC009S0764W0121; S0764 is the speaker.
            match = re.search(r"(S\d{4})", utt_id)
            speaker = match.group(1) if match else "unknown"

            rows.append(OrderedDict([
                ("utt_id", "aishell_%s" % utt_id),
                ("audio", str(wav_path)),
                ("text", text),
                ("speaker_id", "aishell_%s" % speaker),
                ("source", "public"),
                ("tts_engine", None),
                ("domain_category", "general"),
                ("template_family", "AISHELL1_%s" % split),
                ("duration", probe_duration(wav_path) or 0.0),
                ("condition", "public_clean"),
                ("snr", None),
                ("sir", None),
            ]))
        print("  %s -> %d rows so far" % (Path(path).name, len(rows)))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Build the AISHELL-1 manifest.")
    ap.add_argument("--root", default=None, help="AISHELL-1 release root")
    ap.add_argument("--parquet_dir", default=None,
                    help="directory of HuggingFace parquet shards (alternative to --root)")
    ap.add_argument("--wav_dir", default=None,
                    help="where to unpack audio when reading parquet")
    ap.add_argument("--split", default="test", choices=["test", "dev", "train"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="fixed-seed subsample size")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_duration", action="store_true",
                    help="do not probe each file for its duration (faster)")
    args = ap.parse_args()

    if args.parquet_dir:
        wav_dir = args.wav_dir or str(_ROOT / "data" / "public" /
                                      ("aishell1_%s_wav" % args.split))
        rows = rows_from_parquet(args.parquet_dir, wav_dir, args.split)
        missing_transcript = []
        transcript_path = args.parquet_dir
        finish(args, rows, missing_transcript, transcript_path, wav_dir)
        return

    if not args.root:
        sys.exit("pass either --root (release directory) or --parquet_dir")

    root = Path(args.root)
    if not root.exists():
        sys.exit("AISHELL-1 root not found: %s" % root)

    transcript_path = find_transcript(root)
    if transcript_path is None:
        sys.exit("no transcript file under %s; looked for:\n  %s"
                 % (root, "\n  ".join(TRANSCRIPT_CANDIDATES)))

    wav_dir = find_wav_dir(root, args.split)
    if wav_dir is None:
        sys.exit("no wav directory for split '%s' under %s" % (args.split, root))

    transcripts = read_transcripts(transcript_path)
    print("transcripts: %d entries from %s" % (len(transcripts), transcript_path))
    print("audio      : %s" % wav_dir)

    rows = []
    missing_transcript = []
    for wav_path in sorted(wav_dir.rglob("*.wav")):
        utt_id = wav_path.stem
        text = transcripts.get(utt_id)
        if not text:
            missing_transcript.append(utt_id)
            continue
        # AISHELL utt ids are BAC009S<speaker>W<index>; the speaker is the
        # directory name, which is the release's own grouping.
        speaker_id = wav_path.parent.name
        rows.append(OrderedDict([
            ("utt_id", "aishell_%s" % utt_id),
            ("audio", str(wav_path)),
            ("text", text),
            ("speaker_id", "aishell_%s" % speaker_id),
            ("source", "public"),
            ("tts_engine", None),
            ("domain_category", "general"),
            ("template_family", "AISHELL1_%s" % args.split),
            ("duration", 0.0 if args.skip_duration else (probe_duration(wav_path) or 0.0)),
            ("condition", "public_clean"),
            ("snr", None),
            ("sir", None),
        ]))

    finish(args, rows, missing_transcript, transcript_path, wav_dir)


def finish(args, rows, missing_transcript, source, wav_dir):
    """Write the manifest, the utt_id list and the report."""
    if not rows:
        sys.exit("no utterances matched between the audio and the transcript")

    if args.limit and args.limit < len(rows):
        rng = random.Random(args.seed)
        rows = sorted(rng.sample(rows, args.limit), key=lambda r: r["utt_id"])
        print("subsampled to %d utterances (seed %d)" % (len(rows), args.seed))

    out_path = Path(args.out or (_ROOT / "data" / "public" /
                                 ("aishell1_%s.jsonl" % args.split)))
    write_manifest(rows, out_path)

    ids_path = out_path.with_suffix(".utt_ids.txt")
    ids_path.write_text("\n".join(r["utt_id"] for r in rows) + "\n",
                        encoding="utf-8")

    # The evaluation runner reads data/manifests/test_aishell1.jsonl by config.
    if args.split == "test":
        alias = _ROOT / "data" / "manifests" / "test_aishell1.jsonl"
        write_manifest(rows, alias)
        print("also wrote %s (the path the configs reference)" % alias)

    report = validate_manifest(rows)
    print("")
    print(format_report(report, "AISHELL-1 %s" % args.split))

    summary = {
        "source": str(source), "split": args.split,
        "transcript_source": str(source), "wav_dir": str(wav_dir),
        "n_utterances": len(rows),
        "n_speakers": len({r["speaker_id"] for r in rows}),
        "n_missing_transcript": len(missing_transcript),
        "missing_transcript_sample": missing_transcript[:20],
        "total_hours": round(report["total_duration_hours"], 4),
        "limit": args.limit, "seed": args.seed,
        "manifest": str(out_path), "utt_id_list": str(ids_path),
        "duration_probed": not args.skip_duration,
    }
    summary_path = out_path.with_name("aishell1_%s_report.json" % args.split)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    if missing_transcript:
        print("")
        print("%d wav files had no transcript entry and were skipped "
              "(first few: %s)" % (len(missing_transcript),
                                   ", ".join(missing_transcript[:5])))

    length_counts = Counter(len(r["text"]) for r in rows)
    print("")
    print("reference length: min %d, max %d chars"
          % (min(length_counts), max(length_counts)))
    print("wrote %s" % out_path)
    print("wrote %s" % summary_path)


if __name__ == "__main__":
    main()
