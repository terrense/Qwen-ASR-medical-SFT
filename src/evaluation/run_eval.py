"""Phase 8 - Evaluation runner for Qwen3-ASR checkpoints.

Scores one model against one manifest and writes two artifacts:

    <outdir>/predictions.jsonl   one record per utterance, raw AND normalized
    <outdir>/metrics.json        aggregates + the exact settings that produced them

Experimental controls that are fixed here on purpose:

  context = ""      Qwen3-ASR accepts a biasing context string. Supplying
                    hospital vocabulary as context would improve CER without any
                    adaptation and would confound the component study, so the
                    context is empty for every condition and the value is
                    recorded in metrics.json.
  language          Forced to Chinese by default. Forcing the language makes the
                    model emit transcription text only, which keeps output
                    parsing identical across checkpoints. Recorded either way.

Nothing about decoding is allowed to differ between the systems being compared;
metrics.json carries the full decode settings so that can be verified after the
fact rather than assumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.manifest import read_manifest  # noqa: E402
from evaluation import normalization as norm  # noqa: E402
from evaluation import metrics as M  # noqa: E402


def load_model(model_path, adapter_path=None, dtype="bfloat16", device="cuda:0",
               max_batch=8):
    """Load the base model and, when given, attach a PEFT adapter."""
    import torch
    from qwen_asr import Qwen3ASRModel

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    asr = Qwen3ASRModel.from_pretrained(
        model_path, dtype=torch_dtype, device_map=device,
        max_inference_batch_size=max_batch)

    if adapter_path:
        from peft import PeftModel

        # The adapter was trained on the inner multimodal module, so it is
        # attached there rather than to the thin inference wrapper.
        asr.model = PeftModel.from_pretrained(asr.model, adapter_path)
        asr.model.eval()
        print("attached adapter: %s" % adapter_path)

    return asr


def transcribe_manifest(asr, rows, batch_size, language, context, log_every=50):
    """Run inference over the manifest, returning hypotheses in manifest order."""
    hypotheses = [None] * len(rows)
    decode_seconds = 0.0
    audio_seconds = 0.0

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        audios = [r["audio"] for r in chunk]
        t0 = time.time()
        try:
            results = asr.transcribe(audios, context=context, language=language)
            texts = [r.text for r in results]
        except Exception as exc:  # a bad file must not void the whole run
            print("  BATCH FAILED at %d-%d: %s" % (start, start + len(chunk), exc))
            texts = []
            for row in chunk:
                try:
                    one = asr.transcribe([row["audio"]], context=context,
                                         language=language)
                    texts.append(one[0].text)
                except Exception as inner:
                    print("    utterance %s failed: %s" % (row["utt_id"], inner))
                    texts.append("")
        decode_seconds += time.time() - t0
        audio_seconds += sum(r.get("duration", 0) or 0 for r in chunk)

        for offset, text in enumerate(texts):
            hypotheses[start + offset] = text

        done = min(start + batch_size, len(rows))
        if done % log_every < batch_size or done == len(rows):
            print("  transcribed %d/%d (%.1fs elapsed)" % (done, len(rows), decode_seconds))

    return hypotheses, decode_seconds, audio_seconds


def score(rows, hypotheses, lexicon, config):
    """Build per-utterance records carrying raw text, normalized text and metrics."""
    records = []
    for row, hyp_raw in zip(rows, hypotheses):
        ref_raw = row["text"]
        hyp_raw = hyp_raw if hyp_raw is not None else ""
        ref_norm = norm.normalize(ref_raw, config)
        hyp_norm = norm.normalize(hyp_raw, config)

        cer_rec = M.utterance_cer(ref_norm, hyp_norm)
        if cer_rec is None:  # empty reference after normalization
            cer_rec = {"edit_distance": None, "reference_length": 0,
                       "hypothesis_length": len(hyp_norm), "cer": None,
                       "substitutions": None, "deletions": None, "insertions": None}

        term_rec = (M.term_metrics(ref_norm, hyp_norm, lexicon) if lexicon
                    else {"medical_entities": [], "entity_errors": [],
                          "n_reference_terms": 0, "n_matched_terms": 0,
                          "term_recall": None, "term_error_rate": None,
                          "all_terms_correct": None})

        records.append(OrderedDict([
            ("utt_id", row["utt_id"]),
            ("reference_raw", ref_raw),
            ("hypothesis_raw", hyp_raw),
            ("reference_normalized", ref_norm),
            ("hypothesis_normalized", hyp_norm),
            ("edit_distance", cer_rec["edit_distance"]),
            ("reference_length", cer_rec["reference_length"]),
            ("hypothesis_length", cer_rec["hypothesis_length"]),
            ("cer", cer_rec["cer"]),
            ("substitutions", cer_rec["substitutions"]),
            ("deletions", cer_rec["deletions"]),
            ("insertions", cer_rec["insertions"]),
            ("medical_entities", term_rec["medical_entities"]),
            ("entity_errors", term_rec["entity_errors"]),
            ("n_reference_terms", term_rec["n_reference_terms"]),
            ("n_matched_terms", term_rec["n_matched_terms"]),
            ("term_recall", term_rec["term_recall"]),
            ("all_terms_correct", term_rec["all_terms_correct"]),
            ("condition", row.get("condition")),
            ("domain_category", row.get("domain_category")),
            ("template_family", row.get("template_family")),
            ("speaker_id", row.get("speaker_id")),
            ("source", row.get("source")),
            ("tts_engine", row.get("tts_engine")),
            ("snr", row.get("snr")),
            ("sir", row.get("sir")),
            ("duration", row.get("duration")),
        ]))
    return records


def aggregate(records, lexicon, meta):
    """Corpus metrics plus the breakdowns Phase 9 requires."""
    scorable = [r for r in records if r["reference_length"] > 0]
    out = OrderedDict()
    out["meta"] = meta
    out["overall"] = M.corpus_cer(scorable)
    out["by_condition"] = M.grouped_cer(scorable, "condition")
    out["by_domain_category"] = M.grouped_cer(scorable, "domain_category")
    out["by_source"] = M.grouped_cer(scorable, "source")
    out["medical_terms"] = M.corpus_term_metrics(records, lexicon)
    return out


def main():
    ap = argparse.ArgumentParser(description="Evaluate a Qwen3-ASR checkpoint.")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default=None,
                    help="Optional PEFT adapter directory")
    ap.add_argument("--manifest", required=True, help="JSONL test manifest")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--lexicon", default=None,
                    help="Medical lexicon JSON; term metrics are skipped without it")
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N utterances (smoke tests)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--language", default="Chinese",
                    help="Forced decode language; pass 'auto' to let the model decide")
    ap.add_argument("--context", default="",
                    help="Biasing context. Must stay empty for controlled comparisons")
    ap.add_argument("--tag", default=None, help="Human-readable run name")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    print("manifest: %s (%d utterances)" % (args.manifest, len(rows)))

    lexicon = None
    if args.lexicon and os.path.exists(args.lexicon):
        lexicon = M.MedicalLexicon.from_json(args.lexicon)
        collisions = lexicon.validate()
        print("lexicon : %d terms%s" % (
            len(lexicon.all_terms),
            ", %d category collisions" % len(collisions) if collisions else ""))
    elif args.lexicon:
        print("lexicon : %s NOT FOUND - term metrics will be null" % args.lexicon)

    config = norm.DEFAULT_CONFIG
    print(norm.describe(config))

    language = None if args.language.lower() == "auto" else args.language
    asr = load_model(args.model_path, args.adapter_path, args.dtype,
                     args.device, args.batch_size)

    t_start = time.time()
    hypotheses, decode_seconds, audio_seconds = transcribe_manifest(
        asr, rows, args.batch_size, language, args.context)
    wall_seconds = time.time() - t_start

    peak_vram_gib = None
    try:
        import torch

        if torch.cuda.is_available():
            peak_vram_gib = torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        pass

    records = score(rows, hypotheses, lexicon, config)

    meta = OrderedDict([
        ("tag", args.tag or Path(args.outdir).name),
        ("model_path", args.model_path),
        ("adapter_path", args.adapter_path),
        ("manifest", args.manifest),
        ("n_utterances", len(rows)),
        ("limit", args.limit),
        ("dtype", args.dtype),
        ("device", args.device),
        ("batch_size", args.batch_size),
        ("decode_language", args.language),
        ("decode_context", args.context),
        ("normalization", config.to_dict()),
        ("normalization_fingerprint", config.fingerprint()),
        ("lexicon_path", args.lexicon),
        ("lexicon_n_terms", len(lexicon.all_terms) if lexicon else 0),
        ("decode_seconds", round(decode_seconds, 2)),
        ("wall_seconds", round(wall_seconds, 2)),
        ("audio_seconds", round(audio_seconds, 2)),
        ("real_time_factor", round(decode_seconds / audio_seconds, 4) if audio_seconds else None),
        ("peak_vram_gib", round(peak_vram_gib, 3) if peak_vram_gib else None),
        ("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
    ])

    summary = aggregate(records, lexicon, meta)

    pred_path = outdir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (outdir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    overall = summary["overall"]
    print("")
    print("== %s ==" % meta["tag"])
    if overall["cer"] is None:
        print("no scorable utterances")
    else:
        print("CER            : %.4f  (%.2f%%)" % (overall["cer"], 100 * overall["cer"]))
        print("macro CER      : %.4f" % overall["macro_cer"])
        print("S/D/I          : %d / %d / %d" % (overall["substitutions"],
                                                 overall["deletions"],
                                                 overall["insertions"]))
        print("utterances     : %d" % overall["n_utterances"])
    terms = summary["medical_terms"]
    if terms["medical_term_error_rate"] is not None:
        print("med term error : %.4f  (recall %.4f over %d terms)"
              % (terms["medical_term_error_rate"], terms["medical_entity_recall"],
                 terms["n_reference_terms"]))
    if meta["real_time_factor"]:
        print("RTF            : %.4f" % meta["real_time_factor"])
    print("")
    print("wrote %s" % pred_path)
    print("wrote %s" % (outdir / "metrics.json"))


if __name__ == "__main__":
    main()
