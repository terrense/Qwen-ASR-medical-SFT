# -*- coding: utf-8 -*-
"""Unit tests for the scoring stack.

Every reported CER, terminology figure and confidence interval flows through
these functions, so they are tested against hand-computed expectations rather
than against their own output.

Run:
    python tests/test_metrics.py        # no pytest needed
    pytest tests/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from augmentation import acoustic as A  # noqa: E402
from evaluation import metrics as M  # noqa: E402
from evaluation import normalization as N  # noqa: E402


def test_edit_distance_counts_operations():
    # identical
    assert M.edit_distance("abc", "abc") == (0, {"S": 0, "D": 0, "I": 0})
    # one substitution
    assert M.edit_distance("abc", "abd") == (1, {"S": 1, "D": 0, "I": 0})
    # one deletion (reference longer)
    assert M.edit_distance("abc", "ac") == (1, {"S": 0, "D": 1, "I": 0})
    # one insertion (hypothesis longer)
    assert M.edit_distance("ac", "abc") == (1, {"S": 0, "D": 0, "I": 1})
    # empty hypothesis: every reference character is a deletion
    assert M.edit_distance("abcd", "") == (4, {"S": 0, "D": 4, "I": 0})
    # empty reference: every hypothesis character is an insertion
    assert M.edit_distance("", "abc") == (3, {"S": 0, "D": 0, "I": 3})


def test_edit_distance_operation_counts_sum_to_distance():
    pairs = [("神经内科门诊在几楼", "神经外科门诊在几层"),
             ("增强ct", "ct"),
             ("我头疼三天了", "我头晕三天"),
             ("", "abc"), ("abc", "")]
    for ref, hyp in pairs:
        distance, counts = M.edit_distance(ref, hyp)
        assert distance == counts["S"] + counts["D"] + counts["I"], (ref, hyp)


def test_utterance_cer_and_empty_reference():
    record = M.utterance_cer("abcdef", "abcdeX")
    assert record["edit_distance"] == 1
    assert record["reference_length"] == 6
    assert abs(record["cer"] - 1 / 6) < 1e-12
    # an empty reference is not scorable and must not become 0.0 or 1.0
    assert M.utterance_cer("", "anything") is None


def test_corpus_cer_is_length_weighted_and_macro_is_not():
    # utterance 1: 1 error in 10 chars; utterance 2: 1 error in 2 chars
    records = [M.utterance_cer("a" * 9 + "b", "a" * 9 + "c"),
               M.utterance_cer("ab", "ax")]
    summary = M.corpus_cer(records)
    # corpus: 2 errors over 12 reference characters
    assert abs(summary["cer"] - 2 / 12) < 1e-12
    # macro: mean of 0.1 and 0.5
    assert abs(summary["macro_cer"] - 0.3) < 1e-12
    assert summary["n_utterances"] == 2


def test_corpus_cer_excludes_empty_references():
    records = [M.utterance_cer("abc", "abd"), None]
    summary = M.corpus_cer(records)
    assert summary["n_utterances"] == 1
    assert summary["n_empty_reference"] == 1


def test_normalization_default_rules():
    config = N.DEFAULT_CONFIG
    # language prefix stripped, full-width folded, punctuation removed,
    # lowercased, whitespace removed
    assert N.normalize("language Chinese<asr_text>不是ＣＴ，我说的是增强CT。",
                       config) == "不是ct我说的是增强ct"
    # numbers are NOT rewritten by default - that would hide real errors
    assert N.normalize("三十七度五", config) == "三十七度五"
    assert N.normalize(None, config) == ""


def test_normalization_number_conversion_is_opt_in():
    config = N.NormalizationConfig(normalize_numbers=True)
    assert N.normalize("三十七", config) == "37"
    assert N.normalize("一百二十", config) == "120"
    # decimals spoken with 点
    assert N.normalize("三十七点五", config) == "37.5"


def test_chinese_numerals_positional_vs_digitwise():
    """Two reading systems share the same characters and must not be confused.

    "三十七" is positional (37) because it contains a unit character; "二零一五"
    is read digit by digit (2015) because it does not. Applying the positional
    algorithm to a digit-wise run silently yields 5 instead of 2015, which is
    how this was originally wrong.
    """
    config = N.NormalizationConfig(normalize_numbers=True)
    # digit-wise: years, identifiers
    assert N.normalize("二零一五", config) == "2015"
    assert N.normalize("二零一五二零一六", config) == "20152016"
    # positional: quantities
    assert N.normalize("三十七", config) == "37"
    assert N.normalize("一百二十", config) == "120"
    assert N.normalize("两千零五", config) == "2005"
    assert N.normalize("十", config) == "10"
    # decimals keep their separator
    assert N.normalize("三十七点五", config) == "37.5"
    # units ride along untouched
    assert N.normalize("五百米", config) == "500米"


def test_digit_separators_survive_punctuation_stripping():
    """Measurements must not be corrupted by sentence-punctuation removal.

    Without this rule "37.5" collapses to "375" and "120/80" to "12080", which
    silently changes the value and can make a real recognition error score as
    correct.
    """
    config = N.DEFAULT_CONFIG
    assert N.normalize("体温37.5度。", config) == "体温37.5度"
    assert N.normalize("血压120/80，正常。", config) == "血压120/80正常"
    assert N.normalize("上午9:30的号", config) == "上午9:30的号"
    # sentence punctuation is still removed
    assert N.normalize("句号在这里。结束", config) == "句号在这里结束"
    # a period not between digits is still stripped
    assert N.normalize("等等.然后", config) == "等等然后"
    # turning the rule off reproduces the corruption, which documents its effect
    lossy = N.NormalizationConfig(keep_digit_separators=False)
    assert N.normalize("体温37.5度", lossy) == "体温375度"


def test_normalization_fingerprint_changes_with_rules():
    a = N.NormalizationConfig()
    b = N.NormalizationConfig(strip_punctuation=False)
    assert a.fingerprint() != b.fingerprint()


def test_lexicon_longest_match_masking():
    lexicon = M.MedicalLexicon({"imaging": ["增强ct", "ct"], "lab": ["hba1c"]})
    # "增强ct" must be consumed whole; the standalone "ct" is a separate hit
    found = lexicon.find("不是ct我说的是增强ct")
    assert found["增强ct"] == 1
    assert found["ct"] == 1
    # a lone 增强ct must not also count as a ct
    assert lexicon.find("增强ct") == {"增强ct": 1}


def test_lexicon_validate_reports_collisions():
    lexicon = M.MedicalLexicon({"a": ["ct"], "b": ["ct"]})
    assert "ct" in lexicon.validate()
    assert M.MedicalLexicon({"a": ["ct"], "b": ["mri"]}).validate() == {}


def test_term_metrics_multiset_and_error_list():
    lexicon = M.MedicalLexicon({"imaging": ["增强ct", "ct"]})
    ref = "不是ct我说的是增强ct"
    result = M.term_metrics(ref, "不是ct我说的是增加cd", lexicon)
    assert result["n_reference_terms"] == 2
    assert result["n_matched_terms"] == 1
    assert result["term_recall"] == 0.5
    assert result["entity_errors"] == ["增强ct"]
    assert result["all_terms_correct"] is False
    # perfect hypothesis
    perfect = M.term_metrics(ref, ref, lexicon)
    assert perfect["term_recall"] == 1.0 and perfect["all_terms_correct"]
    # utterance with no terms is excluded, not scored as 0 or 1
    assert M.term_metrics("你好", "你好", lexicon)["term_recall"] is None


def test_paired_bootstrap_detects_a_real_difference():
    # system A makes one error per utterance, system B makes none
    records_a, records_b = [], []
    for i in range(200):
        ref = "abcdefghij"
        rec_a = M.utterance_cer(ref, "Xbcdefghij")
        rec_b = M.utterance_cer(ref, ref)
        rec_a["utt_id"] = rec_b["utt_id"] = "u%03d" % i
        records_a.append(rec_a)
        records_b.append(rec_b)

    result = M.paired_bootstrap(records_a, records_b, n_samples=2000, seed=42)
    assert abs(result["cer_a"] - 0.1) < 1e-9
    assert result["cer_b"] == 0.0
    assert abs(result["cer_difference"] - 0.1) < 1e-9
    assert result["ci_lower"] > 0.0          # CI excludes zero
    assert result["significant_at_confidence"] is True


def test_paired_bootstrap_finds_no_difference_between_identical_systems():
    records = []
    for i in range(100):
        rec = M.utterance_cer("abcdefghij", "abcdefghiX")
        rec["utt_id"] = "u%03d" % i
        records.append(rec)
    import copy

    result = M.paired_bootstrap(records, copy.deepcopy(records),
                                n_samples=1000, seed=42)
    assert result["cer_difference"] == 0.0
    assert result["ci_lower"] == 0.0 and result["ci_upper"] == 0.0


def test_paired_bootstrap_rejects_unpaired_input():
    a = [dict(M.utterance_cer("abc", "abd"), utt_id="u1")]
    b = [dict(M.utterance_cer("abc", "abd"), utt_id="u2")]
    try:
        M.paired_bootstrap(a, b, n_samples=10)
    except ValueError as exc:
        assert "order differs" in str(exc)
    else:
        raise AssertionError("mismatched utt_ids must raise")

    # different reference lengths mean different normalization - must raise
    c = [dict(M.utterance_cer("abcd", "abcd"), utt_id="u1")]
    try:
        M.paired_bootstrap(a, c, n_samples=10)
    except ValueError as exc:
        assert "reference length differs" in str(exc)
    else:
        raise AssertionError("mismatched reference lengths must raise")


def test_grouped_cer_splits_by_metadata():
    records = []
    for condition, hyp in (("clean", "abc"), ("noise", "aXc"), ("noise", "aXX")):
        rec = M.utterance_cer("abc", hyp)
        rec["condition"] = condition
        records.append(rec)
    grouped = M.grouped_cer(records, "condition")
    assert grouped["clean"]["cer"] == 0.0
    # noise: 1 + 2 errors over 6 reference characters
    assert abs(grouped["noise"]["cer"] - 3 / 6) < 1e-12


def test_augmentation_snr_is_exact():
    sample_rate = 16000
    signal = (0.1 * np.sin(2 * np.pi * 220 *
                           np.arange(sample_rate) / sample_rate)).astype("float32")
    noise = np.random.default_rng(1).normal(0, 1, sample_rate).astype("float32")
    for requested in (5.0, 10.0, 20.0):
        scaled = A.scale_to_snr(signal, noise, requested)
        measured = 20 * np.log10(A.rms(signal) / A.rms(scaled))
        assert abs(measured - requested) < 1e-6, (requested, measured)


def test_augmentation_is_deterministic_given_seed_and_utt_id():
    bank = A.NoiseBank()
    signal = np.random.default_rng(0).normal(0, 0.05, 16000).astype("float32")
    first, meta_a = A.augment(signal, "utt-1", 42, bank)
    second, meta_b = A.augment(signal, "utt-1", 42, bank)
    assert np.array_equal(first, second)
    assert meta_a["condition"] == meta_b["condition"]
    # a different utterance id gives an independent draw
    third, _ = A.augment(signal, "utt-2", 42, bank)
    assert not np.array_equal(first, third)


def test_augmentation_condition_mix_matches_target():
    ids = ["utt%05d" % i for i in range(8000)]
    plan = A.plan_conditions(ids, 42)
    from collections import Counter

    counts = Counter(plan.values())
    for name, target in A.CONDITION_WEIGHTS.items():
        realized = counts[name] / len(ids)
        assert abs(realized - target) < 0.02, (name, realized, target)


def test_competing_speech_records_the_interferer():
    bank = A.NoiseBank()
    signal = np.random.default_rng(0).normal(0, 0.05, 16000).astype("float32")

    def provider(rng):
        return (np.random.default_rng(7).normal(0, 0.05, 16000).astype("float32"),
                {"utt_id": "other-9", "speaker_id": "SPK99"})

    _, meta = A.augment(signal, "utt-3", 42, bank, condition="competing_speech",
                        competing_provider=provider)
    assert meta["sir"] is not None
    assert A.SIR_RANGE[0] <= meta["sir"] <= A.SIR_RANGE[1]
    assert meta["competing_utt_id"] == "other-9"
    assert meta["competing_speaker_id"] == "SPK99"

    # without a provider the substitution is recorded, not hidden
    _, fallback = A.augment(signal, "utt-4", 42, bank,
                            condition="competing_speech")
    assert fallback["competing_speech_unavailable"] is True
    assert fallback["condition"] == "noise"


def main():
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, test in tests:
        try:
            test()
            print("  PASS  %s" % name)
        except Exception as exc:  # report every failure, do not stop at the first
            print("  FAIL  %s: %s" % (name, exc))
            failures.append((name, exc))
    print("")
    print("%d/%d passed" % (len(tests) - len(failures), len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
