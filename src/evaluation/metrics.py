"""Phase 9 - Metrics for Chinese hospital-domain ASR.

Implements, with no third-party dependencies:

  * character edit distance with substitution/deletion/insertion breakdown
  * corpus-level and macro-averaged CER
  * medical-term error rate and medical-entity recall / exact match
  * grouped CER (by domain category, by acoustic condition, by any manifest key)
  * paired bootstrap resampling with a 95% confidence interval (Phase 18)

Conventions fixed here and used everywhere downstream:

  Corpus CER = sum(edit_distance) / sum(len(reference))
      This is the primary number. It weights utterances by length, which is the
      standard definition and is what the bootstrap resamples.

  Macro CER = mean(per-utterance CER)
      Reported alongside because a corpus CER can be dominated by a few long
      utterances. Per-utterance CER is clipped at 1.0 only for the macro average
      (insertions can otherwise push a single utterance above 100%); the corpus
      figure is never clipped.

  Empty references are excluded from CER and counted separately, never treated
  as CER 0.0 or 1.0.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter, OrderedDict

SUB, DEL, INS, OK = "S", "D", "I", "="


def edit_distance(ref, hyp):
    """Levenshtein distance with an error-type breakdown.

    Returns ``(distance, {"S": n, "D": n, "I": n})``. Ties are resolved in the
    order substitution > deletion > insertion, which is the conventional choice
    and keeps the breakdown deterministic across runs.
    """
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, {SUB: 0, DEL: 0, INS: m}
    if m == 0:
        return n, {SUB: 0, DEL: n, INS: 0}

    # Full matrix: utterances are short, and the backtrace needs it.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        ref_char = ref[i - 1]
        row, prev_row = dp[i], dp[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_char == hyp[j - 1] else 1
            row[j] = min(prev_row[j - 1] + cost, prev_row[j] + 1, row[j - 1] + 1)

    counts = {SUB: 0, DEL: 0, INS: 0}
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost:
                    counts[SUB] += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            counts[DEL] += 1
            i -= 1
            continue
        counts[INS] += 1
        j -= 1
    return dp[n][m], counts


def utterance_cer(ref, hyp):
    """Per-utterance CER record. Returns None when the reference is empty."""
    if not ref:
        return None
    dist, counts = edit_distance(ref, hyp)
    return {
        "edit_distance": dist,
        "reference_length": len(ref),
        "hypothesis_length": len(hyp),
        "cer": dist / len(ref),
        "substitutions": counts[SUB],
        "deletions": counts[DEL],
        "insertions": counts[INS],
    }


def corpus_cer(records):
    """Aggregate per-utterance records into corpus and macro CER."""
    scored = [r for r in records if r is not None and r["reference_length"] > 0]
    if not scored:
        return {"n_utterances": 0, "cer": None, "macro_cer": None,
                "n_empty_reference": len(records) - len(scored)}
    total_dist = sum(r["edit_distance"] for r in scored)
    total_len = sum(r["reference_length"] for r in scored)
    return {
        "n_utterances": len(scored),
        "n_empty_reference": len(records) - len(scored),
        "total_edit_distance": total_dist,
        "total_reference_length": total_len,
        "cer": total_dist / total_len,
        "macro_cer": sum(min(r["cer"], 1.0) for r in scored) / len(scored),
        "substitutions": sum(r["substitutions"] for r in scored),
        "deletions": sum(r["deletions"] for r in scored),
        "insertions": sum(r["insertions"] for r in scored),
    }


def grouped_cer(records, keys):
    """CER broken down by a metadata field (domain_category, condition, ...).

    ``records`` are dicts that carry both the scoring fields and the metadata.
    Groups are returned sorted by descending utterance count.
    """
    buckets = {}
    for rec in records:
        value = rec.get(keys)
        if value is None:
            value = "UNSPECIFIED"
        buckets.setdefault(value, []).append(rec)
    out = OrderedDict()
    for name in sorted(buckets, key=lambda k: (-len(buckets[k]), str(k))):
        out[str(name)] = corpus_cer(buckets[name])
    return out


class MedicalLexicon:
    """Medical term inventory with a category map.

    Built independently of any test prediction (Phase 9 requirement): the term
    list is derived from the corpus generation vocabulary and curated clinical
    term lists, never from inspecting model outputs.
    """

    def __init__(self, terms_by_category):
        self.terms_by_category = {
            cat: sorted(set(terms), key=len, reverse=True)
            for cat, terms in terms_by_category.items()
        }
        self.category_of = {}
        for cat, terms in self.terms_by_category.items():
            for term in terms:
                # First category wins; categories are curated to be disjoint and
                # any collision is surfaced by validate().
                self.category_of.setdefault(term, cat)
        # Longest-first so that "增强ct" is matched before "ct".
        self.all_terms = sorted(self.category_of, key=len, reverse=True)

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload["terms_by_category"])

    def to_json(self, path, extra=None):
        payload = {"terms_by_category": self.terms_by_category,
                   "n_terms": len(self.all_terms)}
        if extra:
            payload.update(extra)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def validate(self):
        """Report terms that appear under more than one category."""
        seen = {}
        collisions = {}
        for cat, terms in self.terms_by_category.items():
            for term in terms:
                if term in seen and seen[term] != cat:
                    collisions.setdefault(term, {seen[term]}).add(cat)
                seen[term] = cat
        return {t: sorted(c) for t, c in collisions.items()}

    def find(self, text):
        """Count non-overlapping term occurrences, longest match first.

        Matched spans are masked out so that a hit on "增强ct" does not also
        produce a hit on the substring "ct".
        """
        if not text:
            return Counter()
        masked = list(text)
        found = Counter()
        for term in self.all_terms:
            if not term:
                continue
            start = 0
            while True:
                idx = "".join(masked).find(term, start)
                if idx < 0:
                    break
                found[term] += 1
                for k in range(idx, idx + len(term)):
                    masked[k] = "\x00"
                start = idx + len(term)
        return found


def term_metrics(ref_norm, hyp_norm, lexicon):
    """Medical term recall / exact-match for one utterance.

    Multiset matching: a term occurring twice in the reference and once in the
    hypothesis scores one hit and one miss. Substring matching on the normalized
    strings is used rather than alignment, because a term recognized in the
    wrong position is still a terminology success and a CER problem, and the two
    are reported separately on purpose.
    """
    ref_terms = lexicon.find(ref_norm)
    hyp_terms = lexicon.find(hyp_norm)
    total = sum(ref_terms.values())
    if total == 0:
        return {"n_reference_terms": 0, "n_matched_terms": 0,
                "term_recall": None, "term_error_rate": None,
                "all_terms_correct": None, "medical_entities": [],
                "entity_errors": []}

    matched = 0
    errors = []
    for term, count in ref_terms.items():
        hit = min(count, hyp_terms.get(term, 0))
        matched += hit
        if hit < count:
            errors.extend([term] * (count - hit))
    return {
        "n_reference_terms": total,
        "n_matched_terms": matched,
        "term_recall": matched / total,
        "term_error_rate": 1.0 - matched / total,
        "all_terms_correct": matched == total,
        "medical_entities": sorted(ref_terms.elements()),
        "entity_errors": sorted(errors),
    }


def corpus_term_metrics(records, lexicon=None):
    """Aggregate medical-term statistics, overall and per term category."""
    scored = [r for r in records if r.get("n_reference_terms")]
    total_ref = sum(r["n_reference_terms"] for r in scored)
    total_hit = sum(r["n_matched_terms"] for r in scored)
    exact = [r for r in scored if r.get("all_terms_correct")]

    out = {
        "n_utterances_with_terms": len(scored),
        "n_reference_terms": total_ref,
        "n_matched_terms": total_hit,
        "medical_term_error_rate": (1.0 - total_hit / total_ref) if total_ref else None,
        "medical_entity_recall": (total_hit / total_ref) if total_ref else None,
        "utterance_exact_match": (len(exact) / len(scored)) if scored else None,
    }

    if lexicon is not None:
        per_cat_ref, per_cat_hit = Counter(), Counter()
        for rec in scored:
            missed = Counter(rec.get("entity_errors", []))
            for term in rec.get("medical_entities", []):
                cat = lexicon.category_of.get(term, "UNCATEGORIZED")
                per_cat_ref[cat] += 1
                if missed[term] > 0:
                    missed[term] -= 1
                else:
                    per_cat_hit[cat] += 1
        out["by_term_category"] = OrderedDict(
            (cat, {
                "n_reference_terms": per_cat_ref[cat],
                "n_matched_terms": per_cat_hit[cat],
                "term_error_rate": 1.0 - per_cat_hit[cat] / per_cat_ref[cat],
            })
            for cat in sorted(per_cat_ref, key=lambda c: -per_cat_ref[c])
        )
    return out


def paired_bootstrap(records_a, records_b, n_samples=10000, seed=42,
                     confidence=0.95):
    """Paired bootstrap over utterances for CER_a - CER_b (Phase 18).

    Both systems must be scored on the *same* utterances in the same order;
    this is asserted rather than assumed. Each bootstrap replicate resamples
    utterance indices with replacement and recomputes corpus CER for both
    systems from the same index set, which preserves the pairing.

    Returns the observed difference, the CI, and the two-sided p-value for
    "the systems are equal", estimated as the fraction of replicates whose
    difference falls on the opposite side of zero from the observed one.
    """
    if len(records_a) != len(records_b):
        raise ValueError("paired bootstrap needs equal-length record lists "
                         "(%d vs %d)" % (len(records_a), len(records_b)))

    pairs = []
    for rec_a, rec_b in zip(records_a, records_b):
        if rec_a.get("utt_id") and rec_b.get("utt_id") and rec_a["utt_id"] != rec_b["utt_id"]:
            raise ValueError("utterance order differs: %s vs %s"
                             % (rec_a["utt_id"], rec_b["utt_id"]))
        if rec_a["reference_length"] != rec_b["reference_length"]:
            raise ValueError(
                "reference length differs for %s (%d vs %d): the two systems "
                "were scored with different normalization"
                % (rec_a.get("utt_id"), rec_a["reference_length"],
                   rec_b["reference_length"]))
        if rec_a["reference_length"] > 0:
            pairs.append((rec_a["edit_distance"], rec_b["edit_distance"],
                          rec_a["reference_length"]))

    if not pairs:
        return {"error": "no scorable utterances"}

    def cer_of(indices):
        dist_a = dist_b = length = 0
        for idx in indices:
            d_a, d_b, ref_len = pairs[idx]
            dist_a += d_a
            dist_b += d_b
            length += ref_len
        return dist_a / length, dist_b / length

    observed_a, observed_b = cer_of(range(len(pairs)))
    observed_diff = observed_a - observed_b

    rng = random.Random(seed)
    n = len(pairs)
    diffs = []
    for _ in range(n_samples):
        indices = [rng.randrange(n) for _ in range(n)]
        cer_a, cer_b = cer_of(indices)
        diffs.append(cer_a - cer_b)
    diffs.sort()

    alpha = (1.0 - confidence) / 2.0
    lo = diffs[max(0, int(math.floor(alpha * n_samples)))]
    hi = diffs[min(n_samples - 1, int(math.ceil((1 - alpha) * n_samples)) - 1)]

    if observed_diff >= 0:
        p_value = 2.0 * sum(1 for d in diffs if d <= 0.0) / n_samples
    else:
        p_value = 2.0 * sum(1 for d in diffs if d >= 0.0) / n_samples
    p_value = min(1.0, p_value)

    return {
        "n_utterances": n,
        "n_bootstrap_samples": n_samples,
        "seed": seed,
        "confidence": confidence,
        "cer_a": observed_a,
        "cer_b": observed_b,
        "cer_difference": observed_diff,
        "ci_lower": lo,
        "ci_upper": hi,
        "significant_at_confidence": not (lo <= 0.0 <= hi),
        "p_value": p_value,
    }
