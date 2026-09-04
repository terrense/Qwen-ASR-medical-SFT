# -*- coding: utf-8 -*-
"""Phase 5 - Synthetic speaker identities.

Identities are *designed*, not cloned. Each one is a natural-language timbre
description handed to Qwen3-TTS VoiceDesign
(``Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign``). No reference audio from a real
person is ever used, so no identifiable voice is reproduced.

Keeping an identity stable across thousands of utterances needs one extra step,
because VoiceDesign re-interprets its instruction on every call and would drift.
The procedure is:

  1. VoiceDesign renders one *anchor* utterance from the identity's description.
  2. ``create_voice_clone_prompt(ref_audio=anchor, x_vector_only_mode=True)``
     extracts a speaker embedding from that anchor.
  3. Every later utterance for that identity is produced with
     ``generate_voice_clone(...)`` using the stored embedding.

The anchor is a synthetic voice, so step 2 clones a generated identity rather
than a person. Anchors are written to disk and their checksums recorded, so a
regenerated corpus can be proven to use the same voices.

Pools are disjoint by construction:

    train 20 identities | dev 6 | test 6

Dev and test voices are never heard during training, which is what makes the
speaker-disjointness requirement of Phase 4 hold. The cross-TTS test uses
CosyVoice3 and is disjoint by engine as well.
"""
from __future__ import annotations

import random
from collections import Counter, OrderedDict

GENDERS = ["女声", "男声"]
AGE_STYLES = ["青年", "中年", "年长"]
RATES = [("偏慢", 0.88), ("中等", 1.0), ("偏快", 1.12)]
PITCHES = ["低沉", "中等", "明亮"]
ACCENTS = [
    "标准普通话",
    "带一点南方口音的普通话",
    "带一点北方口音的普通话",
    "带一点西南官话口音的普通话",
]
ENERGIES = ["平稳", "温和", "有力"]

# Anchor sentences are generic and domain-neutral on purpose: the speaker
# embedding should encode the voice, not hospital vocabulary.
ANCHOR_TEXTS = [
    "今天天气还不错，我们出去走一走吧。",
    "这件事情我需要再考虑一下，明天给你答复。",
    "请把桌上的那本书递给我，谢谢。",
    "我早上一般七点起床，然后去公园散步。",
]

N_TRAIN, N_DEV, N_TEST = 20, 6, 6
N_SPEAKERS = N_TRAIN + N_DEV + N_TEST


def _instruct(gender, age, rate_name, pitch, accent, energy):
    """Natural-language timbre description for VoiceDesign."""
    return ("一位%s%s，音色%s，语速%s，%s，说话语气%s，吐字清晰自然。"
            % (age, gender, pitch, rate_name, accent, energy))


def build_speaker_inventory(seed=42):
    """Construct the full inventory deterministically.

    Attributes are laid out with coprime strides rather than sampled, so the
    inventory is balanced and reproducible by inspection: every gender, age,
    rate, pitch, accent and energy value appears a predictable number of times,
    and no attribute correlates with the pool an identity lands in.
    """
    speakers = []
    for i in range(N_SPEAKERS):
        gender = GENDERS[i % 2]
        age = AGE_STYLES[i % 3]
        rate_name, rate_scale = RATES[(i // 2) % 3]
        pitch = PITCHES[(i // 3) % 3]
        accent = ACCENTS[(i * 3) % 4]
        energy = ENERGIES[(i // 4) % 3]

        speakers.append(OrderedDict([
            ("speaker_id", "SPK%02d" % i),
            ("gender", gender),
            ("age_style", age),
            ("rate_name", rate_name),
            ("rate_scale", rate_scale),
            ("pitch", pitch),
            ("accent", accent),
            ("energy", energy),
            ("instruct", _instruct(gender, age, rate_name, pitch, accent, energy)),
            ("anchor_text", ANCHOR_TEXTS[i % len(ANCHOR_TEXTS)]),
            ("tts_engine", "qwen3-tts-12hz-1.7b-voicedesign"),
        ]))

    # Pool assignment is stratified, not random. With only six dev and six test
    # voices, a random draw regularly leaves an accent absent from the held-out
    # pools, which would make any accent-related robustness claim untestable.
    # Instead dev and test are filled by cycling through accents and alternating
    # gender, so each held-out pool covers all four accent styles.
    rng = random.Random(seed)
    remaining = {s["speaker_id"]: s for s in speakers}
    by_accent = {}
    for speaker in speakers:
        by_accent.setdefault(speaker["accent"], []).append(speaker)
    for bucket in by_accent.values():
        rng.shuffle(bucket)

    def take(pool_name, count):
        chosen = []
        accents = sorted(by_accent)
        want_gender = 0
        index = 0
        while len(chosen) < count:
            accent = accents[index % len(accents)]
            index += 1
            bucket = by_accent[accent]
            pick = None
            # Prefer the gender that keeps the pool balanced, but never stall.
            for candidate in bucket:
                if candidate["speaker_id"] not in remaining:
                    continue
                if candidate["gender"] == GENDERS[want_gender % 2]:
                    pick = candidate
                    break
            if pick is None:
                for candidate in bucket:
                    if candidate["speaker_id"] in remaining:
                        pick = candidate
                        break
            if pick is None:
                if index > len(accents) * (count + len(speakers)):
                    raise RuntimeError("ran out of speakers building pool %s" % pool_name)
                continue
            pick["pool"] = pool_name
            del remaining[pick["speaker_id"]]
            chosen.append(pick)
            want_gender += 1
        return chosen

    take("test", N_TEST)
    take("dev", N_DEV)
    for speaker in remaining.values():
        speaker["pool"] = "train"

    return speakers


def pool_members(speakers, pool):
    return [s for s in speakers if s["pool"] == pool]


def inventory_report(speakers):
    """Balance report - checked before any audio is generated."""
    report = OrderedDict()
    report["n_speakers"] = len(speakers)
    report["by_pool"] = dict(Counter(s["pool"] for s in speakers))
    for attribute in ("gender", "age_style", "rate_name", "pitch", "accent", "energy"):
        report["by_" + attribute] = dict(Counter(s[attribute] for s in speakers))
    report["pool_composition"] = {
        pool: {
            "gender": dict(Counter(s["gender"] for s in pool_members(speakers, pool))),
            "age_style": dict(Counter(s["age_style"] for s in pool_members(speakers, pool))),
            "accent": dict(Counter(s["accent"] for s in pool_members(speakers, pool))),
        }
        for pool in ("train", "dev", "test")
    }
    return report


def assign_speakers(script_ids, speakers, pool, seed=42, max_share=None):
    """Assign a speaker to every script so no voice dominates the split.

    Scripts are dealt round-robin over a shuffled speaker list, which bounds the
    per-speaker count to within one utterance of ``len(scripts)/len(pool)``.
    ``max_share`` (a fraction) is verified afterwards and raises if violated,
    so a future change to this function cannot quietly unbalance the corpus.
    """
    members = pool_members(speakers, pool)
    if not members:
        raise ValueError("speaker pool '%s' is empty" % pool)

    rng = random.Random(seed)
    ordered = sorted(script_ids)
    rng.shuffle(ordered)

    rotation = list(members)
    rng.shuffle(rotation)

    assignment = OrderedDict()
    for position, script_id in enumerate(ordered):
        assignment[script_id] = rotation[position % len(rotation)]["speaker_id"]

    counts = Counter(assignment.values())
    if max_share is not None and counts:
        top = max(counts.values()) / len(assignment)
        if top > max_share:
            raise AssertionError(
                "speaker %s covers %.3f of pool '%s', above the %.3f cap"
                % (counts.most_common(1)[0][0], top, pool, max_share))
    return assignment, dict(counts)
