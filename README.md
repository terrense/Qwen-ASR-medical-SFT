# Where to Adapt Qwen3-ASR?

Component-wise parameter-efficient adaptation for low-resource Chinese
hospital-domain speech recognition.

The question this repository answers is **not** whether fine-tuning Qwen3-ASR
helps. It is: *under limited target-domain supervision, which component should
be adapted* — the audio encoder body, the audio projection head, or the language
decoder?

Current progress, failures and pending work: **[EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md)**.

---

## The component split

Qwen3-ASR is divided into three adaptable components. The boundaries are
derived from the loaded module graph by `src/models/components.py`, not
hard-coded from module names, so an upstream rename surfaces as a failed
assertion instead of a silently wrong experiment.

| Component | How it is identified | Qwen3-ASR-0.6B modules |
|---|---|---|
| `AUDIO_ENCODER` | everything under the submodule whose class ends in `AudioEncoder` | 18 transformer layers (`q/k/v/out_proj`, `fc1`, `fc2`), conv front-end |
| `AUDIO_PROJECTION` | `nn.Linear` children of the audio root whose `in_features == d_model` | `audio_tower.proj1` (896→896), `audio_tower.proj2` (896→1024) |
| `TEXT_DECODER` | everything under the submodule whose class ends in `TextModel`, plus `lm_head` | 28 Qwen3 layers (`q/k/v/o_proj`, MLP), `lm_head` |

### The targeting hazard this design exists to avoid

The audio tower and the text decoder **both** define `q_proj`, `k_proj` and
`v_proj`. Writing

```python
LoraConfig(target_modules=["q_proj", "v_proj"])   # WRONG for this study
```

adapts *both* branches at once, which makes "audio-only LoRA" and "text-only
LoRA" the same experiment. This repository never targets by leaf name. It emits
full-path regexes and asserts they are disjoint against the live inventory:

```
^thinker\.audio_tower\.layers\.\d+\.self_attn\.(k_proj|out_proj|q_proj|v_proj)$
^thinker\.model\.layers\.\d+\.self_attn\.(k_proj|o_proj|q_proj|v_proj)$
```

Note also that the attention output projection is `out_proj` in the audio tower
and `o_proj` in the decoder. Both spellings are discovered from the inventory.

---

## Adaptation arms (Phase 10)

| Arm | Audio LoRA | Projection | Text LoRA | Full SFT |
|---|:--:|:--:|:--:|:--:|
| `A0_zero_shot` | | | | |
| `A1_full_sft` | | ✓ | | ✓ |
| `A2_audio_lora` | ✓ | | | |
| `A3_projection_only` | | ✓ | | |
| `A4_text_lora` | | | ✓ | |
| `A5_audio_lora_proj` | ✓ | ✓ | | |
| `A6_text_lora_proj` | | ✓ | ✓ | |
| `A7_dualpeft` | ✓ | ✓ | ✓ | |

LoRA defaults: `r=16`, `alpha=32`, `dropout=0.05`, attention projections only
(no MLP modules in the initial grid).

---

## Layout

```
configs/                 one YAML per experiment (Phase 16)
data/
  scripts/               all_scripts.jsonl + generation report + rejections
  manifests/splits/      family-disjoint train/dev/test script lists
  synthetic/             generated audio
  public/                AISHELL-1 and other public sets
  medical_lexicon.json   417 curated terms, 7 categories
src/
  data/                  manifest schema, vocabulary, corpus generator
  models/                component identification + adaptation arms
  training/              training entry points
  evaluation/            normalization, metrics, eval runner
  augmentation/          deterministic acoustic augmentation
scripts/                 CLI entry points
tests/                   trainable-parameter safety check (mandatory)
experiments/             one directory per run
results/                 predictions, metrics, figures, tables
```

---

## Pipeline

```bash
# Phase 0 - record the environment
python scripts/audit_environment.py

# Phase 1 - module inventory and verified LoRA targets
python scripts/inspect_model.py --model_path models/Qwen3-ASR-0.6B --device cpu

# Phase 3 - text corpus
python scripts/generate_scripts.py --total 18000 --seed 42

# Phase 4 - family-disjoint split
python scripts/split_data.py --seed 42

# Phase 9 - medical entity lexicon
python scripts/build_medical_lexicon.py

# Phase 5 - synthesize audio (env_tts). Anchors first, then the corpus.
sh scripts/setup_env_tts.sh
env_tts/bin/python scripts/generate_tts.py --stage anchors
env_tts/bin/python scripts/generate_tts.py --stage corpus --split train

# Phase 4b - nested duration budgets, D1 subset of D5 subset of D10 subset of D20
python scripts/build_duration_subsets.py --manifest data/manifests/train_synthetic.jsonl

# Phase 7 - acoustic augmentation (deterministic on seed)
python scripts/augment_corpus.py \
    --manifest data/manifests/train_20h.jsonl \
    --outdir data/synthetic/audio/train_aug \
    --out_manifest data/manifests/train_20h_aug.jsonl

# Phase 14 - general-domain control set
python scripts/prepare_aishell.py --root /path/to/aishell1 --split test

# Phase 11 - MANDATORY before any training job
python tests/test_trainable_parameters.py \
    --model_path models/Qwen3-ASR-0.6B --arm all --outdir experiments/safety

# Phase 16 - run one experiment end to end
python scripts/make_configs.py
python scripts/run_experiment.py configs/qwen06_dualpeft_20h.yaml

# Phase 8 - evaluate a single checkpoint directly
python src/evaluation/run_eval.py \
    --model_path models/Qwen3-ASR-0.6B \
    --manifest data/manifests/test_synthetic.jsonl \
    --lexicon data/medical_lexicon.json \
    --outdir experiments/qwen06_zero/test_synthetic

# Phases 19 and 20 - figures and tables (XX wherever a run is missing)
python scripts/make_figures.py
python scripts/make_tables.py

# Scoring-stack regression tests
python tests/test_metrics.py
```

---

## Measurement conventions

These are fixed once, applied everywhere, and recorded in every `metrics.json`.

**Normalization** (`src/evaluation/normalization.py`) — each rule is a named
flag, and the active set is fingerprinted into every metrics file. Defaults:
NFKC on, punctuation stripped, lowercase on, whitespace removed, the
`language X<asr_text>` prompt prefix stripped. Traditional→Simplified
conversion and Chinese-numeral→Arabic conversion are **off** by default: both
are semantic rewrites that can hide genuine recognition errors. A
number-normalized CER is reported separately when needed.

**CER** — corpus CER is `sum(edit_distance) / sum(len(reference))`. Macro CER
(mean of per-utterance CER, clipped at 1.0) is reported alongside because a
corpus figure can be dominated by long utterances. Empty references are
excluded and counted, never scored as 0.0 or 1.0.

**Terminology** — medical-term error rate uses multiset matching over the
lexicon, with longest-match-first masking so a hit on `增强ct` does not also
count as a hit on `ct`. CER and terminology are reported separately on purpose:
an utterance can score 16.7% CER while missing 50% of its medical terms, and
that gap is a finding, not noise.

**Decoding controls** — Qwen3-ASR accepts a biasing `context` string. It is
empty for every condition in every comparison, because hospital vocabulary
supplied as context would improve CER with no adaptation at all and confound the
component study. The value is written into every `metrics.json` so this can be
checked after the fact rather than trusted.

**Statistics** — paired bootstrap over utterances, ≥10,000 replicates, 95% CI
(`metrics.paired_bootstrap`). Pairing is asserted, not assumed: the two systems
must cover the same `utt_id`s in the same order with identical reference
lengths, otherwise the call raises.

---

## Data provenance

All 17,986 utterances are generated from hand-written spoken-Mandarin patterns
filled from a curated hospital vocabulary (`src/data/hospital_vocab.py`).
Nothing is copied from web consultations, and no entry refers to a real person,
hospital or case. Synthetic speaker identities are generated voices, not clones
of identifiable people.

Splits are disjoint by **template family** — a semantic sentence pattern — not
merely by waveform, so a test utterance is never a re-voicing of a pattern seen
in training. Both properties are asserted at split time.
