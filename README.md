<div align="center">

# Where to Adapt Qwen3-ASR?

**Component-wise parameter-efficient adaptation for low-resource Chinese hospital-domain speech recognition**

组件级参数高效适配 · 低资源中文医院域语音识别

[![Status](https://img.shields.io/badge/status-work%20in%20progress-orange)](EXPERIMENT_STATUS.md)
[![Model](https://img.shields.io/badge/model-Qwen3--ASR%200.6B%20%2F%201.7B-blue)](https://github.com/QwenLM)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA%20r16-8A2BE2)](configs/)
[![Python](https://img.shields.io/badge/python-3.12-3776AB)](https://www.python.org/)
[![torch](https://img.shields.io/badge/torch-2.9.1%2Bcu128-EE4C2C)](https://pytorch.org/)
[![transformers](https://img.shields.io/badge/transformers-4.57.6-FFD21E)](https://github.com/huggingface/transformers)

**[Experiment log](EXPERIMENT_STATUS.md)** · [Component split](#the-component-split) · [Arms](#adaptation-arms) · [Results](#results) · [Reproduce](#reproduce) · [Team](#team)

</div>

---

## TL;DR

The question is **not** whether fine-tuning Qwen3-ASR helps. It is: *under
limited target-domain supervision, which component should be adapted* — the
audio encoder body, the audio projection head, or the language decoder?

Qwen3-ASR-0.6B splits **23.06% / 0.22% / 76.17%** across those three. The
projection head is **1.7 M parameters, 0.22% of the model**. If most of the
hospital-domain gain lives there, domain adaptation costs almost nothing to
train, store and serve — one adapter per department instead of one model.

Everything here is built so that a wrong answer is visible rather than
plausible: component boundaries are derived from the live module graph,
gradient isolation is asserted per arm before every run, and every failure —
including the ones that invalidated a completed sweep — is written down in
[EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).

> **Current state.** General-domain zero-shot baselines are final. The first
> seven-arm sweep ran on 2026-09-04 and was then **invalidated** by a label-mask
> bug (failure #15); it is being repeated against the corrected objective. No
> adaptation result should be cited until that rerun lands.

---

## The component split

The boundaries are derived from the loaded module graph by
`src/models/components.py`, not hard-coded from module names, so an upstream
rename surfaces as a failed assertion instead of a silently wrong experiment.

| Component | How it is identified | Qwen3-ASR-0.6B modules | Share of parameters |
|---|---|---|--:|
| `AUDIO_ENCODER` | everything under the submodule whose class ends in `AudioEncoder` | 18 transformer layers (`q/k/v/out_proj`, `fc1`, `fc2`), conv front-end | 23.06% |
| `AUDIO_PROJECTION` | `nn.Linear` children of the audio root whose `in_features == d_model` | `audio_tower.proj1` (896→896), `audio_tower.proj2` (896→1024) | **0.22%** |
| `TEXT_DECODER` | everything under the submodule whose class ends in `TextModel`, plus `lm_head` | 28 Qwen3 layers (`q/k/v/o_proj`, MLP), `lm_head` | 76.17% |

Totals: 782,426,112 parameters for 0.6B; 2,038,052,480 for 1.7B, where the
projection head is smaller still at **0.155%**.

### The targeting hazard this design exists to avoid

The audio tower and the text decoder **both** define `q_proj`, `k_proj` and
`v_proj`. Writing

```python
LoraConfig(target_modules=["q_proj", "v_proj"])   # WRONG for this study
```

hits 92 modules across both branches on 0.6B (104 on 1.7B), which makes
"audio-only LoRA" and "text-only LoRA" the same experiment. This repository
never targets by leaf name. It emits full-path regexes and asserts they are
disjoint against the live inventory:

```
^thinker\.audio_tower\.layers\.\d+\.self_attn\.(k_proj|out_proj|q_proj|v_proj)$
^thinker\.model\.layers\.\d+\.self_attn\.(k_proj|o_proj|q_proj|v_proj)$
```

Note also that the attention output projection is `out_proj` in the audio tower
and `o_proj` in the decoder. Both spellings are discovered from the inventory.

Isolation is then **proved per arm, per run**: A2 touches 0 decoder tensors, A4
touches 0 encoder tensors, and a run aborts if any unintended parameter
receives gradient.

---

## Adaptation arms

| Arm | Audio LoRA | Projection | Text LoRA | Full SFT | Trainable (0.6B) |
|---|:--:|:--:|:--:|:--:|--:|
| `A0_zero_shot` | | | | | — |
| `A1_full_sft` | | ✓ | | ✓ | 782,426,112 · 100% |
| `A2_audio_lora` | ✓ | | | | 2,064,384 · 0.263% |
| `A3_projection_only` | | ✓ | | | **1,722,240 · 0.220%** |
| `A4_text_lora` | | | ✓ | | 4,587,520 · 0.583% |
| `A5_audio_lora_proj` | ✓ | ✓ | | | 3,786,624 · 0.483% |
| `A6_text_lora_proj` | | ✓ | ✓ | | 6,309,760 · 0.802% |
| `A7_dualpeft` | ✓ | ✓ | ✓ | | 8,374,144 · 1.061% |

LoRA defaults: `r=16`, `alpha=32`, `dropout=0.05`, attention projections only
(no MLP modules in the initial grid). Counts are verified arithmetically:
A2 = 72 modules × (16×896 + 896×16); A4 = 28 layers × 163,840;
A7 = A2 + A4 + projection.

---

## Results

### Zero-shot baselines — final

AISHELL-1 test, 7,176 utterances / 10.03 h / 20 speakers, full set, no subsampling.

| Model | CER | macro CER |
|---|--:|--:|
| Qwen3-ASR-0.6B | 2.10% | — |
| Qwen3-ASR-1.7B | **1.54%** | — |

Paired bootstrap over utterances, 10,000 replicates: the 1.7B advantage is
**+0.56 pp, 95% CI [+0.48, +0.65], p < 0.0001**. Pairing is asserted, not
assumed. This is the reference point for catastrophic forgetting.

### Adaptation sweep — rerunning

The 2026-09-04 seven-arm sweep at the 1 h general budget completed and was then
discarded. Every arm scored worse than zero-shot, and the losses sat near
`ln(151936) = 11.93` — the cross-entropy of a uniform guess over the vocabulary.

Root cause: `Qwen3ASRProcessorKwargs` sets `padding_side="left"`, overriding the
tokenizer's own `"right"`. The collator masked labels with a **left-anchored**
slice, correct only under right padding. Under left padding that slice landed on
padding, leaving the entire prefix supervised — chat scaffolding and every one
of the 26–54 `<|audio_pad|>` tokens. The model was being trained to predict the
audio placeholders.

| Batch | Padding | Loss before fix | Loss after fix |
|---|---|--:|--:|
| n=1, each of 8 rows | none | 4.330 (mean) | 4.330 (mean) |
| n=2, rows 0–1 (120 / 118 tokens) | 2 tokens | 3.863 | 3.863 |
| n=8, rows 0–7 (120 … 60 tokens) | up to 60 tokens | **16.434** | **4.077** |

The safety check had missed it for a specific and instructive reason: it built
its probe batch from the first two manifest rows, which happened to be 120 and
118 tokens long. **A batch with no padding cannot test a collator.** The check
now spans the duration range and refuses to run on an unpadded batch.

Full record, including the discarded numbers and the other fourteen failures:
[EXPERIMENT_STATUS.md](EXPERIMENT_STATUS.md).

---

## Measurement conventions

Fixed once, applied everywhere, recorded in every `metrics.json`.

**Normalization** (`src/evaluation/normalization.py`) — each rule is a named
flag, and the active set is fingerprinted into every metrics file. Defaults:
NFKC on, punctuation stripped, lowercase on, whitespace removed, the
`language X<asr_text>` prompt prefix stripped. Traditional→Simplified and
Chinese-numeral→Arabic conversion are **off** by default: both are semantic
rewrites that can hide genuine recognition errors. A number-normalized CER is
reported separately when needed.

**CER** — corpus CER is `sum(edit_distance) / sum(len(reference))`. Macro CER
(mean of per-utterance CER, clipped at 1.0) is reported alongside, because a
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
(`metrics.paired_bootstrap`). Pairing is asserted: the two systems must cover
the same `utt_id`s in the same order with identical reference lengths,
otherwise the call raises.

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

Synthesis specification and acceptance checklist: [TTS_GENERATION_SPEC.md](TTS_GENERATION_SPEC.md).
Prompt used to commission additional spoken lines: [LLM_SCRIPT_PROMPT.md](LLM_SCRIPT_PROMPT.md).

---

## Layout

```
configs/                 one YAML per experiment
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

Two isolated venvs live under the project root, because `qwen-asr` pins
`transformers==4.57.6` and `qwen-tts` pins `4.57.3` — they cannot coexist.

| Env | Purpose | Key pins |
|---|---|---|
| `env_asr` | training + evaluation | torch 2.9.1+cu128, transformers 4.57.6, qwen-asr 0.0.6, peft 0.20.0 |
| `env_tts` | data generation only | transformers 4.57.3, qwen-tts |

---

## Reproduce

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

Every run writes `config.yaml`, `config.source.yaml`, `environment.json`,
`command.txt`, `train.log`, `trainable_parameters.txt` and `metrics.json` into
its own `experiments/<run>/` directory. Seeds are pinned in the config; the
environment record is captured per run, not per project.

---

## Team

<table>
<tr>
<td align="center" width="50%">
<a href="https://github.com/terrense">
<img src="https://github.com/terrense.png?size=180" width="120" alt=""><br>
<b>沈鑫 · terrense</b>
</a><br>
<sub><b>后训练算法 · Post-training algorithms</b></sub><br>
<sub>组件级适配设计与梯度隔离验证</sub><br>
<sub>训练与评测栈 · 统计分析</sub>
</td>
<td align="center" width="50%">
<a href="https://github.com/NaYangyeee">
<img src="https://github.com/NaYangyeee.png?size=180" width="120" alt=""><br>
<b>娜样 · NaYangyeee</b>
</a><br>
<sub><b>SFT · LoRA 调参 · 数据准备</b></sub><br>
<sub>监督微调与 LoRA 超参搜索</sub><br>
<sub>数据清洗与语料准备</sub>
</td>
</tr>
</table>

### Contributors

| | Contributor | Focus |
|:--:|---|---|
| <img src="https://github.com/terrense.png?size=64" width="32"> | [沈鑫 · @terrense](https://github.com/terrense) | Post-training algorithms, component adaptation design, training & evaluation stack |
| <img src="https://github.com/NaYangyeee.png?size=64" width="32"> | [娜样 · @NaYangyeee](https://github.com/NaYangyeee) | Supervised fine-tuning, LoRA hyperparameter search, data cleaning & preparation |

---

## Acknowledgements

- [QwenLM/Qwen3-ASR](https://github.com/QwenLM) — base models and the reference SFT script
- [huggingface/peft](https://github.com/huggingface/peft) — LoRA implementation
- [AISHELL-1](https://www.openslr.org/33/) — general-domain control set
- [RIRS_NOISES](https://www.openslr.org/28/) — real room impulse responses and point-source noise for acoustic augmentation

---

## Citation

Work in progress; please cite the repository until the paper is available.

```bibtex
@misc{shen2026whereadapt,
  title  = {Where to Adapt {Qwen3-ASR}? Component-wise Parameter-Efficient
            Adaptation for Low-Resource Chinese Hospital-Domain Speech Recognition},
  author = {Shen, Xin and Cui, Lina},
  year   = {2026},
  note   = {Work in progress},
  url    = {https://github.com/terrense/Qwen-ASR-medical-SFT}
}
```
