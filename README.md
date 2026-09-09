<div align="center">

<img src="assets/logo.png" width="140" alt="Qwen-ASR medical SFT">

# Where to Adapt Qwen3-ASR?

**Component-wise parameter-efficient adaptation for low-resource Chinese hospital-domain ASR**

[![status](https://img.shields.io/badge/status-work%20in%20progress-orange?style=flat-square)](EXPERIMENT_STATUS.md)
[![model](https://img.shields.io/badge/Qwen3--ASR-0.6B%20%7C%201.7B-5A4FCF?style=flat-square)](https://github.com/QwenLM)
[![peft](https://img.shields.io/badge/PEFT-LoRA%20r16-5A4FCF?style=flat-square)](configs/)
[![python](https://img.shields.io/badge/python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![torch](https://img.shields.io/badge/torch-2.9.1%2Bcu128-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![transformers](https://img.shields.io/badge/transformers-4.57.6-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://github.com/huggingface/transformers)

[**Experiment log**](EXPERIMENT_STATUS.md) ·
[Arms](#2-the-seven-arms) ·
[Results](#4-results) ·
[Why it broke](#5-postmortem-the-bug-that-cost-a-full-sweep) ·
[Reproduce](#7-reproduce) ·
[Team](#team)

</div>

---

## 📰 News

- **[2026-09-09]** Sweep re-run against the fixed objective. `A4` text-LoRA reaches **2.03%**, beating both zero-shot (p=0.035) and full SFT (p=0.0002) with 0.583% of the parameters. `A3` projection-only is genuinely last (p<0.0001). [Results ↓](#42-adaptation-sweep-1-h-general-domain-budget)
- **[2026-09-09]** Failure #15 found and fixed: the label mask was anchored to the wrong end of the sequence under left padding. The 2026-09-04 sweep is **retracted**. Batched loss 16.43 → 4.08. [Postmortem ↓](#5-postmortem-the-bug-that-cost-a-full-sweep)
- **[2026-09-04]** All 7 arms execute end to end on a 1 h general-domain budget. Numbers later retracted; two blocking bugs (#13, #14) found and fixed in the process.
- **[2026-09-03]** Zero-shot baselines final on AISHELL-1 test, full 7,176 utterances: **0.6B 2.10%**, **1.7B 1.54%**, paired bootstrap +0.56 pp, 95% CI [+0.48, +0.65].
- **[2026-09-03]** Gradient isolation proved for 7/7 arms: A2 touches 0 decoder tensors, A4 touches 0 encoder tensors.

---

## 1. The question

Not *whether* fine-tuning Qwen3-ASR helps. **Which component to adapt.**

```
 waveform ──▶ ┌──────────────────┐ ──▶ ┌────────────┐ ──▶ ┌──────────────────┐ ──▶ text
              │  AUDIO ENCODER   │     │ PROJECTION │     │  TEXT DECODER    │
              │  18 layers       │     │ 896→896    │     │  28 Qwen3 layers │
              │                  │     │ 896→1024   │     │  + lm_head       │
              └──────────────────┘     └────────────┘     └──────────────────┘
                   184.7 M                  1.7 M               596.0 M
                    23.06%                  0.22%                76.17%
```

Three components, three incompatible explanations for why hospital speech fails,
three different things to fix:

| If the bottleneck is… | Symptom | Then adapt | Arm |
|---|---|---|---|
| 🔊 **Acoustics** | ward noise, masks, monitor beeps, accent | audio encoder | `A2` |
| 📖 **Language** | terms heard right, written wrong — 布洛芬缓释胶囊 *(ibuprofen SR capsules)*, 增强CT *(contrast-enhanced CT)*, 窦性心律不齐 *(sinus arrhythmia)* | text decoder | `A4` |
| 🔗 **Alignment** | both halves fine, the bridge is miscalibrated for the domain | projection head | `A3` |

The third is the interesting one. The projection head is **1,722,240 parameters —
0.22% of the model** (0.155% on 1.7B). If most of the domain gain lives there,
adaptation costs ~7 MB per department instead of a 1.5 GB model, and adapters
hot-swap.

Component boundaries are derived from the loaded module graph by
[`src/models/components.py`](src/models/components.py), never hard-coded, so an
upstream rename fails an assertion instead of silently changing the experiment.

---

## 2. The seven arms

❄️ frozen · 🔥 trained

| Arm | Audio encoder | Projection | Text decoder | Trainable (0.6B) | Answers |
|---|:--:|:--:|:--:|--:|---|
| `A0_zero_shot` | ❄️ | ❄️ | ❄️ | — | baseline |
| `A1_full_sft` | 🔥 full | 🔥 | 🔥 | 782,426,112 · 100% | upper bound |
| `A2_audio_lora` | 🔥 LoRA | ❄️ | ❄️ | 2,064,384 · 0.263% | acoustics only |
| `A3_projection_only` | ❄️ | 🔥 | ❄️ | **1,722,240 · 0.220%** | **the hypothesis** |
| `A4_text_lora` | ❄️ | ❄️ | 🔥 LoRA | 4,587,520 · 0.583% | language only |
| `A5_audio_lora_proj` | 🔥 LoRA | 🔥 | ❄️ | 3,786,624 · 0.483% | acoustics + bridge |
| `A6_text_lora_proj` | ❄️ | 🔥 | 🔥 LoRA | 6,309,760 · 0.802% | language + bridge |
| `A7_dualpeft` | 🔥 LoRA | 🔥 | 🔥 LoRA | 8,374,144 · 1.061% | is PEFT enough? |

LoRA: `r=16`, `alpha=32`, `dropout=0.05`, attention projections only. Counts are
checked arithmetically — A2 = 72 modules × (16×896 + 896×16), A4 = 28 layers ×
163,840, A7 = A2 + A4 + projection.

**How to read the outcome:**

| Observation | Conclusion |
|---|---|
| `A2` ≫ `A4` | the hospital bottleneck is acoustic |
| `A4` ≫ `A2` | the bottleneck is terminology and language prior |
| `A3` ≈ `A1` | **the thesis holds** — 0.22% of parameters buys most of the gain |
| `A7` ≈ `A1` | PEFT suffices; full SFT is not required |
| `A1` alone wins | negative result: component-level adaptation is not enough |

Every arm is also re-scored on AISHELL-1 to quantify catastrophic forgetting
against the 2.10% / 1.54% zero-shot reference.

---

## 3. ⚠️ The trap this repo exists to avoid

The audio tower and the text decoder **both** define `q_proj`, `k_proj`, `v_proj`.
The obvious line:

```python
LoraConfig(target_modules=["q_proj", "v_proj"])   # WRONG for this study
```

hits **92 modules across both branches** on 0.6B (104 on 1.7B). `A2` and `A4`
silently become the same experiment and the study is void — with no error, just
plausible-looking numbers.

Targeting is therefore by full-path regex, asserted disjoint against the live
inventory:

```
^thinker\.audio_tower\.layers\.\d+\.self_attn\.(k_proj|out_proj|q_proj|v_proj)$
^thinker\.model\.layers\.\d+\.self_attn\.(k_proj|o_proj|q_proj|v_proj)$
```

Note the attention output projection is `out_proj` in the audio tower and
`o_proj` in the decoder. Both spellings come from the inventory, not from memory.
Isolation is re-proved per arm per run; a run aborts if any unintended parameter
receives gradient.

---

## 4. Results

### 4.1 Zero-shot baselines — final ✅

AISHELL-1 test, 7,176 utterances / 10.03 h / 20 speakers, full set, no subsampling.

| Model | CER | Params | Note |
|---|--:|--:|---|
| Qwen3-ASR-0.6B | 2.10% | 782 M | forgetting reference for all 0.6B arms |
| Qwen3-ASR-1.7B | **1.54%** | 2.04 B | forgetting reference for all 1.7B arms |

Paired bootstrap over utterances, 10,000 replicates: **+0.56 pp, 95% CI
[+0.48, +0.65], p < 0.0001**. Pairing is asserted, not assumed — same `utt_id`s,
same order, identical reference lengths, or the call raises.

These do not go through the training collator and are **unaffected** by failure #15.

### 4.2 Adaptation sweep, 1 h general-domain budget

810 AISHELL-1 utterances (1.00 h), 3 epochs, batch 8, seed 42, evaluated on the
full AISHELL-1 test set. Configs are byte-identical to the retracted run apart
from the output path, so the collator fix (§5) is the only variable.

| Arm | CER | Δ vs zero-shot | train_loss | Trainable | Train |
|---|--:|--:|--:|--:|--:|
| `A4` text-LoRA | **2.03%** | −0.069 | 0.250 | 0.583% | 54 s |
| `A7` DualPEFT | 2.05% | −0.047 | 0.182 | 1.061% | 115 s |
| `A6` text-LoRA + proj | 2.07% | −0.031 | 0.184 | 0.802% | 55 s |
| `A5` audio-LoRA + proj | 2.08% | −0.020 | 0.232 | 0.483% | 102 s |
| `A0` zero-shot | 2.10% | — | — | — | — |
| `A1` full SFT | 2.17% | +0.071 | 0.305 | 100% | 135 s |
| `A2` audio-LoRA | 2.20% | +0.100 | 0.573 | 0.263% | 99 s |
| `A3` projection only | 2.70% | +0.601 | 0.473 | 0.220% | 44 s |

Losses now converge to 0.18–0.57, against 4.7–12.7 before the fix. Peak VRAM
5.9–8.7 GiB; the whole sweep is 70 minutes wall-clock including evaluation.

**Significance.** Paired bootstrap over utterances, 10,000 replicates, seed 42
(`results/metrics/paired_bootstrap_general_1h_v2.json`). Only these comparisons
were tested:

| Comparison | Δ (pp) | 95% CI | p | |
|---|--:|---|--:|:--|
| `A3` − `A0` | +0.601 | [+0.513, +0.691] | <0.0001 | ✅ significant |
| `A2` − `A3` | −0.501 | [−0.593, −0.408] | <0.0001 | ✅ significant |
| `A4` − `A1` | −0.139 | [−0.211, −0.068] | 0.0002 | ✅ significant |
| `A4` − `A0` | −0.069 | [−0.132, −0.005] | 0.035 | ✅ marginal |
| `A7` − `A0` | −0.047 | [−0.104, +0.010] | 0.107 | ✗ |
| `A1` − `A0` | +0.071 | [−0.006, +0.148] | 0.071 | ✗ |
| `A4` − `A5` | −0.049 | [−0.118, +0.022] | 0.176 | ✗ |
| `A4` − `A7` | −0.022 | [−0.086, +0.043] | 0.521 | ✗ |

Three tests clear significance, one is marginal, the remaining four are null:

- **`A3` projection-only is genuinely last**, by a wide and unambiguous margin.
- **`A4` text-LoRA genuinely beats `A1` full SFT** — 0.583% of the parameters
  outperforms updating all of them, and full SFT does not separate from
  zero-shot at all.
- `A4` clears zero-shot, but the CI lower bound is −0.005 pp. Treat as marginal.
- **`A4`, `A7`, `A6` and `A5` are mutually indistinguishable.** Do not rank them.

> **This is not the adaptation experiment.** Training on AISHELL and testing on
> AISHELL is in-distribution refinement of a model that is already strong there —
> there is no domain shift for a projection head to correct, so `A3` finishing
> last is uninformative about the hospital-domain hypothesis. These runs exist to
> show the pipeline produces analyzable results. The component question is
> decided by §6's pending corpus, not here.

---

## 5. Postmortem: the bug that cost a full sweep

### 5.1 What a training sequence looks like

```
<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
  <|audio_pad|> × N            ← N = 26…54, scales with audio length
<|audio_end|><|im_end|>\n<|im_start|>assistant\n
  但考虑到目前的价格水平<|im_end|>    ← the transcript. ONLY this may enter the loss
                                     ("but considering the current price level")
```

Everything before the transcript is conditioning and must be masked to `-100`.
The collator did:

```python
labels[i, :prefix_len] = -100     # mask the first prefix_len positions
```

which assumes **the sequence starts at index 0**.

### 5.2 Padding side

```
right padding   [ real tokens ................ ][ PAD PAD PAD ]
                  ▲ index 0 is the real start                      ✅ assumption holds

left padding    [ PAD PAD PAD ][ real tokens ................ ]
                  ▲ index 0 is padding                             ❌ assumption broken
```

`Qwen3ASRProcessorKwargs._defaults["text_kwargs"]` sets `padding_side="left"`,
**overriding the tokenizer's own `"right"`**. Reading `tokenizer_config.json`
does not reveal this. Only decoding an actual batch does.

### 5.3 What that did to the shortest row in the batch

```
row s3 — batch width 120, 60 real tokens

 index    0 ······················ 59 │ 60 ································ 119
 content  PAD × 60                    │ <|im_start|>system … <|audio_pad|>×40 … 但考虑到目前的价格水平<|im_end|>
 masked   ├──── labels[:54] = -100 ───┤   ← nothing masked from here on
                                        ╰────────────── all of this entered the loss
```

The model was trained to **predict the audio placeholders** — tokens that carry
no causal information, since their content is injected by the audio tower.

### 5.4 Why the safety check passed anyway

It built its probe batch from `manifest_rows[:2]`. Those two rows happened to be
**120 and 118 tokens**. Two positions of misalignment is harmless, so the check
reported a healthy `initial_loss = 3.8626` — identical on every arm, every run —
while real training at batch 8 (lengths 60…120) ran at 16.4.

> **A batch with no padding cannot test a collator.**

Measured on the untrained 0.6B, same rows:

| Batch | Padding | Before | After |
|---|---|--:|--:|
| n=1, each of 8 rows | none | 4.330 (mean) | 4.330 |
| n=2, rows 0–1 (120 / 118) | 2 tokens | 3.863 | 3.863 |
| **n=8, rows 0–7 (120 … 60)** | up to 60 tokens | **16.434** | **4.077** |

After the fix the supervised span decodes to exactly the transcript plus
`<|im_end|>` on all 8 rows.

**Fix.** `build_labels()` counts back from the last real token
(`end - n_target : end`), correct under either padding side, and is now the
single implementation shared by the trainer and the safety check. The check
selects rows spanning the duration range and refuses to run on an unpadded batch.

### 5.5 Why the numbers had to be thrown out, not merely flagged

With a corrupted target, the ranking partly measures *capacity to memorise the
corruption* rather than fitness for adaptation:

- `A1`, 782 M trainable → drove loss to 1.34 by memorising it
- `A3`, 1.7 M trainable → could not, and stalled at 10.5

The rerun quantifies the distortion. `A5` fell from first place to fourth, `A4`
rose from third to first, and `A3`'s deficit against zero-shot shrank from
**+4.01 pp to +0.60 pp** — the bug overstated it by 6.7×.

`A3` did stay last, so the bug did not manufacture that result. It inflated it
beyond recognition and scrambled everything above it, which is why the table had
to be discarded rather than annotated.

---

## 6. Why training started before the hospital data exists

That sweep was **not** trained on medical data. It ran on 810 AISHELL-1
utterances (1 h) to retire engineering risk. Three bugs surfaced that are
invisible until training actually runs:

| # | Failure | Blast radius |
|---|---|---|
| 13 | `ValueError: GenerationConfig is invalid` on save. Qwen3-ASR ships `temperature=1e-06` with `do_sample=False`; transformers only warns at load but validates on `save_pretrained`. PEFT arms escaped it by serialising an adapter | **2/7 arms, including A3** |
| 14 | `Can't find 'adapter_config.json'`. The runner passed every non-A0 checkpoint as `--adapter_path`, but A1 and A3 write a standalone model directory | **2/7 arms, including A3** |
| 15 | Label mask anchored to the wrong end under left padding (§5) | **7/7 arms** |

Had the first training run waited for the hospital corpus, all three would have
been found on the most expensive data under schedule pressure — and #13 and #14
would each have silently removed `A3` from the study.

AISHELL was not an arbitrary choice: it is the general-domain control set the
paper needs anyway for catastrophic forgetting, so validating on it costs nothing
extra.

**The scientific value of these runs is zero by construction** — training and
testing on the same distribution can only show that training does something. The
hospital-domain baseline, the real arm ranking, terminology error rate and the
forgetting measurement all still require the corpus.

---

## 7. Reproduce

Two isolated venvs under the project root: `qwen-asr` pins
`transformers==4.57.6` and `qwen-tts` pins `4.57.3`, so they cannot coexist.

| Env | Purpose | Key pins |
|---|---|---|
| `env_asr` | training + evaluation | torch 2.9.1+cu128, transformers 4.57.6, qwen-asr 0.0.6, peft 0.20.0 |
| `env_tts` | data generation only | transformers 4.57.3, qwen-tts |

```bash
# 0 — record the environment
python scripts/audit_environment.py

# 1 — module inventory and verified LoRA targets
python scripts/inspect_model.py --model_path models/Qwen3-ASR-0.6B --device cpu

# 3 — text corpus
python scripts/generate_scripts.py --total 18000 --seed 42

# 4 — family-disjoint split
python scripts/split_data.py --seed 42

# 9 — medical entity lexicon
python scripts/build_medical_lexicon.py

# 5 — synthesize audio (env_tts). Anchors first, then the corpus.
sh scripts/setup_env_tts.sh
env_tts/bin/python scripts/generate_tts.py --stage anchors
env_tts/bin/python scripts/generate_tts.py --stage corpus --split train

# 4b — nested duration budgets, D1 ⊂ D5 ⊂ D10 ⊂ D20
python scripts/build_duration_subsets.py --manifest data/manifests/train_synthetic.jsonl

# 7 — acoustic augmentation (deterministic on seed)
python scripts/augment_corpus.py \
    --manifest data/manifests/train_20h.jsonl \
    --outdir data/synthetic/audio/train_aug \
    --out_manifest data/manifests/train_20h_aug.jsonl

# 14 — general-domain control set
python scripts/prepare_aishell.py --root /path/to/aishell1 --split test

# 11 — MANDATORY before any training job
python tests/test_trainable_parameters.py \
    --model_path models/Qwen3-ASR-0.6B --arm all --outdir experiments/safety

# 16 — run one experiment end to end
python scripts/make_configs.py
python scripts/run_experiment.py configs/qwen06_dualpeft_20h.yaml

# 8 — evaluate a single checkpoint directly
python src/evaluation/run_eval.py \
    --model_path models/Qwen3-ASR-0.6B \
    --manifest data/manifests/test_synthetic.jsonl \
    --lexicon data/medical_lexicon.json \
    --outdir experiments/qwen06_zero/test_synthetic

# 19, 20 — figures and tables (XX wherever a run is missing)
python scripts/make_figures.py
python scripts/make_tables.py

# scoring-stack regression tests
python tests/test_metrics.py
```

Every run writes `config.yaml`, `config.source.yaml`, `environment.json`,
`command.txt`, `train.log`, `trainable_parameters.txt` and `metrics.json` into
its own `experiments/<run>/`. Seeds are pinned in the config; the environment is
captured per run, not per project.

---

## 8. Measurement conventions

Fixed once, applied everywhere, recorded in every `metrics.json`.

**Normalization** — each rule is a named flag and the active set is fingerprinted
into every metrics file. Defaults: NFKC on, punctuation stripped, lowercase on,
whitespace removed, the `language X<asr_text>` prompt prefix stripped.
Traditional→Simplified and Chinese-numeral→Arabic are **off**: both are semantic
rewrites that can hide genuine recognition errors. A number-normalized CER is
reported separately when needed.

**CER** — corpus CER is `sum(edit_distance) / sum(len(reference))`. Macro CER
(mean per-utterance, clipped at 1.0) is reported alongside, because a corpus
figure can be dominated by long utterances. Empty references are excluded and
counted, never scored as 0.0 or 1.0.

**Terminology** — medical-term error rate uses multiset matching over the
417-term lexicon with longest-match-first masking, so a hit on `增强ct`
*(contrast-enhanced CT)* does not also count as a hit on the substring `ct`.
Reported separately from CER on purpose: an
utterance can score 16.7% CER while missing 50% of its medical terms, and that
gap is a finding.

**Decoding controls** — Qwen3-ASR accepts a biasing `context` string. It is empty
for every condition in every comparison; hospital vocabulary supplied as context
would improve CER with no adaptation at all. The value is written into every
`metrics.json` so this can be checked rather than trusted.

**Statistics** — paired bootstrap over utterances, ≥10,000 replicates, 95% CI.

---

## 9. Data provenance

All 17,986 utterances are generated from hand-written spoken-Mandarin patterns
filled from a curated hospital vocabulary
([`src/data/hospital_vocab.py`](src/data/hospital_vocab.py)). Nothing is copied
from web consultations; no entry refers to a real person, hospital or case.
Synthetic speaker identities are generated voices, not clones of identifiable
people.

Splits are disjoint by **template family** — a semantic sentence pattern — not
merely by waveform, so a test utterance is never a re-voicing of a pattern seen
in training. Both properties are asserted at split time.

Synthesis spec and acceptance checklist: [TTS_GENERATION_SPEC.md](TTS_GENERATION_SPEC.md).
Prompt for commissioning additional lines: [LLM_SCRIPT_PROMPT.md](LLM_SCRIPT_PROMPT.md).

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

---

## Team

<table>
<tr>
<td align="center" width="33%">
<a href="https://github.com/terrense">
<img src="https://github.com/terrense.png?size=180" width="120" alt=""><br>
<b>Shen Xin 沈鑫</b>
</a><br>
<sub><b>Post-training algorithms</b></sub><br>
<sub>Component adaptation design · gradient isolation</sub><br>
<sub>Training &amp; evaluation stack · statistics</sub>
</td>
<td align="center" width="33%">
<a href="https://github.com/NaYangyeee">
<img src="https://github.com/NaYangyeee.png?size=180" width="120" alt=""><br>
<b>Na Yang 娜样</b>
</a><br>
<sub><b>SFT · LoRA tuning</b></sub><br>
<sub>Supervised fine-tuning · LoRA hyperparameter search</sub><br>
<sub>Data cleaning &amp; corpus preparation</sub>
</td>
<td align="center" width="33%">
<a href="https://github.com/FHY163valey">
<img src="https://github.com/FHY163valey.png?size=180" width="120" alt=""><br>
<b>Feng Hongyang 冯鸿阳</b>
</a><br>
<sub><b>Data cleaning · evaluation</b></sub><br>
<sub>Corpus cleaning &amp; quality control</sub><br>
<sub>Training-result evaluation</sub>
</td>
</tr>
</table>

| | Contributor | Focus |
|:--:|---|---|
| <img src="https://github.com/terrense.png?size=64" width="28"> | [@terrense](https://github.com/terrense) | Post-training algorithms, component adaptation design, training & evaluation stack |
| <img src="https://github.com/NaYangyeee.png?size=64" width="28"> | [@NaYangyeee](https://github.com/NaYangyeee) | Supervised fine-tuning, LoRA hyperparameter search, data cleaning & preparation |
| <img src="https://github.com/FHY163valey.png?size=64" width="28"> | [@FHY163valey](https://github.com/FHY163valey) | Corpus cleaning and quality control, evaluation of training results |

---

## Acknowledgements

- [QwenLM/Qwen3-ASR](https://github.com/QwenLM) — base models and the reference SFT script
- [huggingface/peft](https://github.com/huggingface/peft) — LoRA implementation
- [AISHELL-1](https://www.openslr.org/33/) — general-domain control set
- [RIRS_NOISES](https://www.openslr.org/28/) — real room impulse responses and point-source noise

---

## Citation

Work in progress. Please cite the repository until the paper is available.

```bibtex
@misc{shen2026whereadapt,
  title  = {Where to Adapt {Qwen3-ASR}? Component-wise Parameter-Efficient
            Adaptation for Low-Resource Chinese Hospital-Domain Speech Recognition},
  author = {Shen, Xin and Cui, Lina and Feng, Hongyang},
  year   = {2026},
  note   = {Work in progress},
  url    = {https://github.com/terrense/Qwen-ASR-medical-SFT}
}
```
