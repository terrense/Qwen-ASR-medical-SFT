# Experiment Status

Single source of truth for what has run, what has not, and what failed.
No result is recorded here unless it exists as a file on disk.

Last updated: 2026-09-03

---

## Compute and storage

| Item | Value |
|---|---|
| Training/eval host | internal H20 container (address and credentials kept out of this repository) |
| GPU | 1× NVIDIA H20-3e, 143,771 MiB, HAMi vGPU (shared physical card) |
| CPU / RAM | 16 cores / ~2 TB |
| Project root (remote) | `/data/shenxin/qwen3_asr_hospital/` |
| Project root (authoring) | `F:\qwen3_asr_hospital\` |
| Free space on `/data` | 14 TB of 33 TB; container overlay 286 GB free |

Authoring happens on `F:`; the remote copy is the execution environment. The
remote path is inside the user's personal space, and no other `/data/*`
directory is written to.

---

## Phase status

| Phase | Description | Status | Evidence |
|---|---|---|---|
| 0 | Environment audit + repo scaffold | **partial** | `logs/audit_authoring_machine/environment.{txt,yaml,json}` for the authoring machine. Remote audit blocked on `env_asr` install. |
| 1 | Model module inventory | **code ready, not run** | `scripts/inspect_model.py`, `src/models/components.py` |
| 2 | Manifest standard | **done** | `src/data/manifest.py` (schema, validator, official-SFT projection) |
| 3 | Text corpus generation | **done** | `data/scripts/all_scripts.jsonl` — 17,986 utterances, 71 families |
| 4 | Splitting (stage A: scripts) | **done** | `data/manifests/splits/` — family-disjoint |
| 4 | Splitting (stage B: duration subsets) | **code ready + logic verified** | `scripts/build_duration_subsets.py`; needs real durations |
| 5 | TTS generation | **code ready, not run** | `scripts/generate_tts.py`, `src/data/speakers.py` (32 identities) |
| 6 | Audio QC | **done (code + verified)** | `src/data/audio_qc.py` |
| 7 | Acoustic augmentation | **done (code + verified end to end)** | `src/augmentation/acoustic.py`, `scripts/augment_corpus.py` |
| 8 | Zero-shot baseline | **pipeline verified; real baseline blocked on audio** | `experiments/pipeline_smoke/` |
| 9 | Metrics + medical lexicon | **done** | `src/evaluation/metrics.py`, `data/medical_lexicon.json` (417 terms) |
| 10 | Training configurations | **code ready** | `src/models/components.py` — arms A0–A7 |
| 11 | Training safety check | **DONE — 7/7 arms pass** | `experiments/safety/<arm>/trainable_parameters.{txt,json}` |
| 12–15 | Experiment matrix | **configs written, not run** | `configs/MATRIX.md` |
| 16 | Reproducibility scaffolding | **done** | 26 configs from `scripts/make_configs.py` |
| 17 | Random seeds | **planned** | seeds 123/2026 configs exist for the 1 h Text-LoRA and DualPEFT comparisons |
| 18 | Statistical analysis | **done (code)** | `metrics.paired_bootstrap`, 10k replicates, pairing asserted |
| 19 | Figures | **code ready; figure 2 generated** | figures 1, 3–6 correctly skipped pending inputs |
| 20 | Tables | **done (code)** | `scripts/make_tables.py` — 7 tables, 197/247 cells `XX` |

---

## Completed work in detail

### Phase 1 — module inventory, VERIFIED against the installed package

`env_asr/bin/python scripts/inspect_model.py --model_path models/Qwen3-ASR-0.6B --device cpu --dtype float32`

Component roots were discovered from the loaded graph, not assumed:
`thinker.audio_tower` (class `*AudioEncoder`) and `thinker.model`
(class `*TextModel`). The audio projection resolved to
`thinker.audio_tower.proj1` (896→896) and `.proj2` (896→1024). The convolutional
front-end's `conv_out` was excluded automatically, as designed, because its
`in_features` is 7680 (the flattened mel/channel width), not `d_model`.

**Qwen3-ASR-0.6B: 782,426,112 parameters, 308 `nn.Linear` modules.**

| Component | Linear modules | Linear params | % of model |
|---|---|---|---|
| AUDIO_ENCODER | 109 | 180,434,688 | 23.06% |
| AUDIO_PROJECTION | 2 | 1,722,240 | **0.22%** |
| TEXT_DECODER | 197 | 595,984,384 | 76.17% |
| OTHER | 0 | 0 | 0.00% |

`OTHER` being empty matters: every Linear in the model is accounted for by one
of the three components, so no parameter can silently fall outside the ablation.

**The targeting hazard, measured.** `q_proj`, `k_proj` and `v_proj` each occur in
*both* AUDIO_ENCODER and TEXT_DECODER. A config written as
`target_modules=["q_proj","v_proj"]` matches **92 modules spanning both
components** — it would make "audio-only LoRA" and "text-only LoRA" the same
experiment. The verified full-path regexes instead give:

```
audio: ^thinker\.audio_tower\.layers\.\d+\.self_attn\.(k_proj|out_proj|q_proj|v_proj)$   → 72 modules
text : ^thinker\.model\.layers\.\d+\.self_attn\.(k_proj|o_proj|q_proj|v_proj)$           → 112 modules
overlap: none (asserted)
```

72 = 18 audio layers × 4 projections; 112 = 28 decoder layers × 4. The attention
output projection is `out_proj` in the audio tower and `o_proj` in the decoder;
both spellings were discovered from the inventory rather than hard-coded.

**Design consequence.** The audio projection head is only **0.22%** of the model
(1.72M parameters). That is the entire trainable budget of arm A3, and it makes
the projection-only cell the cheapest possible adaptation by two orders of
magnitude against full SFT — which is precisely the comparison the paper is
built to make.

The earlier static reading of
`qwen_asr/core/transformers_backend/modeling_qwen3_asr.py` at commit
`878647b0f94c5f17cfc3346d5faed831bcb675fd` agrees with the installed package in
every respect.

Qwen3-ASR-0.6B, from `config.json`:

- audio encoder: 18 layers, `d_model=896`, 14 heads, FFN 3584, `output_dim=1024`
- text decoder: Qwen3, 28 layers, hidden 1024, 16 heads / 8 KV heads, vocab 151,936

| Component | Module paths |
|---|---|
| AUDIO_ENCODER | `thinker.audio_tower.conv2d{1,2,3}`, `.conv_out`, `.layers.{0..17}.self_attn.{q,k,v,out}_proj`, `.layers.{0..17}.{fc1,fc2}` |
| AUDIO_PROJECTION | `thinker.audio_tower.proj1` (896→896), `thinker.audio_tower.proj2` (896→1024) |
| TEXT_DECODER | `thinker.model.layers.{0..27}.self_attn.{q,k,v,o}_proj`, `.mlp.{gate,up,down}_proj`, `thinker.lm_head` |

**Confirmed hazard.** `q_proj`, `k_proj` and `v_proj` exist in *both* the audio
tower and the text decoder. A PEFT config written as
`target_modules=["q_proj","v_proj"]` would adapt both branches at once and
silently invalidate every component comparison. All targeting therefore uses
full-path regexes, asserted disjoint against the live module inventory in
`components.attention_regexes()`. The attention output projection is spelled
`out_proj` in the audio tower but `o_proj` in the decoder; both are discovered
from the inventory rather than assumed.

### Training stack — all 7 arms execute end to end (pipeline validation)

Run before the hospital audio exists, on 300 AISHELL-1 utterances (0.43 h), to
retire engineering risk early. **The CER column is not a result** — 300
general-domain utterances for one epoch can only damage a general-domain test
set; it is here to prove the checkpoint reloads and produces sane output.

| Arm | Trainable | % | Train time | Peak VRAM | Checkpoint | Validation CER |
|---|---|---|---|---|---|---|
| A1 full SFT | 782,426,112 | 100% | 31.1 s | 7.26 GiB | 1.5 G | 14.44% |
| A2 audio-LoRA | 2,064,384 | 0.263% | 19.6 s | 4.22 GiB | 18 M | 2.53% |
| A3 projection only | 1,722,240 | 0.220% | 17.3 s | 3.99 GiB | 1.5 G | 8.88% |
| A4 text-LoRA | 4,587,520 | 0.583% | 18.0 s | 4.36 GiB | 18 M | 3.18% |
| A5 audio-LoRA + proj | 3,786,624 | 0.483% | 23.8 s | 4.23 GiB | 18 M | 2.60% |
| A6 text-LoRA + proj | 6,309,760 | 0.802% | 13.7 s | 4.37 GiB | 18 M | 3.54% |
| A7 DualPEFT | 8,374,144 | 1.061% | 25.9 s | 4.61 GiB | 18 M | 3.25% |

Zero-shot on the same 100 held-out utterances is 2.10%. Every arm is worse,
which is the expected direction. Loss moved 12.685 → 4.396 on A4, so training is
doing work rather than idling.

**Compute is not the bottleneck.** Extrapolating from 0.43 h of audio: a 20-hour
budget is roughly 17 minutes per epoch for a LoRA arm and 29 minutes for full
SFT. The whole 26-configuration matrix is hours of GPU time, not days. The
bottleneck is TTS synthesis (measured RTF ≈ 2 per utterance), which is being
handled separately.

Storage note: A1 and A3 write a full 1.5 G model per checkpoint, the PEFT arms
write an 18 M adapter.

### Phase 8/14 — general-domain zero-shot baseline (FIRST REAL RESULT)

AISHELL-1 test, 7,176 utterances / 10.03 h / 20 speakers, full set, no subsampling.

| Model | CER | macro CER | S / D / I | RTF (batch 16) |
|---|---|---|---|---|
| Qwen3-ASR-0.6B | **2.10%** | 2.12% | 2037 / 79 / 84 | 0.0072 |
| Qwen3-ASR-1.7B | **1.54%** | 1.57% | 1470 / 61 / 78 | 0.0089 |

**Paired bootstrap (Phase 18, first real use):** difference 0.56 pp in favour of
1.7B, 95% CI [+0.48, +0.65] pp, p < 0.0001 over 10,000 replicates on 7,176
paired utterances. The interval excludes zero, so the gap is real rather than
sampling noise. Pairing was asserted, not assumed.

**A claim I had to retract.** Error analysis surfaced three striking cases where
1.7B wrote Arabic digits ("500米") against AISHELL's Chinese numerals ("五百米")
and was charged 8 substitutions for a correct recognition. I inferred the metric
was systematically penalising 1.7B. Re-scoring the whole set with
`normalize_numbers` shows the effect is **0.04 pp on 0.6B and 0.005 pp on 1.7B** —
one to two orders of magnitude below the 0.56 pp model gap. The inference from
three vivid examples did not survive contact with the statistics; the 1.7B
advantage is genuine recognition quality, not orthography. The number-normalized
figures are kept in `results/metrics/number_normalization_effect.json`.

Worth carrying forward: the hospital corpus has a `numeric` category at 10% that
is entirely digits and measurements, so the same orthography effect is expected
to be materially larger there and will be reported per category.

2.10% is a plausible AISHELL-1 figure for a 0.6B model, which is the first
end-to-end confirmation that the evaluation stack — decoding, normalization,
alignment, aggregation — is correct on real data at scale. Errors are
substitution-dominated (2037 S vs 79 D / 84 I), the normal profile for a healthy
ASR system rather than one that is truncating or hallucinating.

**This is the reference point for Phase 14.** Every fine-tuned checkpoint is
scored on exactly this set, and the gap against 2.10% is the general-domain
capability that hospital adaptation cost.

Only 11 medical-lexicon terms occur in the whole general-domain set (recall
0.909), confirming the lexicon is domain-specific rather than matching generic
Mandarin.

### Phase 8 — evaluation pipeline connectivity check (NOT a result)

Five real hospital recordings were pushed through the full evaluation path to
verify the code end to end before any synthetic audio exists. **The numbers
below are not a scientific baseline**: the "references" are ASR output from an
earlier streaming run, not human transcripts, so they cannot measure accuracy.
What they verify is that the path works.

Model loaded, 5/5 transcribed in 2.9 s, RTF 0.289, normalization fingerprint
`c2a1a0f12d`, medical term `神经内科` correctly detected, punctuation correctly
stripped, `predictions.jsonl` and `metrics.json` written with every required
field.

The check also produced a useful negative result: one of the five utterances
disagreed with its stored text (`高威尔随访` vs `高威二水房`, 3 substitutions).
The stored text therefore does not even reproduce under the current offline
decoder, which independently confirms it is unusable as ground truth — the
Phase 13 human rows stay `XX`.

### Phase 11 — gradient targeting, 7/7 arms pass

`env_asr/bin/python tests/test_trainable_parameters.py --model_path models/Qwen3-ASR-0.6B --arm all --device cuda:0`

Component isolation is **demonstrated, not argued**. The last column is the set
of components whose tensors actually received a non-zero gradient during a real
forward/backward pass on Qwen3-ASR-0.6B:

| Arm | Trainable params | % of model | Components receiving gradient |
|---|---|---|---|
| A1 full SFT | 782,426,112 | 100.000% | AUDIO_ENCODER 297, AUDIO_PROJECTION 4, TEXT_DECODER 310 |
| A2 audio-LoRA | 2,064,384 | 0.263% | **AUDIO_ENCODER 144 only** |
| A3 projection only | 1,722,240 | 0.220% | **AUDIO_PROJECTION 4 only** |
| A4 text-LoRA | 4,587,520 | 0.583% | **TEXT_DECODER 224 only** |
| A5 audio-LoRA + proj | 3,786,624 | 0.483% | AUDIO_ENCODER 144, AUDIO_PROJECTION 4 |
| A6 text-LoRA + proj | 6,309,760 | 0.802% | TEXT_DECODER 224, AUDIO_PROJECTION 4 |
| A7 DualPEFT | 8,374,144 | 1.061% | AUDIO_ENCODER 144, AUDIO_PROJECTION 4, TEXT_DECODER 224 |

A2 touches **zero** decoder tensors and A4 touches **zero** encoder tensors —
the two arms whose separation the entire study depends on. Every arm produced a
finite loss (7.81–8.63) on a real forward pass, and no arm leaked gradient onto
a parameter outside its declared set.

Each LoRA arm additionally passed the perturbation proof: `lora_A` is zero at
step 0 by construction, so `lora_B` is perturbed off zero and the backward pass
repeated, after which all 72 / 112 / 184 `lora_A` tensors receive gradient. A
genuinely detached adapter would fail this, so it has real discriminating power.

### Phase 3 — text corpus

`python scripts/generate_scripts.py --total 18000 --seed 42`

17,986 unique utterances, 71 template families. Rejections: 13 exact
duplicates, 3 near-duplicates, 16 unfilled quota slots. Length: min 4,
median 11, mean 10.8, p90 14, max 22 characters.

| Category | N | Actual | Target |
|---|---|---|---|
| chief_complaint | 4,491 | 24.97% | 25% |
| examination | 2,700 | 15.01% | 15% |
| registration | 1,799 | 10.00% | 10% |
| navigation | 1,800 | 10.01% | 10% |
| disease | 1,800 | 10.01% | 10% |
| medication | 1,800 | 10.01% | 10% |
| numeric | 1,796 | 9.99% | 10% |
| code_switch | 898 | 4.99% | 5% |
| disfluency | 900 | 5.00% | 5% |

Near-duplicate control uses MinHash/LSH over character 3-grams at threshold
0.90. The threshold is deliberately high: utterances differing only in the
medical term ("我头疼三天了" / "我胃疼三天了") are wanted variation, not
duplicates. Rejections are logged to `data/scripts/rejected_samples.jsonl`.

### Phase 4 stage A — script-disjoint split

| Split | Scripts | Share | Families |
|---|---|---|---|
| train | 13,520 | 75.17% | 39 |
| dev | 1,739 | 9.67% | 15 |
| test | 2,727 | 15.16% | 17 |
| cross_tts | 360 | (drawn from test families) | — |

Asserted at runtime: no template family and no identical text appears in two
splits. All nine categories are present in all three splits.

### Phase 4 stage B — nested duration subsets (logic verified on synthetic input)

D1 ⊂ D5 ⊂ D10 ⊂ D20 verified. Category proportions hold exactly at every
budget (chief_complaint 25.0%, code_switch 5.0% at all four sizes) because
selection deals round-robin over (category, speaker) groups, so *any* prefix of
the ordering is balanced.

### Phase 5 — speaker inventory (designed, not yet rendered)

32 identities, pools 20 train / 6 dev / 6 test. Perfectly balanced by gender
(16/16 overall, 3/3 in dev and test) and accent (8/8/8/8). Pool assignment is
stratified rather than random: with only six held-out voices a random draw
regularly left an accent absent from dev or test, which would make an
accent-robustness claim untestable. Round-robin script assignment gives exactly
676 utterances per training voice (5% share, under the 8% cap).

Persistence procedure: VoiceDesign renders one domain-neutral anchor per
identity → `create_voice_clone_prompt(x_vector_only_mode=True)` extracts a
speaker embedding → all corpus utterances use `generate_voice_clone` with that
embedding. Without this, VoiceDesign re-interprets its instruction per call and
the voice drifts.

### Phase 9 — medical lexicon

417 terms across 7 categories (imaging_exam 47, lab_test 57, disease 101,
medication 76, department 44, symptom 77, abbreviation 15), no cross-category
collisions. 86.93% of scripts contain at least one term; all 417 terms occur in
the corpus. Derived only from `src/data/hospital_vocab.py` — never from
predictions or test references.

### Phase 6 — audio QC, verified

Correctly accepts clean audio and rejects: all-zero, clipped (66.8% of samples
at full scale), too-short (0.06 s), and excessive lead-in (2.99 s). Trimming
reduces a 7.5 s padded clip to 1.82 s while keeping a 0.15 s margin.

Deliberately **not** implemented: filtering synthesized audio by Qwen3-ASR's own
CER, which would select the corpus in the target model's favour. The optional
consistency check uses an independent ASR system at a loose threshold (CER >
0.60) to catch catastrophic TTS failures only.

### Phase 7 — augmentation, verified

Realized condition mix over 6,000 planned utterances: clean 40.15%, noise
family 29.83% (noise 18.33 / reverb 6.03 / noise+reverb 5.47), competing speech
15.02%, codec 15.00% — against a 40/30/15/15 target. Regeneration from
`(seed, utt_id)` is bit-identical. SNR scaling is exact: requested 5/10/15/20 dB
measured back as 5.000/10.000/15.000/20.000 dB.

`scripts/augment_corpus.py` was run end to end on a 40-utterance synthetic
corpus (5 speakers): 40/40 processed, 0 manifest errors, realized mix identical
to the planned mix. The foreground-dominance invariants were checked on the
output rather than assumed — of 7 competing-speech utterances, **0** had an
interferer from the same speaker or the same utterance; SIR fell in
[6.69, 13.76] dB ⊂ [6, 15]; SNR in [11.38, 16.33] dB ⊂ [5, 20]; every `clean`
row carried `snr = sir = null`.

### Scoring-stack regression tests

`tests/test_metrics.py` — 20/20 passing on both the authoring machine and the
H20 (it needs only numpy). Covers edit-distance S/D/I decomposition, corpus vs
macro CER weighting, empty-reference exclusion, longest-match lexicon masking,
multiset term matching, bootstrap pairing assertions, and augmentation
determinism.

**A real defect this suite caught.** With default normalization,
`体温37.5度。` scored as `体温375度` and `血压120/80，` as `血压12080`: the
punctuation rule was deleting `.` and `/` inside measurements. That changes the
value and can make a genuine recognition error score as *correct*, which would
have corrupted the numeric category (10% of the corpus) and its Phase 9
breakdown. Fixed with a `keep_digit_separators` rule that protects `.` `:` `/`
**only between digits**; sentence punctuation is still stripped. A regression
test pins both the fixed behaviour and the old corruption when the flag is off.

---

## Generated artifacts currently on disk

| Artifact | Detail |
|---|---|
| `data/scripts/all_scripts.jsonl` | 17,986 utterances, 71 families |
| `data/manifests/splits/*.jsonl` | train 13,520 / dev 1,739 / test 2,727 / cross-TTS 360 |
| `data/medical_lexicon.json` | 417 terms, 7 categories |
| `configs/*.yaml` | 26 experiment configs + `MATRIX.md` |
| `results/tables/*.{csv,json,tex}` | 7 tables, all cells `XX` (nothing run yet) |
| `results/figures/figure2_data_pipeline.{png,pdf}` | real corpus counts |
| H20 `models/Qwen3-ASR-0.6B`, `models/Qwen3-ASR-1.7B` | downloaded |

---

## Failures and deviations

| # | Item | What happened | Cause | Fix | Rerun needed |
|---|---|---|---|---|---|
| 1 | SSH to the training host | Client refused to connect: host key not in the configured list | The pod had been recreated, so its ED25519 host key changed | Re-scanned the host key with `ssh-keyscan` and updated the stored fingerprint | no |
| 2 | Model download | `RuntimeError: CAS Client Error … 401 Unauthorized` from `cas-server.xethub.hf.co` | `huggingface_hub` used the Xet backend, which hf-mirror.com does not serve | `HF_HUB_DISABLE_XET=1` | no |
| 3 | pip cache | "directory is not owned or is not writable" | Samba mount forces permissions on `/data` | Moved `PIP_CACHE_DIR` to container-local `/tmp/pipcache` | no |
| 4 | Corpus quota | First run gave 17,178/18,000, categories up to 2.6 pp off target | Family capacity computed per family instead of per pattern, so quotas exceeded what any pattern could produce | Per-pattern capacity + distinct-index sampling | corpus regenerated |
| 5 | Split proportions | First run gave train 51% / dev 23% / test 26% | Greedy sequential fill overshoots when family sizes are very unequal | Largest-deficit partitioning | split regenerated |
| 6 | `env_asr` install | pip downloaded 13 different gradio wheels (31 MB each) and kept going | `qwen-asr` leaves `gradio` unpinned; gradio 6.x needs `huggingface_hub>=1.0`, which `transformers==4.57.6` forbids, so the resolver backtracked | Stopped the pip (user approved) and pinned `gradio==5.50.0`. Full dependency tree retained. | no |
| 7 | pip cache hit rate | 4.4 GB cached but 0 cache hits on restart | The aliyun mirror returns responses pip will not cache, so large wheels are re-fetched | none; accepted the re-download | no |
| 8 | Local file writes | `results/tables/*.csv` and one `.md` read back as `%TSD-Header-###%` binary | The Windows authoring machine runs a DLP agent that encrypts `.csv`/`.md` **when written by python**; the editor tool path is unaffected | Tables also emit a `.json` twin; `.md` files are written through the editor tool. Authoritative CSVs are generated on the Linux H20, where DLP does not apply | no |
| 9 | Misdiagnosis (mine) | Read the install as stalled because the log stopped advancing | pip block-buffers stdout when redirected to a file; the log lags reality | Confirmed liveness from `/proc/<pid>/io` instead (1.8 MB/s) | no |

| 10 | Phase 11 gradient check (mine) | 5/7 arms reported FAILED: every LoRA arm claimed its `lora_A` tensors "received no gradient". A1 (full SFT) and A3 (projection-only) passed — the two arms that use no LoRA | PEFT initialises `lora_B` to zeros so the adapter is a no-op at step 0. That makes `dL/d(lora_A) = lora_B^T(...) = 0` on the first backward pass. A zero gradient on `lora_A` is the expected mathematics of LoRA; my check treated "zero" as "absent" | Graph membership is now checked by the *presence* of a gradient tensor. Adapter presence on the forward path is separately **proved**: `lora_B` is perturbed away from zero, backward is re-run, and `lora_A` must then receive a non-zero gradient — a genuinely detached adapter would still fail. Non-LoRA trainable parameters must be non-zero at step 0 | check rerun |

| 11 | Chinese numeral conversion (mine) | Enabling `normalize_numbers` *raised* CER instead of lowering it | `二零一五` (read digit by digit, = 2015) was parsed with the positional algorithm meant for `三十七` (= 37), yielding **5**. Years, room numbers and identifiers were all silently corrupted | The two reading systems are now separated by the presence of a unit character (十/百/千/万/亿). Locked in by a regression test covering both systems, decimals, and trailing units | affects only the optional number-normalized diagnostic; the primary metric never used it |
| 12 | Over-reading three examples (mine) | Claimed the metric "systematically penalises 1.7B" on digit orthography, from 3 error-analysis cases | Three vivid examples are not a statistic | Re-scored all 7,176 utterances: effect is 0.04 pp / 0.005 pp, vs a 0.56 pp model gap. Claim retracted in the Phase 8/14 section | no |

| 13 | A1 and A3 could not save a checkpoint | `ValueError: GenerationConfig is invalid` at save time. The 5 PEFT arms were fine | Qwen3-ASR ships `generation_config` with `temperature=1e-06` while `do_sample=False`. transformers only *warns* at load but validates strictly on `save_pretrained`. The PEFT arms escaped it because they serialize an adapter instead of a model | `sanitize_generation_config()` clears the sampling-only flags before training (so intermediate `save_steps` checkpoints are safe too) and restores them afterwards. The flags are already ignored during greedy decoding, so nothing changes behaviourally | found and fixed before any real run |
| 14 | A1 and A3 could not be evaluated | `Can't find 'adapter_config.json'` | `run_experiment.py` passed the checkpoint as `--adapter_path` for every non-A0 arm, but the two non-PEFT arms write a standalone model directory | `classify_checkpoint()` now detects adapter vs full model and routes accordingly; `make_checkpoint_inferable()` copies the tokenizer/processor sidecar files a multimodal checkpoint needs (mirrors the official script's `copy_required_hf_files_for_qwen_asr`) | found and fixed before any real run |

Bugs 13 and 14 would each have blocked **2 of the 7 arms entirely**, including
A3 projection-only — the cheapest and most interesting arm in the study. Both
were only visible by actually running training, which is why the pipeline was
validated on public data rather than waiting for the hospital corpus.

Nothing has been trained yet on hospital data, so no experimental result is
affected by any of the above.

The Phase 11 parameter accounting was independently correct throughout, and the
arithmetic checks out by hand: A2 = 72 modules × (16×896 + 896×16) = 2,064,384;
A4 = 28 layers × 163,840 = 4,587,520; A7 = 2,064,384 + 4,587,520 + 1,722,240 =
8,374,144. Only the gradient *predicate* was wrong.

---

## Deviations from the specification, and why

1. **Cross-TTS test scripts.** The specification asks for "entirely held-out
   scripts and voices". Reserving a fourth disjoint family group per category
   would have left categories such as `code_switch` with too few families to
   train on. Instead the cross-TTS scripts are drawn from the **test** families,
   which are already held out from training, and the same scripts appear in the
   in-domain synthetic test. This satisfies "never used for training or tuning"
   and additionally isolates the TTS engine as the only changing variable. Both
   sets are written separately so either analysis remains possible.

2. **Human test sets (Phase 13).** `F:\asr_final_20260807plus` holds 1,987 real
   hospital-domain recordings, but the accompanying `rows.json` contains ASR
   output, not human transcripts (confirmed with the user). These are therefore
   **not** usable as references, and the human quiet / far-field / noisy rows in
   Phase 13 tables will read `XX` until a human-verified subset exists.

3. **`gradio` pinned to 5.50.0** rather than left unpinned as the upstream
   `qwen-asr` metadata has it. Recorded as a version deviation; it affects only
   the demo CLI, not training or evaluation.

---

## Environment

Two isolated venvs under the project root. Existing environments
(`rlhf_lab/env`, `rlhf_lab/vllm_env`, `tts_lab/env`) are not modified —
`qwen-asr` pins `transformers==4.57.6` while `rlhf_lab/env` carries 5.6.0, so
sharing was never an option.

| Env | Purpose | Key pins |
|---|---|---|
| `env_asr` | training + evaluation | torch 2.9.1, transformers 4.57.6, qwen-asr 0.0.6, peft, gradio 5.50.0 |
| `env_tts` | data generation only | transformers 4.57.3, qwen-tts |

`env_asr` install **in progress**. `env_tts` not yet created.

---

## Next actions, in order

1. Finish `env_asr`; run `scripts/audit_environment.py` on the H20 → remote `environment.txt`
2. Run `scripts/inspect_model.py` → `results/model_module_inventory.csv` (Phase 1)
3. Run `tests/test_trainable_parameters.py --arm all` (Phase 11)
4. Build `env_tts`, download Qwen3-TTS VoiceDesign + CustomVoice, compare 20 samples (Phase 5 decision)
5. Render anchors → 20-sample TTS smoke test → first real manifest → tiny zero-shot eval (Phase 8)
