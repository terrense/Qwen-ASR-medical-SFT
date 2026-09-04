# TTS 合成任务交接规格

给负责语音合成的同事。本文档自带全部已验证结论，按此执行即可，不需要读其余代码。

有疑问先看第 7 节「验收清单」，产出对不上清单里任何一条都会导致下游训练无法使用。

---

## 0. 一句话任务

把 `data/manifests/splits/*_scripts.jsonl` 里的中文句子，用 **Qwen3-TTS VoiceDesign + 声纹克隆**合成成 16 kHz 单声道 WAV，产出符合第 4 节格式的 manifest。

**环境和脚本都已就绪**，正常情况下只需要跑第 6 节的两条命令。

---

## 1. 已经验证过的结论（请勿推翻重来）

这三条是实测出来的，改动会直接破坏实验设计。

### 1.1 必须用 anchor + 声纹克隆，不能逐句调 VoiceDesign

VoiceDesign 每次调用会重新解释音色描述，**同一个身份的音高会在句子之间乱跳**。实测 6 个身份 × 4 句：

| 流程 | 同身份 F0 波动(std) | 身份间差距(std) | 比值(越大越好) |
|---|---|---|---|
| 逐句直接调 VoiceDesign | 37.80 | 45.52 | 1.20 |
| **anchor + x-vector 克隆** | **16.34** | 41.89 | **2.56** |

极端例子：SPK05（设计为年长男声）直接调用时四句 F0 是 `83 / 251 / 117 / 185 Hz`（男声跳到女声）；改用克隆后收敛到 `101 / 155 / 121 / 152 Hz`。

**正确流程**（`scripts/generate_tts.py` 已实现）：

```
每个身份:
  1. 用 VoiceDesign + 该身份的音色描述, 合成 1 条中性锚点句  -> anchors/SPKxx.wav
  2. base_model.create_voice_clone_prompt(ref_audio=锚点, x_vector_only_mode=True)
  3. 该身份的所有语料句一律用 base_model.generate_voice_clone(voice_clone_prompt=第2步的结果)
```

锚点句是**领域中性**的日常句（不是医院用语），目的是让声纹编码音色而不是编码词汇。

### 1.2 不要用 CustomVoice

实测已排除，两个原因：

- **只有 9 个内置音色**（`aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu, vivian`），本项目需要 32 个，且无法控制年龄/口音维度。
- **中文上会失控**：`ryan` 把一句短句念成 **76.24 秒**（QC 直接判失败），`ono_anna` 念成 22.32 秒。这些是英/日/韩语音色。

### 1.3 不要用 speaker embedding 的余弦相似度来判断音色是否不同

Base 模型 speaker encoder 的输出在这里**没有判别力**：它给 F0 相差 3 倍（89.6 Hz vs 243.1 Hz）的两个音色打 0.95 相似度。如果需要检查音色，用**基频 F0**（`scripts/verify_identity_stability.py` 里有现成实现）。

---

## 2. 说话人身份（32 个，已定义好，不要改）

定义在 `src/data/speakers.py`，运行 `build_speaker_inventory(seed=42)` 得到。属性是按固定步长排布的，不是随机采样，所以可复现。

- 性别 16/16 完全平衡；口音四类各 8 个
- 分池：**train 20 / dev 6 / test 6**，且经过分层，保证 dev 和 test 各自都覆盖全部 4 种口音

> ⚠️ **三个池子绝对不能混用。** train 的音色一旦出现在 dev/test，说话人无关性就没了，整个实验的测试集失效。`generate_tts.py` 已按 `--split` 自动选池，不要手工指定音色。

每个身份的音色描述形如：

```
一位中年男声，音色低沉，语速偏慢，带一点西南官话口音的普通话，说话语气平稳，吐字清晰自然。
```

---

## 3. 要合成什么、合成多少

| 输入脚本文件 | 条数 | 用途 | 音色池 |
|---|---|---|---|
| `splits/train_scripts.jsonl` | 13,520 | 训练 | train (20) |
| `splits/dev_scripts.jsonl` | 1,739 | 验证 | dev (6) |
| `splits/test_scripts.jsonl` | 2,727 | 测试 | test (6) |
| `splits/cross_tts_scripts.jsonl` | 360 | 跨引擎测试 | **用 CosyVoice3，不是 Qwen3-TTS** |

### 训练集要合成两遍

句子偏短（平均 12 字，实测每条约 1–4 秒音频）。13,520 条每条只配一个音色，总量只有约 13.5–17 小时，**达不到 D20 = 20 小时的预算**。

所以 **train 每条脚本要用 2 个不同音色各合成一次**，得到约 27,040 条 ≈ 30 小时。dev / test 各合成一遍即可。

命名要能区分：`{script_id}_{speaker_id}.wav`（脚本已如此实现）。

---

## 4. 产出格式

每个 split 一个 JSONL，一行一条：

```json
{"utt_id":"SCR-CC_SYMPTOM_DURATION-a1b2c3_SPK07","audio":"/绝对路径/xxx.wav",
 "text":"我头疼三天了","speaker_id":"SPK07","source":"synthetic_qwen3tts",
 "tts_engine":"qwen3-tts-12hz-1.7b (voicedesign anchor + x-vector clone)",
 "domain_category":"chief_complaint","template_family":"CC_SYMPTOM_DURATION",
 "duration":3.42,"condition":"clean","snr":null,"sir":null,
 "script_id":"SCR-CC_SYMPTOM_DURATION-a1b2c3","split":"train"}
```

- `text` 必须和输入脚本**逐字一致**，不要加标点、不要改写、不要规范化数字
- `duration` 必须是**实际测得**的秒数，不是估计值（下游按音频小时数切分预算，估错会导致 D1/D5/D10/D20 全错）
- `condition` 一律填 `"clean"`；加噪、混响等由下游 `augment_corpus.py` 统一处理，**不要在合成阶段加任何噪声**

音频：**16 kHz、单声道、PCM 16bit WAV**。

---

## 5. 质量控制

`generate_tts.py` 已自动执行，被拒绝的样本写入 `removed_<split>.jsonl` 并附原因。

| 检查 | 阈值 |
|---|---|
| 时长 | 0.3 s ~ 30 s |
| 削波比例 | ≤ 1% |
| 静音比例 | ≤ 85% |
| RMS | 1e-4 ~ 0.99 |
| 首/尾静音 | ≤ 2 s（超出会自动裁剪，保留 0.15 s 余量） |
| NaN / 全零 | 直接拒绝 |

> ⚠️ **绝对不要用 Qwen3-ASR 的识别结果来筛掉"识别不好"的音频。** 那等于用被测模型来挑选测试集，会系统性地美化最终结果，是本项目明确禁止的做法。QC 只看波形本身。

**移除率超过 2% 请先反馈再继续**，通常意味着某个身份的锚点有问题。

---

## 6. 怎么跑

环境已就绪：`/data/shenxin/qwen3_asr_hospital/env_tts`（transformers 4.57.3 + qwen-tts 0.1.1）
模型已下载：`models/Qwen3-TTS-12Hz-1.7B-{VoiceDesign,Base,CustomVoice}`

```bash
cd /data/shenxin/qwen3_asr_hospital

# 第一步：生成 32 个锚点（约 2 分钟）。产出 data/synthetic/speakers.json
./env_tts/bin/python scripts/generate_tts.py --stage anchors

# 先检查锚点：确认没有 anchor_qc_passed=false 的身份，有的话重跑那几个
# 一个坏锚点会污染该身份的全部语料

# 第二步：先跑 20 条冒烟，确认没问题再铺开
./env_tts/bin/python scripts/generate_tts.py --stage corpus --split train --limit 20

# 第三步：全量
./env_tts/bin/python scripts/generate_tts.py --stage corpus --split train
./env_tts/bin/python scripts/generate_tts.py --stage corpus --split dev
./env_tts/bin/python scripts/generate_tts.py --stage corpus --split test
```

### 吞吐与提速

实测 **batch=1 时 4.48 秒/条**。按这个速度，27,040 条训练音频要 **33 小时以上**，太慢。

`generate_tts.py` 支持 `--batch_size`（默认 8，按身份分组批量送入）。**请先实测 batch 加速比**再决定全量怎么跑：

```bash
# 比较 batch 1 / 8 / 16 的实际吞吐
for b in 1 8 16; do
  ./env_tts/bin/python scripts/generate_tts.py --stage corpus --split dev --limit 64 --batch_size $b
done
```

显存充裕（H20 有 143 GB，1.7B 模型只占约 4 GB），batch 可以往大了试。如果 batch=16 仍然不够快，再讨论多进程或多卡。

---

## 7. 验收清单

交付前逐条自查：

- [ ] 每个 split 一个 JSONL，字段齐全（第 4 节）
- [ ] `text` 与输入脚本逐字一致（可用 `script_id` 对回去比）
- [ ] `duration` 是实测值，且 manifest 里的总时长 ≥ 20 小时（train）
- [ ] 音频全部 16 kHz / 单声道 / PCM16
- [ ] **train / dev / test 三个池的 speaker_id 集合互不相交**（最关键的一条）
- [ ] train 每个 script_id 恰好出现 2 次，且两次的 speaker_id 不同
- [ ] `condition` 全部为 `"clean"`，`snr`/`sir` 全部为 `null`
- [ ] QC 移除率 < 2%，`removed_*.jsonl` 一并交付
- [ ] `data/synthetic/speakers.json` 一并交付（含 32 个锚点的 SHA-256，用于复现）

自查命令：

```bash
cd /data/shenxin/qwen3_asr_hospital
./env_asr/bin/python -c "
import json,collections
S={}
for sp in ['train','dev','test']:
    rows=[json.loads(l) for l in open('data/manifests/%s_synthetic.jsonl'%sp,encoding='utf-8') if l.strip()]
    S[sp]={r['speaker_id'] for r in rows}
    h=sum(r['duration'] for r in rows)/3600
    c=collections.Counter(r['script_id'] for r in rows)
    print('%-5s %6d条 %6.2f小时 %2d音色 每脚本次数=%s'%(sp,len(rows),h,len(S[sp]),sorted(set(c.values()))))
for a,b in [('train','dev'),('train','test'),('dev','test')]:
    ov=S[a]&S[b]
    print('%s∩%s = %s %s'%(a,b,ov if ov else '空','<-- 必须为空' if ov else 'OK'))
"
```

---

## 8. 跨引擎测试集（CosyVoice3）

`cross_tts_scripts.jsonl` 那 360 条**必须用另一个引擎**合成，用来测试模型对合成器的过拟合程度。

服务器上已有：`/data/shenxin/tts_lab/models/Fun-CosyVoice3-0.5B-2512`，环境 `/data/shenxin/tts_lab/env`。

要求同上，但：
- `source` 填 `"synthetic_cosyvoice3"`
- `tts_engine` 填 `"Fun-CosyVoice3-0.5B-2512"`
- 音色用 CosyVoice3 自己的，和 Qwen3-TTS 天然不相交
- 输出到 `data/manifests/test_cross_tts.jsonl`

**这批数据绝对不能进入训练或调参**，只用于最终测试。

---

## 9. 遇到问题

- 锚点 QC 不过 → 重跑该身份的锚点，或微调 `speakers.py` 里那条音色描述（改了要记录）
- 某个身份合成出来明显不像描述（比如"男声"出来是女声）→ 用 `scripts/verify_identity_stability.py` 量 F0 确认，别靠听感
- 移除率 > 2% → 先反馈，不要自行放宽 QC 阈值
- 想改任何 QC 阈值或流程 → 先讨论，这些数字会写进论文的方法部分
