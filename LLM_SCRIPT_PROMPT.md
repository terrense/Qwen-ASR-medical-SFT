# 台词生成提示词（交给 ChatGPT）

用途：产出一批**新的**医院域中文口语台词，供 ① Qwen3-TTS 合成，或 ② 真人朗读录制。
产出格式与 `data/scripts/all_scripts.jsonl` 兼容，回收后走既有去重/切分/合成链路。

配套文档：合成侧看 `TTS_GENERATION_SPEC.md`，状态看 `EXPERIMENT_STATUS.md`。

---

## 1. 为什么要这一批（不是推翻 Phase 3，是补它的短板）

现有 17,986 条是**槽位组合模板**生成的（71 family / 9 category，均长 10.8 字）。
它保证了覆盖度和可复现性，但抽查暴露出三类模板固有缺陷，LLM 生成正好补这一块：

| 缺陷 | 现有语料中的真实例子 |
|---|---|
| 拼接叠字 | `已经断断续续有半年了了`、`吃辣的的时候特别头疼` |
| 语义不自洽（槽位随机组合） | `孩子四十七岁半`、`心功能不全用乳果糖管用吗` |
| 概念张冠李戴 | `我有卵巢囊肿，医生让做ICU`（ICU 不是检查项目） |

所以这批的**唯一卖点是自然度和语义自洽**。如果生成出来仍是模板腔，这批就没有价值。

---

## 2. 两条路线，先选一条（决定 `template_family` 怎么填）

切分是**按 template_family 整族划分**的（`scripts/split_data.py`），不是按句子。因此：

**路线 A — 扩充 TTS 训练量**
`template_family` 必须**复用现有 71 个族名之一**（见第 5 节清单）。新句子自动继承该族已有的 train/dev/test 归属，不需要重新切分。

**路线 B — 真人朗读的 held-out 测试集（推荐）**
`template_family` 用**全新族名**，统一加前缀 `RS_`（real speech），例如 `RS_CC_SELF_REPORT`。
新族与训练族天然不相交，整族划入 test，可直接支撑论文里"合成数据训练 → 真人语音测试"的迁移结论。
规模建议 **600–1000 条**：每条 1–4 秒，算上报 ID、重读和间隔，实际约 15–20 秒/条，600 条≈单人 3 小时。

同一次生成里**不要混用两条路线**。

---

## 3. 提示词正文（直接复制粘贴给 ChatGPT）

> 分批跑：**一次只做一个 `domain_category`，每批 ≤ 120 条**。一次要几百条会让模型后半段退化成重复句式。
> 下面 `【】` 里的内容按批次替换。

```
你是中文医院场景的语料工程师。请为语音识别（ASR）数据集撰写患者/家属在医院里真实会说出口的短句。

## 任务
生成 【120】 条 domain_category = 【chief_complaint】 的中文口语句子。
可用的 template_family：【RS_CC_SELF_REPORT / RS_CC_FOR_FAMILY / RS_CC_ASK_SERIOUS】
（每条句子必须归到其中一个族；各族条数尽量均衡。）

## 场景设定
说话人是来门诊/急诊的患者本人或陪同家属，对象是导医台、挂号窗口、护士或医生。
是**说出来的话**，不是写下来的字：短、直接、有口语黏着（"我这个""麻烦问一下""大夫"），不要书面语，不要完整的主谓宾学生腔。

## 硬约束（违反任意一条，整批作废）
1. 长度 4–22 个汉字，整批平均落在 10–13 字。
2. **数字一律写成中文口语形式**：三点五、一米六二、二十毫克、周六上午。禁止阿拉伯数字（3.5、20mg）。
3. 英文缩写保持英文原样大写：CT、MRI、ALT、CEA、B超。不要展开成中文，也不要加空格。
4. 标点：句中可用逗号，**句末不加任何标点**（不加句号、问号、感叹号）。省略号只在表示口吃/自我更正时用。
5. **语义必须自洽**，这是重点：
   - 药物要对得上适应症（别让降糖药去治心衰）
   - 检查项目要对得上部位和科室（ICU、CCU 是病房不是检查项目）
   - 称谓要对得上年龄（"孩子"不会四十七岁）
   - 症状组合要是临床上真会同时出现的
   写完每一句自问："真人会这么说吗？说了会不会被听懂？"
6. 不出现任何真实或虚构的：人名、电话号、身份证号、就诊卡号、具体医院名称。
   楼层/科室可以说（"三楼""消化内科"），但不要编造具体门牌号。
7. 不要复述、不要改写下面给出的示例句；示例只用来对齐语气和长度。
8. 同一批内不得出现雷同句式扎堆：任意开头词（如"我"、"请问"）不超过全批的四分之一。

## 语气与句式要有分布
自述 / 提问 / 请求 / 确认 四类都要有；礼貌开场（"麻烦您""大夫"）约三成，其余直接进入正题。
允许并鼓励真实口语现象：轻微重复、自我更正、语气词。但**不要每句都加**，控制在一成以内。

## 参考示例（对齐语气，不要模仿具体内容）
【此处粘贴该 category 的 2–3 条现有例句，见本文档第 5 节】

## 医学词汇
优先使用常见门诊词汇。可参考仓库 `data/medical_lexicon.json`（417 个术语，分 imaging_exam / lab / medication / disease 等类）。
罕见病、超说明书用法、复杂术语不要出现——这是导医台前的对话，不是病例讨论。

## 输出格式
只输出 JSONL，一行一条，不要 markdown 代码块，不要编号，不要任何解释文字。
每行恰好三个字段：

{"text":"我这两天右边肋骨底下发胀","domain_category":"chief_complaint","template_family":"RS_CC_SELF_REPORT"}

不要输出 script_id、不要输出字数统计——这两项由本地脚本生成。
```

---

## 4. 如果是给真人朗读（路线 B 追加）

### 4.1 给读稿人的稿件

从回收后的 JSONL 转成朗读单，每行 `序号 + script_id + 句子`。分页，每页 ≤ 25 条。

### 4.2 录音须知（务必随稿件一起发出去）

⚠️ **上一批 1,987 条真人录音就是因为下面第 3 条被判废弃**——实测发现经过噪声门 + 高频滤除，不是原始采集，无法用作 ASR 评测参照。

1. 安静房间，关空调外机/风扇；嘴离麦克风约 15 cm，全程别变。
2. 采样率 **≥16 kHz、单声道、WAV 或无损**。手机录音 app 选"无损/PCM"，不要 m4a 有损压缩。
3. **关闭一切降噪、回声消除、自动增益、EQ、"人声增强"**。宁可底噪大一点，也不要处理过的干净音——处理过的音等于废件。
4. 每条句子读之前**先口播一遍 script_id**（"S C R 横杠 …"），再停顿一秒读正文。方便后期切分对齐。
5. 读错了就停下来，**重读整句**，不要只补半句。后期取最后一遍。
6. **按字面读**，不要自己改词、不要加"呃""那个"（除非稿子里本来就有）、不要加感情色彩。
7. 每人一个独立文件夹，文件名 `{说话人编号}_{页码}.wav`；连续录一整页即可，不必一句一文件。

### 4.3 转写校对（不能省）

真人一定会念错、漏字、改口。录完必须逐条听 + 校对，把 manifest 里的 `text` 改成**实际念出来的字**。
`text` 与音频不一致会直接让 CER 失真——这条比什么都重要。

---

## 5. 现有 9 个 category 与 71 个 family（路线 A 用族名，两条路线都用来抄示例）

| category | 占比 | 现有 family（数量） | 可粘进提示词的示例 |
|---|---|---|---|
| chief_complaint | 25% | CC_WITH_HISTORY, CC_MED_NO_EFFECT, CC_TWO_SYMPTOMS, CC_SYMPTOM_DURATION, CC_FAMILY_MEMBER, CC_TRIGGER, CC_ASK_SERIOUS, CC_HEDGED, CC_PAIN_SITE_DURATION, CC_PAIN_SITE_QUALITY, CC_NIGHT, CC_AFTER_MEAL, CC_WORSENING, CC_ONSET_SUDDEN (14) | `我妈多梦，半个月了` / `大概有点黑便` / `今天早上突然咳血` |
| examination | 15% | EX_LAB_TWO, EX_EXAM_AND_LAB, EX_BOOK, EX_NEED, EX_LAB_DETAIL, EX_LAB_ORDER, EX_PRICE, EX_PREP, EX_WHERE, EX_CONTRAST, EX_RESULT (11) | `心电图要提前多久到` / `我来取腹部平片的报告` |
| navigation | 10% | NAV_ANSWER_FLOOR, NAV_DEPT_FLOOR, NAV_WHERE_FACILITY, NAV_HOURS, NAV_ROUTE, NAV_PROCESS (6) | `麻烦您，消化内科怎么走` / `服务台是往左还是往右` |
| medication | 10% | MED_DOSAGE, MED_INTERACTION, MED_FOR_DISEASE, MED_ADHERENCE, MED_SIDE_EFFECT, MED_REFILL, MED_HOW_TO_TAKE (7) | `别嘌醇有什么副作用` / `利伐沙班可以停吗` |
| disease | 10% | DIS_COMORBID, DIS_HISTORY_YEARS, DIS_LIFESTYLE, DIS_FAMILY, DIS_ASK_SEVERITY, DIS_DIAGNOSED (6) | `肺炎平时饮食要注意什么` / `我父亲有干眼症` |
| registration | 10% | REG_WHICH_DEPT, REG_CHILD_ELDER, REG_BOOK_DEPT, REG_EXPERT, REG_CANCEL_CHANGE, REG_PROCEDURE (6) | `第一次来需要办卡吗` / `我想取消下周三上午的号` |
| numeric | 10% | NUM_DOSE_COUNT, NUM_LAB_VALUE, NUM_DATE, NUM_VITAL_VALUE, NUM_PRICE, NUM_AGE_WEIGHT, NUM_IDENTIFIER (7) | `一天三次，一次六毫克` / `今天早上量的身高是一米六二` |
| disfluency | 5% | DF_CORRECT_MED, DF_RESTART, DF_FILLER_SYMPTOM, DF_CORRECT_EXAM, DF_CORRECT_DEPT, DF_CORRECT_NUMBER, DF_REPEAT (7) | `开文拉法辛，啊不是，塞来昔布` / `膝关节核磁...哦不对，是B超` |
| code_switch | 5% | CS_ABBR_DISEASE, CS_ABBR_DEPT, CS_ABBR_TWO, CS_ABBR_TIME, CS_ABBR_RESULT, CS_ABBR_NEED, CS_ABBR_EXPAND (7) | `CEA和US都要做吗` / `CRP就是C反应蛋白吧` |

新一批建议沿用这个配比，除非只做真人小集（那时可以压缩到 chief_complaint / registration / navigation / numeric 四类）。

---

## 6. 回收流程（本地做，不要指望 ChatGPT 自己保证）

ChatGPT 一定会产生重复句和越界句，**去重和校验一律在本地跑**，复用已验收的代码路径：

1. 合并各批 JSONL → `data/scripts/llm_raw.jsonl`
2. 补 `script_id`：`"SCR-%s-%s" % (template_family, sha1(text)[:12])` —— 与 `src/data/corpus_generator.py:662` 完全一致
3. 补 `n_chars`，加 `"source": "llm"` 区分来源
4. 组内 + 跨 `all_scripts.jsonl` 近重复过滤：复用 `corpus_generator.MinHashLSH(threshold=0.9)` 与 `normalize_for_dedup`
5. 硬校验：长度 4–22、无阿拉伯数字、句末无标点、字段齐全、family 名合法
6. 路线 B 额外断言：新 family 与 `all_scripts.jsonl` 的 71 个族**零交集**

第 2–6 步是一个小脚本（`scripts/ingest_llm_scripts.py`，尚未编写）。

---

## 7. 验收清单

- [ ] 每批条数 = 要求条数，无截断
- [ ] 长度均值 10–13 字，无 <4 或 >22
- [ ] 零阿拉伯数字，零句末标点
- [ ] 英文缩写保持大写英文
- [ ] 抽查 30 条人工判语义自洽（重点看药-病、检查-部位、称谓-年龄）
- [ ] 无人名/电话/证件号/真实医院名
- [ ] 跨 `all_scripts.jsonl` 近重复率 < 2%
- [ ] 路线 B：新 family 与旧 71 族零交集
- [ ] 真人集：`text` 已按实际念出内容逐条校对
