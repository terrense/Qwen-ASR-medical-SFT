# -*- coding: utf-8 -*-
"""Phase 3 - Hospital-domain Mandarin utterance generator.

Generates original spoken-Mandarin utterances from parameterized template
families. Nothing is copied from web consultations: every sentence is produced
by filling a hand-written spoken pattern with vocabulary from
``hospital_vocab.py``.

Two properties matter for the experiment design:

  template_family
      A semantic pattern id. Train/dev/test are split by template family
      (Phase 4), not merely by waveform, so a test utterance is never a
      re-voicing of a sentence pattern the model was trained on.

  uniqueness
      Exact duplicates are removed on the normalized string. Near-duplicates are
      removed with MinHash/LSH over character 3-grams. The threshold is
      deliberately high (0.9): two utterances that differ only in the medical
      term ("我头疼三天了" vs "我胃疼三天了") are *wanted* variation, not
      duplicates, and must survive. Every removal is recorded with its reason.
"""
from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import Counter, OrderedDict

from . import hospital_vocab as V
from .manifest import DOMAIN_CATEGORIES as V_DOMAIN

# ---------------------------------------------------------------------------
# Template families
# ---------------------------------------------------------------------------
# Each family: id, domain category, list of patterns, and the slot vocabularies
# the patterns draw from. Slot names ending in a digit reuse the same vocabulary
# with a distinct draw (e.g. symptom / symptom2 must differ).

S = V.SYMPTOMS
B = V.BODY_PARTS
Q = V.PAIN_QUALITY
D = V.DEPARTMENTS
F = V.FACILITIES
E = V.IMAGING_EXAMS
L = V.LAB_TESTS
DIS = V.DISEASES
MED = V.MEDICATIONS
DUR = V.DURATIONS
TS = V.TIME_SLOTS
AB = V.ABBREVIATIONS

TEMPLATE_FAMILIES = [
    # ---------------- chief complaints and symptoms (25%) ------------------
    {"id": "CC_SYMPTOM_DURATION", "category": "chief_complaint",
     "patterns": ["我{symptom}{duration}了", "我{symptom}，已经{duration}了",
                  "{symptom}{duration}了，一直没好"],
     "slots": {"symptom": S, "duration": DUR}},
    {"id": "CC_PAIN_SITE_QUALITY", "category": "chief_complaint",
     "patterns": ["我{part}{quality}", "{part}这里{quality}",
                  "我这个{part}{quality}，特别难受"],
     "slots": {"part": B, "quality": Q}},
    {"id": "CC_PAIN_SITE_DURATION", "category": "chief_complaint",
     "patterns": ["我{part}疼{duration}了", "{part}疼了{duration}",
                  "我{part}疼，{duration}了"],
     "slots": {"part": B, "duration": DUR}},
    {"id": "CC_TWO_SYMPTOMS", "category": "chief_complaint",
     "patterns": ["我{symptom}，还{symptom2}", "最近老是{symptom}，有时候还{symptom2}",
                  "{symptom}，伴着{symptom2}"],
     "slots": {"symptom": S, "symptom2": S}},
    {"id": "CC_ONSET_SUDDEN", "category": "chief_complaint",
     "patterns": ["今天早上突然{symptom}", "昨天晚上开始{symptom}",
                  "刚才一下子就{symptom}了"],
     "slots": {"symptom": S}},
    {"id": "CC_WORSENING", "category": "chief_complaint",
     "patterns": ["{symptom}，最近越来越厉害了", "这个{symptom}比上个月严重多了",
                  "{symptom}，一天比一天重"],
     "slots": {"symptom": S}},
    {"id": "CC_AFTER_MEAL", "category": "chief_complaint",
     "patterns": ["吃完饭以后{symptom}", "一吃东西就{symptom}",
                  "空着肚子的时候{symptom}"],
     "slots": {"symptom": S}},
    {"id": "CC_NIGHT", "category": "chief_complaint",
     "patterns": ["晚上{symptom}得厉害，白天好一点", "夜里总是{symptom}，睡不好",
                  "一到晚上就{symptom}"],
     "slots": {"symptom": S}},
    {"id": "CC_WITH_HISTORY", "category": "chief_complaint",
     "patterns": ["我有{disease}，最近老{symptom}", "我本身有{disease}，这两天又{symptom}",
                  "有{disease}病史，现在{symptom}"],
     "slots": {"disease": DIS, "symptom": S}},
    {"id": "CC_MED_NO_EFFECT", "category": "chief_complaint",
     "patterns": ["吃了{medication}还是{symptom}", "用了{medication}，{symptom}没见好",
                  "{medication}吃了三天了，还是{symptom}"],
     "slots": {"medication": MED, "symptom": S}},
    {"id": "CC_FAMILY_MEMBER", "category": "chief_complaint",
     "patterns": ["孩子{symptom}{duration}了", "我妈{symptom}，{duration}了",
                  "我爱人{symptom}，{duration}"],
     "slots": {"symptom": S, "duration": DUR}},
    {"id": "CC_ASK_SERIOUS", "category": "chief_complaint",
     "patterns": ["{opener}，我{symptom}，要紧吗", "我{symptom}，严重不严重",
                  "{opener}，这个{symptom}需要住院吗"],
     "slots": {"opener": V.POLITE_OPENERS, "symptom": S}},
    {"id": "CC_HEDGED", "category": "chief_complaint",
     "patterns": ["我{hedge}是{symptom}", "{hedge}有点{symptom}",
                  "我{hedge}最近{symptom}比较多"],
     "slots": {"hedge": V.HEDGES, "symptom": S}},
    {"id": "CC_TRIGGER", "category": "chief_complaint",
     "patterns": ["一{trigger}就{symptom}", "只要{trigger}，马上就{symptom}",
                  "{trigger}的时候特别{symptom}"],
     "slots": {"trigger": ["走路", "爬楼梯", "受凉", "熬夜", "生气", "吃辣的",
                           "喝酒", "运动", "低头", "起床", "蹲下再站起来",
                           "天气变化", "干活儿", "着急"],
               "symptom": S}},

    # ---------------- examinations and laboratory tests (15%) --------------
    {"id": "EX_NEED", "category": "examination",
     "patterns": ["{opener}，我这个情况需要做{exam}吗", "我用不用做个{exam}",
                  "是不是得拍个{exam}"],
     "slots": {"opener": V.POLITE_OPENERS, "exam": E}},
    {"id": "EX_BOOK", "category": "examination",
     "patterns": ["我想约一个{exam}", "帮我预约{time}的{exam}",
                  "{exam}能不能约到{time}"],
     "slots": {"exam": E, "time": TS}},
    {"id": "EX_WHERE", "category": "examination",
     "patterns": ["{exam}在哪里做", "做{exam}去几楼", "{exam}是在哪个科室做"],
     "slots": {"exam": E}},
    {"id": "EX_PREP", "category": "examination",
     "patterns": ["做{exam}需要空腹吗", "做{exam}之前要注意什么",
                  "{exam}要提前多久到"],
     "slots": {"exam": E}},
    {"id": "EX_RESULT", "category": "examination",
     "patterns": ["{exam}的结果什么时候出来", "我来取{exam}的报告",
                  "{exam}的片子在哪儿拿"],
     "slots": {"exam": E}},
    {"id": "EX_PRICE", "category": "examination",
     "patterns": ["{exam}大概多少钱", "做{exam}能报销吗", "{exam}医保能报多少"],
     "slots": {"exam": E}},
    {"id": "EX_CONTRAST", "category": "examination",
     "patterns": ["做{exam}要打造影剂吗", "{exam}是平扫还是增强",
                  "{exam}有辐射吗"],
     "slots": {"exam": E}},
    {"id": "EX_LAB_ORDER", "category": "examination",
     "patterns": ["医生给我开了{lab}", "我要查一个{lab}", "帮我加一个{lab}"],
     "slots": {"lab": L}},
    {"id": "EX_LAB_DETAIL", "category": "examination",
     "patterns": ["{lab}要抽几管血", "{lab}需要空腹吗", "{lab}多久出结果"],
     "slots": {"lab": L}},
    {"id": "EX_LAB_TWO", "category": "examination",
     "patterns": ["查一下{lab}和{lab2}", "{lab}、{lab2}都要做",
                  "先抽血查{lab}，再查{lab2}"],
     "slots": {"lab": L, "lab2": L}},
    {"id": "EX_EXAM_AND_LAB", "category": "examination",
     "patterns": ["先做{exam}，再查{lab}", "{exam}和{lab}今天都能做吗",
                  "医生开了{exam}还有{lab}"],
     "slots": {"exam": E, "lab": L}},

    # ---------------- registration and department selection (10%) ----------
    {"id": "REG_WHICH_DEPT", "category": "registration",
     "patterns": ["我{symptom}应该挂哪个科", "{symptom}挂什么科",
                  "{opener}，{symptom}是看{dept}吗"],
     "slots": {"symptom": S, "dept": D, "opener": V.POLITE_OPENERS}},
    {"id": "REG_BOOK_DEPT", "category": "registration",
     "patterns": ["我想挂{time}的{dept}", "帮我挂个{dept}",
                  "{dept}{time}还有号吗"],
     "slots": {"dept": D, "time": TS}},
    {"id": "REG_EXPERT", "category": "registration",
     "patterns": ["{dept}有专家号吗", "{dept}的主任今天出诊吗",
                  "我想挂{dept}的专家门诊"],
     "slots": {"dept": D}},
    {"id": "REG_CANCEL_CHANGE", "category": "registration",
     "patterns": ["我想取消{time}的号", "能不能把号改到{time}",
                  "{time}的号我来不了，可以退吗"],
     "slots": {"time": TS}},
    {"id": "REG_PROCEDURE", "category": "registration",
     "patterns": ["第一次来需要办卡吗", "挂号在哪个窗口", "怎么用手机预约挂号",
                  "自助机怎么取号", "没带身份证能挂号吗"],
     "slots": {}},
    {"id": "REG_CHILD_ELDER", "category": "registration",
     "patterns": ["孩子{symptom}挂儿科还是{dept}", "老人{symptom}挂{dept}行吗",
                  "小孩看{dept}要挂儿童专科吗"],
     "slots": {"symptom": S, "dept": D}},

    # ---------------- navigation and procedures (10%) ----------------------
    {"id": "NAV_WHERE_FACILITY", "category": "navigation",
     "patterns": ["{facility}在哪儿", "请问{facility}怎么走", "{facility}在哪个方向"],
     "slots": {"facility": F}},
    {"id": "NAV_DEPT_FLOOR", "category": "navigation",
     "patterns": ["{dept}在几楼", "{dept}门诊在哪一层", "{opener}，{dept}怎么走"],
     "slots": {"dept": D, "opener": V.POLITE_OPENERS}},
    {"id": "NAV_ANSWER_FLOOR", "category": "navigation",
     "patterns": ["{dept}在{building}{floor}", "您到{building}{floor}就是{facility}",
                  "{facility}在{floor}，出电梯右转"],
     "slots": {"dept": D, "building": V.BUILDINGS, "floor": V.FLOORS,
               "facility": F}},
    {"id": "NAV_ROUTE", "category": "navigation",
     "patterns": ["怎么去{facility}", "从这里到{facility}怎么走",
                  "{facility}是往左还是往右"],
     "slots": {"facility": F}},
    {"id": "NAV_HOURS", "category": "navigation",
     "patterns": ["{facility}几点上班", "{facility}中午休息吗",
                  "{facility}周末开门吗"],
     "slots": {"facility": F}},
    {"id": "NAV_PROCESS", "category": "navigation",
     "patterns": ["缴费在哪里办", "出院手续在哪儿办", "病历怎么复印",
                  "住院要办什么手续", "医保报销在哪个窗口", "怎么打印检查报告"],
     "slots": {}},

    # ---------------- diseases (10%) ---------------------------------------
    {"id": "DIS_DIAGNOSED", "category": "disease",
     "patterns": ["医生说我是{disease}", "上次诊断是{disease}",
                  "报告上写的{disease}"],
     "slots": {"disease": DIS}},
    {"id": "DIS_ASK_SEVERITY", "category": "disease",
     "patterns": ["{disease}严重吗", "{disease}能治好吗", "{disease}会不会复发"],
     "slots": {"disease": DIS}},
    {"id": "DIS_HISTORY_YEARS", "category": "disease",
     "patterns": ["我有{disease}好几年了", "{disease}有{duration}了",
                  "我{disease}是去年查出来的"],
     "slots": {"disease": DIS, "duration": DUR}},
    {"id": "DIS_FAMILY", "category": "disease",
     "patterns": ["我父亲有{disease}", "家里人有{disease}史",
                  "我妈也是{disease}"],
     "slots": {"disease": DIS}},
    {"id": "DIS_LIFESTYLE", "category": "disease",
     "patterns": ["{disease}平时饮食要注意什么", "{disease}能不能运动",
                  "{disease}要忌口吗"],
     "slots": {"disease": DIS}},
    {"id": "DIS_COMORBID", "category": "disease",
     "patterns": ["{disease}会不会引起{disease2}", "我又有{disease}又有{disease2}",
                  "{disease}和{disease2}有关系吗"],
     "slots": {"disease": DIS, "disease2": DIS}},

    # ---------------- medication and treatment (10%) -----------------------
    {"id": "MED_HOW_TO_TAKE", "category": "medication",
     "patterns": ["{medication}怎么吃", "{medication}饭前吃还是饭后吃",
                  "{medication}要吃多久"],
     "slots": {"medication": MED}},
    {"id": "MED_DOSAGE", "category": "medication",
     "patterns": ["{medication}一次{num}{form}，{freq}",
                  "{medication}{freq}，一次{num}{form}",
                  "医生让我{freq}吃{num}{form}{medication}"],
     "slots": {"medication": MED, "num": V.NUMBERS_SMALL,
               "form": V.DOSAGE_FORMS, "freq": V.FREQUENCIES}},
    {"id": "MED_SIDE_EFFECT", "category": "medication",
     "patterns": ["{medication}有什么副作用", "吃{medication}会伤肝吗",
                  "{medication}吃了会不会有反应"],
     "slots": {"medication": MED}},
    {"id": "MED_REFILL", "category": "medication",
     "patterns": ["我来开{medication}", "帮我开一盒{medication}",
                  "药房有没有{medication}"],
     "slots": {"medication": MED}},
    {"id": "MED_INTERACTION", "category": "medication",
     "patterns": ["{medication}和{medication2}能一起吃吗",
                  "吃着{medication}还能加{medication2}吗",
                  "{medication}跟{medication2}冲突吗"],
     "slots": {"medication": MED, "medication2": MED}},
    {"id": "MED_ADHERENCE", "category": "medication",
     "patterns": ["昨天忘记吃{medication}了怎么办", "{medication}可以停吗",
                  "{medication}能不能减量"],
     "slots": {"medication": MED}},
    {"id": "MED_FOR_DISEASE", "category": "medication",
     "patterns": ["{disease}一般吃什么药", "我{disease}，在吃{medication}",
                  "{disease}用{medication}管用吗"],
     "slots": {"disease": DIS, "medication": MED}},

    # ---------------- numbers, dates, measurements (10%) -------------------
    {"id": "NUM_VITAL_VALUE", "category": "numeric",
     "patterns": ["我{vital}{value}", "今天早上量的{vital}是{value}",
                  "{vital}{value}，正常吗"],
     "slots": {"__vital_pair__": True}},
    {"id": "NUM_LAB_VALUE", "category": "numeric",
     "patterns": ["{lab}是{value}", "我的{lab}{value}，高不高",
                  "上次查{lab}是{value}"],
     "slots": {"lab": L,
               "value": ["三点二", "五点八", "六点七", "十二点四", "零点九",
                         "一百二十", "三十五", "四点五", "8.6", "13.2", "0.45",
                         "二十八", "一百零五", "七十六"]}},
    {"id": "NUM_DOSE_COUNT", "category": "numeric",
     "patterns": ["一天{num}次，一次{num2}{form}", "{num}天吃完一盒",
                  "早上{num}片，晚上{num2}片"],
     "slots": {"num": V.NUMBERS_SMALL, "num2": V.NUMBERS_SMALL,
               "form": V.DOSAGE_FORMS}},
    {"id": "NUM_DATE", "category": "numeric",
     "patterns": ["{time}做的检查", "我是{time}来复诊", "{time}住的院",
                  "{month}月{day}号的号"],
     "slots": {"time": TS,
               "month": ["一", "二", "三", "四", "五", "六", "七", "八", "九",
                         "十", "十一", "十二"],
               "day": ["一", "三", "五", "八", "十", "十二", "十五", "十八",
                       "二十", "二十三", "二十六", "二十八", "三十"]}},
    {"id": "NUM_AGE_WEIGHT", "category": "numeric",
     "patterns": ["我今年{age}岁", "孩子{age}岁半", "我体重{weight}",
                  "身高{height}，体重{weight}"],
     "slots": {"age": ["二十六", "三十四", "四十五", "五十二", "六十三",
                       "七十一", "八十", "十八", "二十九", "三十八", "四十七",
                       "五十五", "六十八", "七十六"],
               "weight": ["六十二公斤", "七十五公斤", "一百三十斤",
                          "五十八点五公斤", "九十公斤", "一百斤"],
               "height": ["一米六二", "一米七五", "一百七十厘米", "一米五八",
                          "一米八"]}},
    {"id": "NUM_IDENTIFIER", "category": "numeric",
     "patterns": ["我的门诊号是{digits}", "叫号叫到{digits}了吗",
                  "{digits}号窗口在哪儿"],
     "slots": {"digits": ["零一二三", "三五七", "一百零八", "二十六",
                          "四零九", "七一二", "一三五七", "八八六",
                          "五十三", "九十九", "二零二四", "六六八"]}},
    {"id": "NUM_PRICE", "category": "numeric",
     "patterns": ["一共{price}块{cents}", "挂号费{price}块",
                  "自付部分是{price}块{cents}"],
     "slots": {"price": ["十五", "二十", "三十八", "五十", "一百二",
                         "两百四十", "三百六十五", "八十七", "四十二"],
               "cents": ["五毛", "两毛", "整", "五角", "八毛"]}},

    # ---------------- code switching and abbreviations (5%) ----------------
    {"id": "CS_ABBR_NEED", "category": "code_switch",
     "patterns": ["医生，需要做{abbr}吗", "要不要加做一个{abbr}",
                  "{abbr}这个检查贵吗"],
     "slots": {"abbr": AB}},
    {"id": "CS_ABBR_TWO", "category": "code_switch",
     "patterns": ["先做个{abbr}，再查{abbr2}", "{abbr}和{abbr2}都要做吗",
                  "{abbr}做完了，{abbr2}还没做"],
     "slots": {"abbr": AB, "abbr2": AB}},
    {"id": "CS_ABBR_EXPAND", "category": "code_switch",
     "patterns": ["{abbr}就是{expansion}吧", "{abbr}是不是{expansion}",
                  "医生说的{abbr}，我理解成{expansion}了"],
     "slots": {"__abbrev_pair__": True}},
    {"id": "CS_ABBR_RESULT", "category": "code_switch",
     "patterns": ["我的{abbr}是{value}", "{abbr}结果偏高",
                  "上次{abbr}查出来正常"],
     "slots": {"abbr": AB,
               "value": ["六点二", "七点一", "5.9", "8.3", "正常范围",
                         "偏高一点", "临界值"]}},

    {"id": "CS_ABBR_DEPT", "category": "code_switch",
     "patterns": ["{abbr}检查在{dept}做吗", "{dept}能开{abbr}吗",
                  "{dept}让我去做个{abbr}"],
     "slots": {"abbr": AB, "dept": D}},
    {"id": "CS_ABBR_TIME", "category": "code_switch",
     "patterns": ["{abbr}约到{time}了", "{time}做{abbr}",
                  "{abbr}能不能改到{time}"],
     "slots": {"abbr": AB, "time": TS}},
    {"id": "CS_ABBR_DISEASE", "category": "code_switch",
     "patterns": ["{disease}要查{abbr}吗", "我有{disease}，医生让做{abbr}",
                  "{disease}复查是不是要做{abbr}"],
     "slots": {"abbr": AB, "disease": DIS}},

    # ---------------- spoken disfluency and self-correction (5%) -----------
    {"id": "DF_CORRECT_EXAM", "category": "disfluency",
     "patterns": ["不是{exam}，我说的是{exam2}", "{exam}...哦不对，是{exam2}",
                  "我要做{exam}，啊不是，{exam2}"],
     "slots": {"exam": E, "exam2": E}},
    {"id": "DF_CORRECT_DEPT", "category": "disfluency",
     "patterns": ["挂{dept}...哦不对，是{dept2}", "我要挂{dept}，不是，{dept2}",
                  "{dept}，呃，应该是{dept2}"],
     "slots": {"dept": D, "dept2": D}},
    {"id": "DF_CORRECT_MED", "category": "disfluency",
     "patterns": ["我吃的是{medication}，不对，是{medication2}",
                  "{medication}...嗯，好像是{medication2}",
                  "开{medication}，啊不是，{medication2}"],
     "slots": {"medication": MED, "medication2": MED}},
    {"id": "DF_FILLER_SYMPTOM", "category": "disfluency",
     "patterns": ["{filler}，{filler2}，就是{symptom}",
                  "{filler}我想问一下，我{symptom}",
                  "{filler}，我这个{symptom}，{filler2}，有点严重"],
     "slots": {"filler": V.FILLERS, "filler2": V.FILLERS, "symptom": S}},
    {"id": "DF_REPEAT", "category": "disfluency",
     "patterns": ["我想问一下，我想问一下{dept}在几楼",
                  "这个，这个{exam}要空腹吗",
                  "就是那个，那个{lab}的结果出来没有"],
     "slots": {"dept": D, "exam": E, "lab": L}},
    {"id": "DF_CORRECT_NUMBER", "category": "disfluency",
     "patterns": ["一天三次...啊不是，一天两次", "吃{num}片，不对，{num2}片",
                  "{time}，呃，改成{time2}吧"],
     "slots": {"num": V.NUMBERS_SMALL, "num2": V.NUMBERS_SMALL,
               "time": TS, "time2": TS}},
    {"id": "DF_RESTART", "category": "disfluency",
     "patterns": ["我{symptom}...不对，是{symptom2}",
                  "我是来看{symptom}的，呃，主要是{symptom2}",
                  "{symptom}，嗯，其实{symptom2}更明显"],
     "slots": {"symptom": S, "symptom2": S}},
]


def _slug(text):
    """Stable short hash used for utterance ids."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def normalize_for_dedup(text):
    """Aggressive normalization used only for duplicate detection."""
    out = unicodedata.normalize("NFKC", str(text)).lower()
    out = re.sub(r"[\s，。、！？；：…·\.\,\!\?\;\:\-\—\~]", "", out)
    return out


def char_ngrams(text, n=3):
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


class MinHashLSH:
    """Minimal MinHash + banding index for near-duplicate detection.

    Pure standard library. ``num_perm`` hashes per document, split into bands;
    two documents become candidates when any band matches, and the candidate
    pair is then verified with exact Jaccard over the 3-gram sets.
    """

    def __init__(self, threshold=0.9, num_perm=64, bands=16, seed=42):
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        rng = random.Random(seed)
        big = (1 << 61) - 1
        self.coeffs = [(rng.randrange(1, big), rng.randrange(0, big))
                       for _ in range(num_perm)]
        self.buckets = [{} for _ in range(bands)]
        self.signatures = {}
        self.grams = {}

    def _signature(self, grams):
        big = (1 << 61) - 1
        base = [int(hashlib.md5(g.encode("utf-8")).hexdigest()[:15], 16) for g in grams]
        if not base:
            return tuple([0] * self.num_perm)
        return tuple(min((a * h + b) % big for h in base) for a, b in self.coeffs)

    def query_and_add(self, key, text):
        """Return the key of a near-duplicate already in the index, or None."""
        grams = char_ngrams(text)
        signature = self._signature(grams)

        candidates = set()
        band_keys = []
        for band in range(self.bands):
            chunk = signature[band * self.rows:(band + 1) * self.rows]
            band_keys.append(chunk)
            candidates.update(self.buckets[band].get(chunk, ()))

        for other in candidates:
            other_grams = self.grams[other]
            if not grams and not other_grams:
                return other
            union = len(grams | other_grams)
            if union and len(grams & other_grams) / union >= self.threshold:
                return other

        self.signatures[key] = signature
        self.grams[key] = grams
        for band, chunk in enumerate(band_keys):
            self.buckets[band].setdefault(chunk, []).append(key)
        return None


def _pattern_slots(pattern):
    """Slot names used by one pattern, in a stable order."""
    seen = []
    for name in re.findall(r"\{(\w+)\}", pattern):
        if name not in seen:
            seen.append(name)
    return seen


def _pattern_capacity(family, pattern):
    """Exact number of distinct fillings for one pattern.

    Capacity is computed per pattern, not per family: a family whose patterns
    use different subsets of the slots would otherwise be credited with
    combinations that no pattern can actually produce, and its quota could never
    be met.
    """
    slots = family["slots"]
    if slots.get("__vital_pair__"):
        return sum(len(values) for _, values in V.VITALS)
    if slots.get("__abbrev_pair__"):
        return len(V.ABBREV_EXPANSION)

    total = 1
    for name in _pattern_slots(pattern):
        pool = slots.get(name)
        if not pool:
            return 0
        total *= len(pool)
    return total


def _family_capacity(family):
    """Upper bound on distinct utterances a family can produce."""
    return sum(_pattern_capacity(family, p) for p in family["patterns"])


def _decode_index(index, pools):
    """Map a flat combination index onto one value per slot (mixed radix)."""
    values = []
    for pool in reversed(pools):
        index, position = divmod(index, len(pool))
        values.append(pool[position])
    values.reverse()
    return values


def _draw_pattern(family, pattern, count, rng):
    """Return up to ``count`` distinct fillings of one pattern.

    Distinct combinations are drawn by sampling distinct integers from the
    combination space and decoding them, rather than by rejection sampling.
    This makes small families fillable (no coupon-collector blowup) and keeps
    generation deterministic given the seed.
    """
    slots = family["slots"]

    if slots.get("__vital_pair__"):
        pairs = [(name, value) for name, values in V.VITALS for value in values]
        rng.shuffle(pairs)
        return [pattern.format(vital=n, value=v) for n, v in pairs[:count]]

    if slots.get("__abbrev_pair__"):
        items = sorted(V.ABBREV_EXPANSION.items())
        rng.shuffle(items)
        return [pattern.format(abbr=a, expansion=e) for a, e in items[:count]]

    names = _pattern_slots(pattern)
    pools = [slots[name] for name in names]
    if not pools:
        return [pattern] if count > 0 else []

    space = 1
    for pool in pools:
        space *= len(pool)

    # Oversample a little: paired slots that must differ will reject some draws.
    want = min(space, int(count * 1.35) + 8)
    indices = rng.sample(range(space), want)

    out = []
    for index in indices:
        values = dict(zip(names, _decode_index(index, pools)))
        skip = False
        for name in names:
            base = name.rstrip("2")
            if name.endswith("2") and base in values and values[name] == values[base]:
                skip = True
                break
        if skip:
            continue
        out.append(pattern.format(**values))
        if len(out) >= count:
            break
    return out


def allocate_quota(total, families):
    """Split the target count across categories, then across families.

    Category proportions come from the Phase 3 specification. Within a category
    the quota is distributed in proportion to each family's real capacity, then
    clipped to that capacity, and any shortfall is redistributed to families
    that still have headroom.
    """
    by_category = {}
    for family in families:
        by_category.setdefault(family["category"], []).append(family)

    quotas = {}
    for category, proportion in V_DOMAIN.items():
        members = by_category.get(category, [])
        if not members:
            continue
        target = int(round(total * proportion))
        capacities = {f["id"]: _family_capacity(f) for f in members}
        pooled = sum(capacities.values())

        assigned = {}
        for family in members:
            fid = family["id"]
            # Weight by capacity but keep an even-share floor so that a family
            # with a big combination space cannot crowd out the others.
            even = target / len(members)
            weighted = target * (capacities[fid] / pooled) if pooled else 0
            want = int(round(0.5 * even + 0.5 * weighted))
            assigned[fid] = min(want, capacities[fid])

        shortfall = target - sum(assigned.values())
        while shortfall > 0:
            headroom = [f for f in members if assigned[f["id"]] < capacities[f["id"]]]
            if not headroom:
                break
            per = max(1, shortfall // len(headroom))
            for family in headroom:
                if shortfall <= 0:
                    break
                room = capacities[family["id"]] - assigned[family["id"]]
                add = min(per, room, shortfall)
                assigned[family["id"]] += add
                shortfall -= add
        while shortfall < 0:  # rounding can overshoot
            for family in members:
                if shortfall >= 0:
                    break
                if assigned[family["id"]] > 0:
                    assigned[family["id"]] -= 1
                    shortfall += 1
        quotas.update(assigned)
    return quotas


def generate(total=18000, seed=42, near_dup_threshold=0.9, families=None,
             verbose=True):
    """Generate the script corpus.

    Returns ``(records, report)``. Each record carries ``script_id``, ``text``,
    ``domain_category`` and ``template_family``, so Phase 4 can split by family
    and Phase 9 can report CER per category.
    """
    families = families or TEMPLATE_FAMILIES
    rng = random.Random(seed)
    quotas = allocate_quota(total, families)

    lsh = MinHashLSH(threshold=near_dup_threshold, seed=seed)
    seen_exact = {}
    records = []
    rejected = Counter()
    rejection_log = []
    shortfalls = {}

    for family in families:
        quota = quotas.get(family["id"], 0)
        if quota <= 0:
            continue

        # Distribute the family quota over its patterns by pattern capacity.
        patterns = family["patterns"]
        capacities = [_pattern_capacity(family, p) for p in patterns]
        pooled = sum(capacities) or 1
        wanted = [int(quota * c / pooled) for c in capacities]
        for i in range(quota - sum(wanted)):
            wanted[i % len(wanted)] += 1

        produced = 0
        for pattern, want in zip(patterns, wanted):
            if want <= 0:
                continue
            for text in _draw_pattern(family, pattern, want, rng):
                key = normalize_for_dedup(text)
                if key in seen_exact:
                    rejected["exact_duplicate"] += 1
                    continue

                duplicate_of = lsh.query_and_add(text, key)
                if duplicate_of is not None:
                    rejected["near_duplicate"] += 1
                    if len(rejection_log) < 200:
                        rejection_log.append({
                            "text": text, "reason": "near_duplicate",
                            "duplicate_of": duplicate_of,
                            "template_family": family["id"]})
                    continue

                script_id = "SCR-%s-%s" % (family["id"], _slug(text))
                seen_exact[key] = script_id
                records.append(OrderedDict([
                    ("script_id", script_id),
                    ("text", text),
                    ("domain_category", family["category"]),
                    ("template_family", family["id"]),
                    ("n_chars", len(key)),
                ]))
                produced += 1

        if produced < quota:
            rejected["quota_unfilled"] += quota - produced
            shortfalls[family["id"]] = {
                "produced": produced, "quota": quota,
                "capacity": _family_capacity(family)}
            if verbose:
                print("  %-24s produced %d/%d (capacity %d)"
                      % (family["id"], produced, quota, _family_capacity(family)))

    rng.shuffle(records)

    category_counts = Counter(r["domain_category"] for r in records)
    report = {
        "requested_total": total,
        "generated_total": len(records),
        "seed": seed,
        "near_dup_threshold": near_dup_threshold,
        "n_template_families": len({r["template_family"] for r in records}),
        "rejections": dict(rejected),
        "rejection_examples": rejection_log[:50],
        "family_shortfalls": shortfalls,
        "by_category": dict(category_counts),
        "by_category_pct": {k: round(100.0 * v / max(1, len(records)), 2)
                            for k, v in category_counts.items()},
        "target_category_pct": {k: round(100.0 * v, 2) for k, v in V_DOMAIN.items()},
        "length_stats": _length_stats(records),
        "family_counts": dict(Counter(r["template_family"] for r in records)),
        "family_capacities": {f["id"]: _family_capacity(f) for f in families},
    }
    return records, report


def _length_stats(records):
    if not records:
        return {}
    lengths = sorted(r["n_chars"] for r in records)
    n = len(lengths)
    return {
        "min": lengths[0], "max": lengths[-1],
        "mean": round(sum(lengths) / n, 2),
        "p10": lengths[int(0.10 * n)], "median": lengths[n // 2],
        "p90": lengths[int(0.90 * n)],
    }
