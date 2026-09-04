"""Phase 8/9 - Text normalization for Chinese hospital-domain ASR scoring.

Every normalization step is an explicit, named flag. The active configuration is
serialized next to every metric so that a reported CER can always be traced back
to the exact rules that produced it. Nothing is removed silently.

Default policy (``NormalizationConfig()``) and its justification:

  nfkc                  ON   Unicode NFKC. Collapses full-width ASCII and
                             compatibility forms so that "ＣＴ" and "CT" are one
                             token. Without this, TTS/ASR width differences show
                             up as pure formatting errors.
  strip_punctuation     ON   Chinese and ASCII punctuation is removed from BOTH
                             reference and hypothesis. Qwen3-ASR emits punctuation
                             whose placement is not part of the recognition task
                             studied here. This is the single most consequential
                             rule, so it is reported explicitly in every table.
  lowercase             ON   English/abbreviation case is folded. "CT" and "ct"
                             count as identical. Medical abbreviation *identity*
                             is what matters, not casing.
  remove_whitespace     ON   All whitespace is removed. Standard practice for
                             Chinese CER (AISHELL-style): word segmentation is
                             not part of the task and spacing around embedded
                             English is not consistently defined.
  traditional_to_simplified OFF  Off by default. Conversion is lossy and the
                             corpus is Simplified by construction; any Traditional
                             character found is *reported*, not silently mapped.
  normalize_numbers     OFF  Off by default. Mapping "三十七度五" to "37.5" is a
                             semantic rewrite that can hide genuine recognition
                             errors. Phase 9 reports a secondary number-normalized
                             CER as a separate, clearly labelled metric.
  keep_digit_separators ON   Protects "." ":" "/" when they sit BETWEEN digits,
                             so "37.5" and "120/80" survive punctuation
                             stripping. Without this rule "37.5" collapses to
                             "375" and "120/80" to "12080", which corrupts every
                             measurement in the numeric category and can make a
                             genuine error score as correct. Sentence
                             punctuation is still removed.
  strip_language_tag    ON   Removes the Qwen3-ASR training prefix
                             ``language Chinese<asr_text>`` when it leaks into a
                             hypothesis. This is a prompt-format artifact, not a
                             recognition error.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass

# Punctuation: ASCII plus the CJK ranges. Kept as an explicit literal set so the
# exact character inventory is auditable rather than hidden behind a category test.
_ASCII_PUNCT = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
_CJK_PUNCT = (
    "　、。《》「」『』【】"
    "〔〕‘’“”！（），－"
    "：；？～･—…·￥．｜"
)
PUNCTUATION = set(_ASCII_PUNCT) | set(_CJK_PUNCT)

# Qwen3-ASR prompt contract: "language <Lang><asr_text>actual transcript".
LANGUAGE_TAG_RE = re.compile(r"^\s*language\s+\w+\s*<asr_text>\s*", re.IGNORECASE)
ASR_TEXT_RE = re.compile(r"^\s*<asr_text>\s*", re.IGNORECASE)

# Ranges used only for *reporting* script composition, never for rewriting.
CJK_RANGE = (0x4E00, 0x9FFF)

_CN_DIGITS = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000, "亿": 100000000}


@dataclass(frozen=True)
class NormalizationConfig:
    """Explicit switch board. Serialized into every metrics.json."""

    nfkc: bool = True
    strip_punctuation: bool = True
    lowercase: bool = True
    remove_whitespace: bool = True
    traditional_to_simplified: bool = False
    normalize_numbers: bool = False
    keep_digit_separators: bool = True
    strip_language_tag: bool = True

    def to_dict(self):
        return asdict(self)

    def fingerprint(self):
        """Short stable id so two runs can be checked for rule equality."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        import hashlib

        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


DEFAULT_CONFIG = NormalizationConfig()


def _chinese_number_to_arabic(text):
    """Convert standalone Chinese numerals to Arabic digits.

    Deliberately conservative: only contiguous runs made purely of Chinese
    numeral/unit characters are converted, and a run that fails to parse is left
    exactly as it was. Used only when ``normalize_numbers`` is enabled.
    """

    def parse_run(run):
        # Two different reading systems share the same characters:
        #   positional  "三十七"   -> 37    (contains a unit: 十/百/千/万/亿)
        #   digit-wise  "二零一五" -> 2015  (no unit; years, room numbers, IDs)
        # Applying the positional algorithm to a digit-wise run is silently
        # wrong - "二零一五" would come out as 5 - so the two are separated by
        # the presence of a unit character.
        if len(run) > 1 and not any(char in _CN_UNITS for char in run):
            return int("".join(str(_CN_DIGITS[char]) for char in run))

        total, section, number = 0, 0, 0
        for char in run:
            if char in _CN_DIGITS:
                number = _CN_DIGITS[char]
            elif char in _CN_UNITS:
                unit = _CN_UNITS[char]
                if unit >= 10000:
                    section = (section + number) * unit
                    total += section
                    section, number = 0, 0
                else:
                    if number == 0:
                        number = 1
                    section += number * unit
                    number = 0
            else:
                return None
        return total + section + number

    pattern = re.compile("[" + "".join(_CN_DIGITS) + "".join(_CN_UNITS) + "]+")

    def replace(match):
        value = parse_run(match.group(0))
        return match.group(0) if value is None else str(value)

    # "点" between numerals marks a decimal point in spoken Chinese.
    text = re.sub(
        r"([零〇一二两三四五六七八九十百千万亿]+)点([零〇一二两三四五六七八九]+)",
        lambda m: "%s.%s" % (
            parse_run(m.group(1)) if parse_run(m.group(1)) is not None else m.group(1),
            "".join(str(_CN_DIGITS[c]) for c in m.group(2)),
        ),
        text,
    )
    return pattern.sub(replace, text)


# Separators that carry meaning between digits ("37.5", "120/80", "9:30").
DIGIT_SEPARATORS = {".": "", ":": "", "/": ""}
_SEPARATOR_RE = re.compile(
    r"(?<=\d)([%s])(?=\d)" % re.escape("".join(DIGIT_SEPARATORS)))


def _protect_digit_separators(text):
    """Swap digit-internal separators for private-use placeholders."""
    return _SEPARATOR_RE.sub(lambda m: DIGIT_SEPARATORS[m.group(1)], text)


def _restore_digit_separators(text):
    for char, placeholder in DIGIT_SEPARATORS.items():
        text = text.replace(placeholder, char)
    return text


def normalize(text, config=DEFAULT_CONFIG):
    """Apply the configured rules and return the normalized string."""
    if text is None:
        return ""
    out = str(text)

    if config.strip_language_tag:
        out = LANGUAGE_TAG_RE.sub("", out)
        out = ASR_TEXT_RE.sub("", out)

    if config.nfkc:
        out = unicodedata.normalize("NFKC", out)

    if config.traditional_to_simplified:
        try:
            from opencc import OpenCC

            out = OpenCC("t2s").convert(out)
        except ImportError as exc:
            raise RuntimeError(
                "traditional_to_simplified=True requires the 'opencc' package; "
                "refusing to silently skip a normalization rule") from exc

    if config.normalize_numbers:
        out = _chinese_number_to_arabic(out)

    if config.lowercase:
        out = out.lower()

    if config.strip_punctuation:
        if config.keep_digit_separators:
            out = _protect_digit_separators(out)
        out = "".join(ch for ch in out if ch not in PUNCTUATION)
        if config.keep_digit_separators:
            out = _restore_digit_separators(out)

    if config.remove_whitespace:
        out = "".join(out.split())
    else:
        out = " ".join(out.split())

    return out


def script_profile(text):
    """Character composition of a string. Used for QC reporting, never to rewrite."""
    profile = {"cjk": 0, "latin": 0, "digit": 0, "punct": 0, "space": 0, "other": 0,
               "traditional_suspect": 0}
    for char in str(text):
        code = ord(char)
        if CJK_RANGE[0] <= code <= CJK_RANGE[1]:
            profile["cjk"] += 1
        elif char.isascii() and char.isalpha():
            profile["latin"] += 1
        elif char.isdigit():
            profile["digit"] += 1
        elif char in PUNCTUATION:
            profile["punct"] += 1
        elif char.isspace():
            profile["space"] += 1
        else:
            profile["other"] += 1
    return profile


def describe(config=DEFAULT_CONFIG):
    """Human-readable rule list to paste into a paper's methods section."""
    rules = [
        ("NFKC unicode normalization", config.nfkc),
        ("strip punctuation (ASCII + CJK)", config.strip_punctuation),
        ("lowercase latin characters", config.lowercase),
        ("remove all whitespace", config.remove_whitespace),
        ("traditional -> simplified", config.traditional_to_simplified),
        ("chinese numerals -> arabic", config.normalize_numbers),
        ("keep . : / between digits", config.keep_digit_separators),
        ("strip 'language X<asr_text>' prefix", config.strip_language_tag),
    ]
    lines = ["normalization fingerprint: %s" % config.fingerprint()]
    lines += ["  [%s] %s" % ("x" if on else " ", name) for name, on in rules]
    return "\n".join(lines)
