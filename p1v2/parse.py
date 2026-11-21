"""
p1v2/parse.py -- read the model's FINAL answer, not the first class that appears anywhere in its text (the rule of shared/inference_engine.py, which returns ESI 1 / option A whenever the text mentions them).

answer(raw, task) -> (answer or None, confidence) with confidence one of
  exact        the whole response is one answer token, optionally in brackets or with a period: "3", "(C)", "2)"
  explicit     the last explicit statement 'answer / level / final / choice / option ... X' (several words may stand between the keyword and X)
  last_token   the last standalone class token in the text (weak: reasoning text can mention other classes)
  refusal      no class token and the reply opens like a refusal ("I'm sorry, but I cannot ..."); never guessed
  unparseable  no class token at all (empty or text-only reply); never guessed
Callers should treat last_token as lower quality and report the share of each confidence.
"""
from __future__ import annotations

import re

ESI_CLASSES = "12345"
USMLE_CLASSES = "ABCD"
_EXACT = re.compile(r"^\s*[\(\[]?\s*([A-D1-5])\s*[\)\]]?\s*[.:]?\s*$")
_EXPLICIT = re.compile(r"(?i)\b(?:final(?:\s+answer)?|answer|level|choice|option|triage\s+level)\b[^A-Za-z0-9]{0,14}(?:is|:|=|of)?[^A-Za-z0-9]{0,6}([A-D1-5])(?![A-Za-z0-9])")
_IS = re.compile(r"(?i)\b(?:level|answer|choice|option)\b[^.\n]{0,40}?\b(?:is|are|would be|=|:)\s*[\(\[]?([A-D1-5])(?![A-Za-z0-9])")
_REFUSAL = re.compile(r"(?i)^(i'?m sorry|i am sorry|sorry|i cannot|i can't|i can not|i am unable|i'm unable|unable to|as an ai|i won't)")
_TOKEN = re.compile(r"(?<![A-Za-z0-9])([A-D1-5])(?![A-Za-z0-9])")


def classes_for(task: str) -> str:
    return ESI_CLASSES if str(task).startswith("esi") else USMLE_CLASSES


def answer(raw: str | None, task: str) -> tuple[str | None, str]:
    classes = classes_for(task)
    raw = raw or ""
    m = _EXACT.match(raw)
    if m and m.group(1) in classes:
        return m.group(1), "exact"
    hits = [(x.end(), x.group(1)) for pat in (_EXPLICIT, _IS) for x in pat.finditer(raw) if x.group(1) in classes]
    if hits:
        return max(hits)[1], "explicit"
    tokens = [x for x in _TOKEN.findall(raw) if x in classes]
    if tokens:
        return tokens[-1], "last_token"
    if _REFUSAL.match(raw.strip()):
        return None, "refusal"
    return None, "unparseable"
