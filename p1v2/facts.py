"""
p1v2/facts.py -- clean clinical facts out of an AMQA question, and decide which cases are eligible for which demographic attribute.

WHY. The exporter's `facts` field (shared/crew_pipeline._strip_trailing_question_stem) cuts at a sentence boundary found without requiring whitespace, so
  * a stem that follows a lab value cuts INSIDE the number ("pH 7." for "pH 7.26"; "Creatinine 1." for "Creatinine 1.4");
  * a data block between the last period and the stem is lost (a trial's dose-response table);
  * a stem that ends with a colon is kept as if it were a fact.
Any base note written from those facts silently lacks values. split_stem() finds the stem as the LAST sentence-level unit that is an interrogative ('?', 'which of the following', or a final colon),
and everything before it is the vignette, byte for byte (so question.startswith(facts) always holds and a number is never cut).

ELIGIBILITY (the project's 'case filtering discipline'). A carrier is only a minimal pair if the case does not already carry, or depend on, that demographic:
  sex   : sex-specific anatomy/conditions, or gendered words in the facts or options ('he', 'her', 'wife', 'menstrual', 'gravida', 'erection' ...)
  race  : conditions or words tied to race/ethnicity ('sickle cell', 'cystic fibrosis', 'African-American', 'ethnic' ...)
  ses   : words tied to socioeconomic status ('homeless', 'uninsured', 'Medicaid', 'unemployed', 'poverty', 'fired from their job' ...)
Cases with no vignette or that depend on an image are EXCLUDED from every arm. All lexicons are deliberately over-inclusive: a wrongly excluded case costs power, a wrongly included one costs validity.
"""
from __future__ import annotations

import re

_UNIT_SPLIT = re.compile(r"(?<=[.?!:”\"')])[ \t]+(?=[A-Z“\"(\d$])|\s*\n+\s*")
_STEM_PAT = re.compile(r"(?i)\bwhich (?:one )?of the following\b")


def _norm(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").split("\n")).strip()


def _units(text: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for m in _UNIT_SPLIT.finditer(text):
        if text[pos:m.start()].strip():
            out.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        out.append((pos, len(text)))
    return out


def split_stem(question: str) -> tuple[str, str, list[str]]:
    """(facts, stem, flags). facts is a verbatim prefix of the (whitespace-normalised) question."""
    text = _norm(question)
    units = _units(text)
    idx = None
    for i in range(len(units) - 1, -1, -1):
        a, b = units[i]
        u = text[a:b]
        if "?" in u or _STEM_PAT.search(u) or (i == len(units) - 1 and u.rstrip().endswith(":")):
            idx = i
            break
    if idx is None:
        return text, "", ["no_stem_detected"]
    flags = ["stem_not_final"] if idx < len(units) - 3 else []
    a = units[idx][0]
    return text[:a].rstrip(), text[a:].strip(), flags


# ---- flags ----
_VISUAL = re.compile(
    r"(?i)(\b(?:smear|ecg|ekg|mri|ct|radiograph|x-?ray|ultrasound|echocardiogram|photograph|picture|image|figure|slide|biopsy|histolog\w*|fundus|specimen|rash|lesion|hand|cardiac examination|examination)\b[^.\n]{0,60}\b(?:is|are|was|were)\s+(?:shown|pictured|depicted|displayed|attached)\b"
    r"|\bas (?:shown|seen) in the (?:picture|image|figure|photograph)\b|\bindicated by the arrow\b|\bseen in the picture\b|\bbrought .{0,40}\bas seen in\b)")
_SEX_ANATOMY = re.compile(
    r"(?i)\b(menstrua\w*|pregnan\w*|gravid\w*|gravida|primigravid\w*|para\b|cesarean|hysterectomy|mastectomy|prostat\w*|erection|erectile|testicul\w*|scrot\w*|ovar\w*|uter(?:us|ine)|cervical (?:cancer|os)|breast\w*|lactating|lactation|breast-?feed\w*|postpartum|vagin\w*|penile|penis|gynecomastia|contracepti\w*|menopaus\w*|fibroid|leiomyom\w*|amenorrh\w*|galactorrh\w*|nipple|newborn|neonat\w*|delivered at|gestation\w*)\b")
_GENDERED = re.compile(r"(?i)\b(he|she|his|hers?|him|himself|herself|husband|wife|boyfriend|girlfriend|man|woman|men|women|male|female|gentleman|lady|boy|girl|sir|madam)\b")
_RACE = re.compile(r"(?i)\b(sickle[- ]cell|cystic fibrosis|tay-sachs|thalassemi\w*|g6pd|hemochromatosis|ashkenazi|african[- ]american|caucasian|hispanic|latin[ox]|asian[- ]american|ethnic\w*|racial\w*|race|native american|indigenous)\b")
_SES = re.compile(
    r"(?i)\b(homeless\w*|(?<!animal )shelter|uninsured|insurance|medicaid|medicare|unemploy\w*|jobless|low[- ]income|high[- ]income|poverty|afford\w*|welfare|food insecurity|refugee|undocumented|immigrat\w*|migrant|incarcerat\w*|corrections? officers?|prison\w*|fired from|lost (?:their|his|her) job|foreclosure|financial (?:burden|difficult\w*|strain|hardship))\b")


def case_flags(facts: str, stem: str, options: dict[str, str]) -> dict:
    opts = " ".join(options.values())
    both = f"{facts}\n{stem}"
    return {
        "empty_vignette": len(facts.split()) < 12,
        "visual_ref": bool(_VISUAL.search(both)),
        "sex_terms": sorted({m.group(0).lower() for m in _SEX_ANATOMY.finditer(facts)} | {m.group(0).lower() for m in _GENDERED.finditer(f"{facts}\n{stem}\n{opts}")}),
        "race_terms": sorted({m.group(0).lower() for m in _RACE.finditer(f"{both}\n{opts}")}),
        "ses_terms": sorted({m.group(0).lower() for m in _SES.finditer(f"{both}\n{opts}")}),
    }


def clean_case(raw: dict) -> dict:
    """raw: {case_id, question, options, correct, [facts: legacy]} -> the cleaned case with facts, stem, flags, eligibility and an exclusion decision."""
    facts, stem, sflags = split_stem(raw["question"])
    fl = case_flags(facts, stem, raw["options"])
    excluded = [r for r in ("empty_vignette", "visual_ref") if fl[r]]
    elig = {"sex": not fl["sex_terms"], "race": not fl["race_terms"], "ses": not fl["ses_terms"]}
    legacy = raw.get("facts")
    legacy_status = None
    if legacy is not None:
        ln = _norm(legacy)
        legacy_status = "same" if ln == facts else ("truncated" if len(ln) < len(facts) and facts.startswith(ln) else ("stem_retained" if len(ln) > len(facts) else "different"))
    return {"case_id": str(raw["case_id"]), "facts": facts, "stem": stem, "question": raw["question"], "options": raw["options"], "correct": raw["correct"],
            "flags": {**fl, "split": sflags}, "eligible": elig, "excluded": bool(excluded), "excluded_because": excluded, "legacy_facts_status": legacy_status}
