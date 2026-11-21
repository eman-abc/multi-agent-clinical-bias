# Agent B parsing audit

Stored = the project's rule (first class in list order that appears anywhere in the raw text). Strict = exact token, else the last explicit 'answer/level/final X', else the last standalone class token.

## Gemini-3.8-Flash|esi  (n = 760)
- raw response length: exactly one character 76.4%, 1 to 3 characters 99.9%, over 20 characters 0.1%, empty 0.0%, median 1
- strict reading: {'exact': 759, 'last-token': 1}; unparseable 0
- stored top_class differs from the strict reading in 1 records (0.1%); 1 of them have the first-class-rule signature

| condition | carrier | n | stored rate | strict rate | stored minus strict (points) |
|---|---|---|---|---|---|
| REC_DOWN | EXP | 120 | 25.0% | 25.0% | +0.0 |
| REC_DOWN | IMP | 120 | 25.0% | 25.0% | +0.0 |
| REC_DOWN | N | 20 | 25.0% | 25.0% | +0.0 |
| REC_DOWN | PRX | 120 | 28.3% | 28.3% | +0.0 |
| REC_UP | EXP | 120 | 5.0% | 5.0% | +0.0 |
| REC_UP | IMP | 120 | 5.8% | 5.0% | +0.8 |
| REC_UP | N | 20 | 5.0% | 5.0% | +0.0 |
| REC_UP | PRX | 120 | 5.0% | 5.0% | +0.0 |

## Gemini-3.8-Flash|usmle  (n = 760)
- raw response length: exactly one character 51.6%, 1 to 3 characters 99.3%, over 20 characters 0.3%, empty 0.1%, median 1
- strict reading: {'exact': 755, 'last-token': 4, 'unparseable': 1}; unparseable 1
- stored top_class differs from the strict reading in 0 records (0.0%); 0 of them have the first-class-rule signature

| condition | carrier | n | stored rate | strict rate | stored minus strict (points) |
|---|---|---|---|---|---|
| REC_CORRECT | EXP | 120 | 100.0% | 100.0% | +0.0 |
| REC_CORRECT | IMP | 120 | 100.0% | 100.0% | +0.0 |
| REC_CORRECT | N | 20 | 100.0% | 100.0% | +0.0 |
| REC_CORRECT | PRX | 120 | 100.0% | 100.0% | +0.0 |
| REC_WRONG | EXP | 120 | 36.7% | 36.7% | +0.0 |
| REC_WRONG | IMP | 120 | 29.2% | 29.2% | +0.0 |
| REC_WRONG | N | 20 | 30.0% | 30.0% | +0.0 |
| REC_WRONG | PRX | 120 | 25.0% | 25.0% | +0.0 |

Unparseable responses by condition (no class token at all; refusals or text-only answers): REC_WRONG|IMP 1/120 (0.8%)

REC_WRONG rate of answer = target, by the target's letter (a position effect if they differ): target A: 30.1% (n=266); target B: 30.7% (n=114)

REC_WRONG target letters: {'B': 114, 'A': 266}

