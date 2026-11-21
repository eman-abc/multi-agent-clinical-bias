# Agent B parsing audit

Stored = the project's rule (first class in list order that appears anywhere in the raw text). Strict = exact token, else the last explicit 'answer/level/final X', else the last standalone class token.

Records WITHOUT raw text (cannot be audited): Gemini-3.8-Flash|esi: 7600, Gemini-3.8-Flash|usmle: 7599

## DeepSeek-V4-Flash|esi  (n = 7600)
- raw response length: exactly one character 99.2%, 1 to 3 characters 99.2%, over 20 characters 0.8%, empty 0.0%, median 1
- strict reading: {'exact': 7542, 'unparseable': 58}; unparseable 58
- stored top_class differs from the strict reading in 0 records (0.0%); 0 of them have the first-class-rule signature

| condition | carrier | n | stored rate | strict rate | stored minus strict (points) |
|---|---|---|---|---|---|
| REC_DOWN | EXP | 1200 | 32.2% | 32.2% | +0.0 |
| REC_DOWN | IMP | 1200 | 33.2% | 33.2% | +0.0 |
| REC_DOWN | N | 200 | 35.0% | 35.0% | +0.0 |
| REC_DOWN | PRX | 1200 | 29.5% | 29.5% | +0.0 |
| REC_UP | EXP | 1200 | 19.0% | 19.0% | +0.0 |
| REC_UP | IMP | 1200 | 21.6% | 21.6% | +0.0 |
| REC_UP | N | 200 | 22.5% | 22.5% | +0.0 |
| REC_UP | PRX | 1200 | 18.9% | 18.9% | +0.0 |

## DeepSeek-V4-Flash|usmle  (n = 7598)
- raw response length: exactly one character 98.6%, 1 to 3 characters 98.6%, over 20 characters 1.0%, empty 0.0%, median 1
- strict reading: {'exact': 7490, 'last-token': 63, 'explicit': 5, 'unparseable': 40}; unparseable 40
- stored top_class differs from the strict reading in 10 records (0.1%); 10 of them have the first-class-rule signature

| condition | carrier | n | stored rate | strict rate | stored minus strict (points) |
|---|---|---|---|---|---|
| REC_CORRECT | EXP | 1200 | 94.1% | 93.6% | +0.5 |
| REC_CORRECT | IMP | 1199 | 93.7% | 93.2% | +0.5 |
| REC_CORRECT | N | 200 | 93.0% | 92.5% | +0.5 |
| REC_CORRECT | PRX | 1199 | 92.6% | 92.2% | +0.4 |
| REC_WRONG | EXP | 1200 | 18.3% | 18.2% | +0.1 |
| REC_WRONG | IMP | 1200 | 16.6% | 16.2% | +0.3 |
| REC_WRONG | N | 200 | 18.5% | 18.5% | +0.0 |
| REC_WRONG | PRX | 1200 | 14.1% | 13.9% | +0.2 |

REC_WRONG target letters: {'B': 855, 'A': 2945}

## GPT-5.6-Luna|esi  (n = 12908)
- raw response length: exactly one character 98.1%, 1 to 3 characters 98.1%, over 20 characters 1.8%, empty 0.1%, median 1
- strict reading: {'exact': 12661, 'unparseable': 247}; unparseable 247
- stored top_class differs from the strict reading in 0 records (0.0%); 0 of them have the first-class-rule signature

| condition | carrier | n | stored rate | strict rate | stored minus strict (points) |
|---|---|---|---|---|---|
| REC_DOWN | EXP | 2038 | 49.2% | 49.2% | +0.0 |
| REC_DOWN | IMP | 2037 | 50.7% | 50.7% | +0.0 |
| REC_DOWN | N | 340 | 53.5% | 53.5% | +0.0 |
| REC_DOWN | PRX | 2039 | 52.5% | 52.5% | +0.0 |
| REC_UP | EXP | 2038 | 19.9% | 19.9% | +0.0 |
| REC_UP | IMP | 2037 | 19.4% | 19.4% | +0.0 |
| REC_UP | N | 341 | 20.8% | 20.8% | +0.0 |
| REC_UP | PRX | 2038 | 18.1% | 18.1% | +0.0 |

## GPT-5.6-Luna|usmle  (n = 12910)
- raw response length: exactly one character 97.6%, 1 to 3 characters 97.6%, over 20 characters 2.2%, empty 0.1%, median 1
- strict reading: {'exact': 12606, 'unparseable': 298, 'last-token': 6}; unparseable 298
- stored top_class differs from the strict reading in 3 records (0.0%); 3 of them have the first-class-rule signature

| condition | carrier | n | stored rate | strict rate | stored minus strict (points) |
|---|---|---|---|---|---|
| REC_CORRECT | EXP | 2040 | 97.2% | 97.2% | +0.0 |
| REC_CORRECT | IMP | 2038 | 97.6% | 97.5% | +0.0 |
| REC_CORRECT | N | 340 | 97.9% | 97.9% | +0.0 |
| REC_CORRECT | PRX | 2039 | 97.8% | 97.8% | +0.0 |
| REC_WRONG | EXP | 2040 | 35.4% | 35.4% | +0.0 |
| REC_WRONG | IMP | 2035 | 32.4% | 32.3% | +0.0 |
| REC_WRONG | N | 340 | 38.8% | 38.8% | +0.0 |
| REC_WRONG | PRX | 2038 | 30.7% | 30.6% | +0.1 |

REC_WRONG target letters: {'B': 1727, 'A': 4726}

