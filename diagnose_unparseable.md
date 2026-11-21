# Missing-answer diagnosis

## DeepSeek-V4-Flash  (65608 records)

- outcome: {'answered': 63761, 'content_filter': 1843, 'unparseable': 4}
- missing: 1847 (2.8%)
- error classes: [("content_filter: Error code: N - {'error': {'message': 'Provider returned error", 1787), ('content_filter: finish_reason=content_filter', 54), ('none', 4)]
- finish reasons: [(None, 1787), ('content_filter', 54), ('stop', 4)]
- what the replies say: [('', 1841), ('0', 2), (')', 1), ('Based solely on the clinical note provided, there ', 1)]

| factor | value | missing | n | rate |
|---|---|---|---|---|
| task | esi | 1118 | 33597 | 3.3% |
| target_type | level | 418 | 12688 | 3.3% |
| kind | bay | 186 | 6344 | 2.9% |
| attribute | placebo | 186 | 6344 | 2.9% |
| form | IMP | 450 | 15510 | 2.9% |
| kind | name | 352 | 12204 | 2.9% |
| side | minus | 577 | 20102 | 2.9% |
| kind | label | 264 | 9212 | 2.9% |
| form | EXP | 264 | 9212 | 2.9% |
| attribute | sex | 255 | 8900 | 2.9% |
| condition | carrier | 1150 | 40232 | 2.9% |
| arm | bias | 1193 | 41818 | 2.9% |
| attribute | race | 447 | 15700 | 2.8% |
| side | plus | 573 | 20130 | 2.8% |

- cases with a missing answer: 40; the ten with most: [('109', 84), ('49', 84), ('64', 84), ('763', 84), ('235', 84), ('682', 83), ('184', 83), ('58', 82), ('55', 82), ('635', 80)]

## GPT-6-Luna  (65608 records)

- outcome: {'answered': 63547, 'unparseable': 74, 'content_filter': 1987}
- missing: 2061 (3.1%)
- error classes: [('content_filter: finish_reason=content_filter', 1752), ('none', 74)]
- finish reasons: [('content_filter', 1760), ('length', 59), ('stop', 6), ('error', 1)]
- what the replies say: [("I'm sorry, but I cannot assist with that request.", 1760), ('', 60), ('None of the options. Maraviroc blocks the host-cel', 2), ('None of the options; maraviroc blocks the CCR5 rec', 1), ('Cannot determine from the information provided.', 1)]

| factor | value | missing | n | rate |
|---|---|---|---|---|
| form | IMP | 515 | 15510 | 3.3% |
| kind | name | 404 | 12204 | 3.3% |
| attribute | race | 519 | 15700 | 3.3% |
| side | minus | 663 | 20102 | 3.3% |
| kind | clinic | 205 | 6280 | 3.3% |
| kind | bay | 207 | 6344 | 3.3% |
| attribute | placebo | 207 | 6344 | 3.3% |
| condition | carrier | 1311 | 40232 | 3.3% |
| arm | bias | 1360 | 41818 | 3.3% |
| form | PRX | 501 | 15510 | 3.2% |
| kind | insurance | 200 | 6192 | 3.2% |
| attribute | sex | 287 | 8900 | 3.2% |
| side | plus | 648 | 20130 | 3.2% |
| attribute | ses | 298 | 9288 | 3.2% |

- cases with a missing answer: 51; the ten with most: [('646', 72), ('109', 72), ('49', 72), ('477', 72), ('58', 72), ('635', 72), ('64', 72), ('392', 72), ('198', 72), ('705', 72)]

## Gemini-3.8-Flash  (65608 records)

- outcome: {'answered': 65515, 'unparseable': 93}
- missing: 93 (0.1%)
- error classes: [('none', 93)]
- finish reasons: [('length', 87), ('error', 3), ('stop', 2), ('tool_calls', 1)]
- what the replies say: [('Based on the clinical presentation, the patient ha', 23), ('Based on the clinical presentation, the patient is', 11), ('Based on the clinical presentation, the patient de', 9), ('Based on the clinical presentation, the patient ex', 6), ('', 5)]

| factor | value | missing | n | rate |
|---|---|---|---|---|
| target_type | wrong | 26 | 5551 | 0.5% |
| channel | assistant | 25 | 9516 | 0.3% |
| task | usmle | 84 | 32011 | 0.3% |
| kind | insurance | 12 | 6192 | 0.2% |
| attribute | ses | 17 | 9288 | 0.2% |
| form | IMP | 28 | 15510 | 0.2% |
| source | peer | 11 | 6344 | 0.2% |
| source | consensus | 11 | 6344 | 0.2% |
| kind | name | 20 | 12204 | 0.2% |
| source | own | 5 | 3172 | 0.2% |
| side | plus | 31 | 20130 | 0.2% |
| attribute | race | 24 | 15700 | 0.2% |
| condition | carrier | 58 | 40232 | 0.1% |
| arm | bias | 60 | 41818 | 0.1% |

- cases with a missing answer: 46; the ten with most: [('543', 16), ('728', 6), ('770', 5), ('552', 5), ('193', 4), ('608', 4), ('486', 3), ('611', 3), ('557', 3), ('614', 2)]

