### Table 1b: ESI Triage (Experiment 1)

| Model             | Attr   | Format   |   CFR |    AD |   Cohen's h | McNemar p   | Excess Mean   | 95% CI                 | Verdict    |
|:------------------|:-------|:---------|------:|------:|------------:|:------------|:--------------|:-----------------------|:-----------|
| DeepSeek-V4-Flash | SES    | EXP      | 0.576 | 0.045 |       0.183 | 0.031*      | -0.035        | [-0.197, 0.106] (Null) | Confounded |
| DeepSeek-V4-Flash | SES    | IMP      | 0.535 | 0.091 |       0.031 | 0.815       | -             | -                      | Null       |
| DeepSeek-V4-Flash | SES    | PRX      | 0.605 | 0.01  |      -0.091 | 0.298       | -0.080        | [-0.230, 0.070] (Null) | Null       |
| DeepSeek-V4-Flash | RACE   | EXP      | 0.475 | 0.076 |      -0.162 | 0.085       | 0.061         | [-0.081, 0.202] (Null) | Null       |
| DeepSeek-V4-Flash | RACE   | IMP      | 0.584 | 0.041 |      -0.102 | 0.237       | -             | -                      | Null       |
| DeepSeek-V4-Flash | RACE   | PRX      | 0.55  | 0.055 |       0.05  | 0.635       | 0.010         | [-0.140, 0.150] (Null) | Null       |
| DeepSeek-V4-Flash | SEX    | EXP      | 0.596 | 0.061 |       0     | 0.897       | 0.141         | [0.000, 0.293] (Null)  | Null       |
| DeepSeek-V4-Flash | SEX    | IMP      | 0.51  | 0.111 |      -0.071 | 0.483       | -             | -                      | Null       |
| DeepSeek-V4-Flash | SEX    | PRX      | 0.53  | 0.015 |       0.08  | 0.403       | 0.055         | [-0.095, 0.211] (Null) | Null       |
| Gemini-3.8-Flash  | SES    | EXP      | 0.89  | 0     |       0.017 | 1.000       | -0.025        | [-0.105, 0.040] (Null) | Null       |
| Gemini-3.8-Flash  | SES    | IMP      | 0.9   | 0.025 |       0.165 | 0.064       | -             | -                      | Null       |
| Gemini-3.8-Flash  | SES    | PRX      | 0.895 | 0.005 |      -0.019 | 1.000       | 0.040         | [-0.025, 0.100] (Null) | Null       |
| Gemini-3.8-Flash  | RACE   | EXP      | 0.884 | 0.015 |      -0.079 | 0.405       | -0.085        | [-0.176, -0.005]       | Null       |
| Gemini-3.8-Flash  | RACE   | IMP      | 0.94  | 0.005 |       0     | 1.000       | -             | -                      | Null       |
| Gemini-3.8-Flash  | RACE   | PRX      | 0.915 | 0.02  |       0.034 | 0.804       | 0.005         | [-0.050, 0.060] (Null) | Null       |
| Gemini-3.8-Flash  | SEX    | EXP      | 0.945 | 0.005 |      -0.065 | 0.344       | -0.055        | [-0.150, 0.040] (Null) | Null       |
| Gemini-3.8-Flash  | SEX    | IMP      | 0.88  | 0.015 |      -0.015 | 1.000       | -             | -                      | Null       |
| Gemini-3.8-Flash  | SEX    | PRX      | 0.9   | 0.015 |       0.086 | 0.359       | -0.015        | [-0.080, 0.050] (Null) | Null       |