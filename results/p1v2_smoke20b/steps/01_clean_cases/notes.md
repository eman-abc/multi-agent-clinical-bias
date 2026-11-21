# Step 01_clean_cases

**Purpose.** split each question into vignette and stem without cutting numbers; flag image-dependent and no-vignette cases; decide per-attribute eligibility

## Input
- {'kind': 'file', 'path': 'p1v2_data\\cases20.jsonl', 'sha256': '528ac59181ac5948f37f5cb8b5a48ef9ad20e13ecdcb38c3edc04444caa3ffb1', 'n_lines': 20}

## Output
- {'path': 'output.jsonl', 'sha256': '9d293f02bcea3a751bea2d16dde51032358b859ec4504292aa5d50ffe2418d5a', 'n_lines': 20}

## Notes and decisions
- 20 cases; excluded 0; ineligible sex/race/ses = 1/0/1

**Status.** ok (0.03 s, started 2026-10-09T21:09:00+00:00)
