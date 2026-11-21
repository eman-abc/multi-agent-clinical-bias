# Step 01_clean_cases

**Purpose.** split each question into vignette and stem without cutting numbers; flag image-dependent and no-vignette cases; decide per-attribute eligibility

## Input
- {'kind': 'file', 'path': 'p1v2_data\\cases.jsonl', 'sha256': 'ec291a198a3efca428a01ec64dd134cef79dee6713c28af995ff9cfcb35320c6', 'n_lines': 801}

## Output
- {'path': 'output.jsonl', 'sha256': '78f9114bd2e3cc4e3d865d6b880c2670ca620725610d92b8558f8a1373f23857', 'n_lines': 801}

## Notes and decisions
- 801 cases; excluded 8; ineligible sex/race/ses = 49/8/20

**Status.** ok (0.42 s, started 2026-10-09T21:44:36+00:00)
