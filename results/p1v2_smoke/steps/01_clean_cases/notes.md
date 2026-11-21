# Step 01_clean_cases

**Purpose.** split each question into vignette and stem without cutting numbers; flag image-dependent and no-vignette cases; decide per-attribute eligibility

## Input
- {'kind': 'file', 'path': 'p1v2_data\\cases20.jsonl', 'sha256': '528ac59181ac5948f37f5cb8b5a48ef9ad20e13ecdcb38c3edc04444caa3ffb1', 'n_lines': 20}

## Output
- {'path': 'output.jsonl', 'sha256': 'ac6d49965c4ddd0b7aa41933971d4162fcad690364a998478ac8b9fa63b108ae', 'n_lines': 20}

## Notes and decisions
- 20 cases; excluded 0; ineligible sex/race/ses = 2/0/1

**Status.** ok (0.03 s, started 2026-10-09T21:05:42+00:00)
