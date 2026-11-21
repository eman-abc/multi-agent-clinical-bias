# Step 07_agent_b_GPT_6_Luna

**Purpose.** Agent B = openai/gpt-6-luna: 63952 calls (1 draw(s) per stimulus, temperature 0.0); strict answer parsing; resumable

## Input
- {'kind': 'file', 'path': 'p1v2_data\\stimuli_luna_p0.jsonl', 'sha256': 'ed3fc872407f868e158cdf67dcaea35d7f7abdec52e1facf0807978026c8a807', 'n_lines': 65608}

## Output
- {'path': 'output.jsonl', 'sha256': 'ee7bfe701b765859e431bd1e7275efcad424bf0eaf3b608de797b76217d35e09', 'n_lines': 47066}

## Notes and decisions
- 47066 of 63952 planned calls completed; unparseable answers: 1376; transport failures: 0
- ABORTED by a guardrail/routing error: Error code: 403 - {'error': {'message': 'API key budget limit exceeded (monthly limit). Contact your org admin.', 'code': 403}}

**Status.** ok (11665.28 s, started 2026-10-09T23:06:46+00:00)
