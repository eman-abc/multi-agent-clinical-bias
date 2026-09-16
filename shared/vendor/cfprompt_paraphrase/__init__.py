"""
shared/vendor/cfprompt_paraphrase/

Vendored subset of cfprompt (github.com/redagavin/cfprompt, MIT license),
the companion package to "Compared to What? Baselines and Metrics for
Counterfactual Prompting" (Yang et al., arXiv:2605.01048).

Only `paraphrase.py` and `tokenization.py` are vendored -- the two files
needed for `generate_adjusted_paraphrase()`, not cfprompt's full `Study`/
`Model` machinery. See each file's header for the full attribution and
vendoring rationale, and phase_0_complete_v3_MODULE4_ADDENDUM.md section A
for the integration decision record.

LICENSE (MIT, cfprompt's original) is included in this directory.
"""

from .paraphrase import AdjustedParaphraseResult, generate_adjusted_paraphrase, is_refusal
from .tokenization import token_edit_distance, token_edit_distance_pct

__all__ = [
    "AdjustedParaphraseResult",
    "generate_adjusted_paraphrase",
    "is_refusal",
    "token_edit_distance",
    "token_edit_distance_pct",
]
