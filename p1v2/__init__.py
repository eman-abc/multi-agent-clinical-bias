"""p1v2 -- the rebuilt behavioural pipeline for Paper 1 (closed models through an API). Standalone: it does not import or modify shared/.

  parse.py       strict final-answer reader with a stored confidence label
  prompts.py     Agent B prompts (the project's own ESI and USMLE templates, verbatim) and the chat messages of each stimulus
  stimuli.py     programmatic stimulus builder: exact minimal pairs, speaker-free / author / peer / consensus recommendations, two channels, counterbalanced wrong options
  provenance.py  run manifest, one folder per step with input, output, notes and SHA-256 hashes, and an append-only call log
"""
