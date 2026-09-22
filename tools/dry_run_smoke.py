import sys
from pathlib import Path

# Ensure repo root is on sys.path when run from tools/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.data_loader import load_amqa_dataset
from shared.conditions import ConditionSpec, Attribute, ConditionLabel, CarrierAuthoring
from shared.crew_pipeline import generate_note

try:
    records = load_amqa_dataset(dry_run=True, mock_n=1)
    case = records[0]
    spec = ConditionSpec(
        case_id=case.question_id,
        attribute=Attribute.SES,
        condition_label=ConditionLabel.NEUTRAL,
        signal_type=None,
        group=None,
        carrier_type=None,
        carrier_value=None,
        carrier_placement_hint="n/a",
        authoring=CarrierAuthoring.NONE,
    )
    note = generate_note(case, spec, dry_run=True)
    print("OK: generated note length", len(note.note_text))
    print(note.note_text[:400])
except Exception as e:
    print("ERROR during dry-run smoke test:", e)
    raise
