import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shared.config_and_utils import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from shared.data_loader import load_amqa_dataset
from shared.conditions import ConditionSpec, Attribute, ConditionLabel, CarrierAuthoring
from shared.crew_pipeline import generate_note
from shared import inference_engine as ie


def main():
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY is not set; aborting live pipeline test.")
        return 2

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

    print("Requesting live generation (Agent A -> Gemini via OpenRouter)...")
    note = generate_note(case, spec, dry_run=False)
    if note.blocked_reason:
        print("Generation blocked:", note.blocked_reason)
        return 1
    print("Generated note (trim):", note.note_text[:600])

    print("Calling DeepSeek-V4-Flash (Agent B) for ESI classification via OpenRouter...")
    extra_body = {"provider": {"order": ["azure"], "allow_fallbacks": False}}
    esi_result = ie.run_esi_triage_chat_completions(note.note_text, model="deepseek/deepseek-v4-flash", base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY, extra_body=extra_body)
    print("ESI result:", esi_result.top_class, esi_result.class_probabilities)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
