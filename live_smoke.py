import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared.config_and_utils import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

import requests


def main():
    if not OPENROUTER_API_KEY:
        print("OPENROUTER_API_KEY is not set; aborting live smoke test.")
        return 2
    base = OPENROUTER_BASE_URL.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "google/gemini-3.8-flash",
        "messages": [{"role": "user", "content": "Say hello from a live smoke test."}],
        "provider": {"order": ["google-ai-studio"], "allow_fallbacks": False},
        "max_tokens": 100,
    }
    print(f"POST {url} (model=google/gemini-3.8-flash)")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
    except Exception as exc:
        print("Request error:", exc)
        return 1
    print("HTTP", r.status_code)
    try:
        data = r.json()
    except Exception:
        print("Non-JSON response:\n", r.text)
        return 1
    print(json.dumps(data, indent=2)[:4000])
    try:
        msg = data.get("choices", [])[0].get("message", {}).get("content")
        print("CHOICE MSG:\n", msg)
    except Exception:
        pass
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
